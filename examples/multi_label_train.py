# -*- coding: utf-8 -*-
"""
多标签对比训练: 探索哪种标签训出的模型最能帮 MACD 选股
3 个标签: 未来5天收益率(回归) / 未来5天涨>3%(分类) / 未来5天MFE(回归)
样本: 500只股, 2022-2024训练 + 2025测试
产出: 对比 IC/方向准确率/分位收益, 选最优标签的模型 .ptp
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from scipy.stats import pearsonr, spearmanr
import json
from pathlib import Path

from simtrademl.data_sources.simtradelab_source import SimTradeLabDataSource
from simtrademl.core.utils.logger import setup_logger

logger = setup_logger('multi_label', level='INFO', log_file='examples/multi_label.log')


def calc_features(price_df, lookback=60):
    """计算特征 (含 valuation-like 衍生, 纯 OHLCV 衍生)"""
    if len(price_df) < lookback:
        return None
    c = price_df['close'].values
    v = price_df['volume'].values
    h = price_df['high'].values
    l = price_df['low'].values
    f = {}
    f['ma5'] = np.mean(c[-5:])
    f['ma10'] = np.mean(c[-10:])
    f['ma20'] = np.mean(c[-20:])
    f['ma60'] = np.mean(c[-60:])
    f['return_1d'] = (c[-1] - c[-2]) / c[-2]
    f['return_5d'] = (c[-1] - c[-6]) / c[-6]
    f['return_10d'] = (c[-1] - c[-11]) / c[-11]
    f['return_20d'] = (c[-1] - c[-21]) / c[-21]
    rets = np.diff(c[-20:]) / c[-20:-1]
    f['volatility_20d'] = np.std(rets)
    f['volume_ratio'] = v[-1] / (np.mean(v[-20:]) + 1e-8)
    f['price_position'] = (c[-1] - np.min(l[-20:])) / (np.max(h[-20:]) - np.min(l[-20:]) + 1e-8)
    # 衍生: MACD 近似 (DIF = EMA12 - EMA26)
    ema12 = pd.Series(c[-60:]).ewm(span=12, adjust=False).mean().iloc[-1]
    ema26 = pd.Series(c[-60:]).ewm(span=26, adjust=False).mean().iloc[-1]
    f['macd_dif'] = (ema12 - ema26) / c[-1]
    # 动量: 20日涨幅排名位置
    f['momentum_20d'] = (c[-1] - c[-21]) / c[-21]
    # 振幅
    f['amplitude_20d'] = (np.max(h[-20:]) - np.min(l[-20:])) / (np.mean(c[-20:]) + 1e-8)
    return f


def collect(ds, n_stocks, lookback=60, predict_days=5):
    """收集样本: 特征 + 3 个候选标签"""
    stocks = ds.get_stock_list()[:n_stocks]
    dates = ds.get_trading_dates()
    sample_dates = dates[lookback + 20 :: 7]  # 每7天采样
    samples, y_ret, y_cls, y_mfe, sdates = [], [], [], [], []
    for sd in sample_dates:
        for stock in stocks:
            try:
                pdf = ds.get_price_data(stock, sd - pd.Timedelta(days=lookback + 30),
                                        sd + pd.Timedelta(days=predict_days + 15))
                if pdf.empty or len(pdf) < lookback:
                    continue
                actual = sd if sd in pdf.index else pdf.index[pdf.index <= sd][-1]
                if pd.isna(actual):
                    continue
                idx = pdf.index.get_loc(actual)
                if idx < lookback or idx + predict_days >= len(pdf):
                    continue
                feat = calc_features(pdf.iloc[idx - lookback:idx], lookback)
                if feat is None:
                    continue
                cur = pdf.iloc[idx]['close']
                fut = pdf.iloc[idx + predict_days]
                if cur <= 0:
                    continue
                future_ret = (fut['close'] - cur) / cur
                # MFE: 未来N天最大涨幅
                window = pdf.iloc[idx + 1:idx + 1 + predict_days]
                mfe = (window['high'].max() - cur) / cur if len(window) else 0
                if not np.isfinite(future_ret) or not np.isfinite(mfe):
                    continue
                samples.append(feat)
                y_ret.append(future_ret)
                y_cls.append(1 if future_ret > 0.03 else 0)  # 涨>3%
                y_mfe.append(mfe)
                sdates.append(actual)
            except Exception:
                continue
    X = pd.DataFrame(samples)
    X = X[sorted(X.columns)]
    logger.info(f"Collected {len(X)} samples, {X.shape[1]} features")
    return X, np.array(y_ret), np.array(y_cls), np.array(y_mfe), pd.Series(sdates)


def split(X, y, dates, train_ratio=0.7, val_ratio=0.15):
    """时序切分: 训练/验证/测试"""
    ud = sorted(dates.unique())
    t1 = int(len(ud) * train_ratio)
    t2 = int(len(ud) * (train_ratio + val_ratio))
    masks = {}
    for name, drange in [('train', ud[:t1]), ('val', ud[t1:t2]), ('test', ud[t2:])]:
        m = dates.isin(drange).values
        masks[name] = m
    return masks


def train_one(X, y, dates, label_name, objective):
    """训练单个标签的模型 + 评估"""
    masks = split(X, y, dates)
    scaler = RobustScaler()
    Xtr = scaler.fit_transform(X.values[masks['train']])
    Xva = scaler.transform(X.values[masks['val']])
    Xte = scaler.transform(X.values[masks['test']])
    ytr, yva, yte = y[masks['train']], y[masks['val']], y[masks['test']]

    params = {'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8,
              'colsample_bytree': 0.8, 'seed': 42, 'objective': objective}
    if objective.startswith('binary'):
        params['eval_metric'] = 'logloss'
    dtr = xgb.DMatrix(Xtr, label=ytr)
    dva = xgb.DMatrix(Xva, label=yva)
    dte = xgb.DMatrix(Xte, label=yte)
    model = xgb.train(params, dtr, num_boost_round=300, evals=[(dtr, 'train'), (dva, 'val')],
                      early_stopping_rounds=40, verbose_eval=False)
    pred = model.predict(dte)
    # 评估
    result = {'label': label_name, 'n_test': len(yte), 'best_round': int(model.best_iteration)}
    if objective.startswith('binary'):
        from sklearn.metrics import accuracy_score, roc_auc_score
        result['test_acc'] = round(float(accuracy_score(yte, pred > 0.5)), 4)
        try:
            result['test_auc'] = round(float(roc_auc_score(yte, pred)), 4)
        except Exception:
            result['test_auc'] = None
        # 分位: 高概率组实际涨>3%占比
        order = np.argsort(-pred)
        top_real = yte[order[:len(order) // 5]].mean()
        bot_real = yte[order[-len(order) // 5:]].mean()
        result['top20pct_hit'] = round(float(top_real), 4)
        result['bot20pct_hit'] = round(float(bot_real), 4)
    else:
        ic, icp = pearsonr(pred, yte)
        ric, ricp = spearmanr(pred, yte)
        result['test_ic'] = round(float(ic), 4)
        result['test_rank_ic'] = round(float(ric), 4)
        # 分位收益: 预测top20% vs bot20% 的实际收益
        order = np.argsort(-pred)
        top_ret = yte[order[:len(order) // 5]].mean()
        bot_ret = yte[order[-len(order) // 5:]].mean()
        result['top20pct_ret'] = round(float(top_ret), 4)
        result['bot20pct_ret'] = round(float(bot_ret), 4)
        result['ls_spread'] = round(float(top_ret - bot_ret), 4)
    # 特征重要性 top5
    imp = model.get_score(importance_type='gain')
    result['top5_features'] = dict(sorted(imp.items(), key=lambda x: -x[1])[:5])
    return model, scaler, result


def main():
    ds = SimTradeLabDataSource()
    X, y_ret, y_cls, y_mfe, dates = collect(ds, n_stocks=500)
    logger.info(f"Date range: {dates.min()} ~ {dates.max()}")
    logger.info(f"Label dist - ret mean={y_ret.mean():.4f} | cls pos%={y_cls.mean():.2%} | mfe mean={y_mfe.mean():.4f}")

    results = []
    # 3 个标签并行训练
    configs = [
        ('return_5d', y_ret, 'reg:squarederror'),
        ('up_3pct', y_cls, 'binary:logistic'),
        ('mfe_5d', y_mfe, 'reg:squarederror'),
    ]
    for name, y, obj in configs:
        logger.info(f"\n=== Training label={name} objective={obj} ===")
        model, scaler, res = train_one(X, y, dates, name, obj)
        results.append(res)
        # 保存
        out = Path(f'examples/model_{name}')
        model.save_model(f'{out}.json')
        import pickle
        with open(f'{out}_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"  Result: {res}")

    # 汇总对比
    print("\n" + "=" * 70)
    print("多标签对比汇总")
    print("=" * 70)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()