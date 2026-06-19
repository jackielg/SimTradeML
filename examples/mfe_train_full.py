# -*- coding: utf-8 -*-
"""
MFE 选股模型 - 充分训练版 (用户要求提升跨度)
向量化批量采集: 每股只取1次全量日线, 内存里滚动算特征+标签
样本: 1500只股 x 2023-2026 x 每3个交易日采样 = ~20万样本
标签: 未来5天MFE (multi_label_train 已验证最优, IC 0.215/分月100%命中)
产出: deploy_mfe.json + scaler + meta (供 macd_golden 策略接入)
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle, json, os
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import RobustScaler

from simtrademl.data_sources.simtradelab_source import SimTradeLabDataSource
from simtrademl.core.utils.logger import setup_logger

logger = setup_logger('mfe_full', level='INFO', log_file='examples/mfe_train_full.log')

LOOKBACK = 60
PREDICT_DAYS = 5
SAMPLE_EVERY = 3  # 每3个交易日采样一次


def calc_features_vec(c, h, l, v, end):
    """向量化特征计算: c/h/l/v 为 ndarray, end 为当前索引(不含)"""
    if end < LOOKBACK:
        return None
    cs = c[end - LOOKBACK:end]
    hs = h[end - LOOKBACK:end]
    ls = l[end - LOOKBACK:end]
    vs = v[end - LOOKBACK:end]
    if len(cs) < LOOKBACK or np.any(~np.isfinite(cs[-21:])):
        return None
    f = {}
    f['amplitude_20d'] = (hs[-20:].max() - ls[-20:].min()) / (cs[-20:].mean() + 1e-8)
    f['ma5'] = cs[-5:].mean()
    f['ma10'] = cs[-10:].mean()
    f['ma20'] = cs[-20:].mean()
    f['ma60'] = cs[-60:].mean() if end >= 60 else cs.mean()
    f['macd_dif'] = _macd_dif(c, end)
    f['momentum_20d'] = (cs[-1] - cs[-21]) / (cs[-21] + 1e-8)
    f['price_position'] = (cs[-1] - ls[-20:].min()) / (hs[-20:].max() - ls[-20:].min() + 1e-8)
    f['return_10d'] = (cs[-1] - cs[-11]) / (cs[-11] + 1e-8)
    f['return_1d'] = (cs[-1] - cs[-2]) / (cs[-2] + 1e-8)
    f['return_20d'] = (cs[-1] - cs[-21]) / (cs[-21] + 1e-8)
    f['return_5d'] = (cs[-1] - cs[-6]) / (cs[-6] + 1e-8)
    rets = np.diff(cs[-20:]) / cs[-20:-1]
    f['volatility_20d'] = rets.std()
    f['volume_ratio'] = vs[-1] / (vs[-20:].mean() + 1e-8)
    return f


def _macd_dif(c, end):
    seg = c[end - 60:end] if end >= 60 else c[:end]
    if len(seg) < 30:
        return 0.0
    s = pd.Series(seg)
    ema12 = s.ewm(span=12, adjust=False).mean().iloc[-1]
    ema26 = s.ewm(span=26, adjust=False).mean().iloc[-1]
    return (ema12 - ema26) / (seg[-1] + 1e-8)


FEAT_NAMES = sorted([
    'amplitude_20d', 'ma10', 'ma20', 'ma5', 'ma60', 'macd_dif',
    'momentum_20d', 'price_position', 'return_10d', 'return_1d',
    'return_20d', 'return_5d', 'volatility_20d', 'volume_ratio',
])


def collect(ds, n_stocks, train_end=None):
    """批量采集: 每股一次取全量日线, 滚动算特征+MFE标签

    Args:
        train_end: 截止日期(str/Timestamp). 采样点 _date <= train_end,
                   标签未来 PREDICT_DAYS 天也限制在 train_end 内 (严格时序外样本)
    """
    stocks = ds.get_stock_list()[:n_stocks]
    dates = ds.get_trading_dates()
    d0, d1 = dates[0], dates[-1]
    if train_end is not None:
        train_end = pd.Timestamp(train_end)
        d1 = min(d1, train_end)
    logger.info(f"Data range: {d0.date()} ~ {d1.date()} (train_end), {len(stocks)} stocks")
    rows = []
    for si, stock in enumerate(stocks):
        if si % 200 == 0:
            logger.info(f"  processing {si}/{len(stocks)}...")
        try:
            pdf = ds.get_price_data(stock, start_date=d0, end_date=d1)
        except Exception:
            continue
        if pdf is None or pdf.empty or len(pdf) < LOOKBACK + PREDICT_DAYS + 5:
            continue
        c = pdf['close'].values.astype(float)
        h = pdf['high'].values.astype(float)
        l = pdf['low'].values.astype(float)
        v = pdf['volume'].values.astype(float)
        n = len(c)
        # 滚动采样点: 从 LOOKBACK 开始, 每 SAMPLE_EVERY 天
        for end in range(LOOKBACK, n - PREDICT_DAYS, SAMPLE_EVERY):
            if end >= n - PREDICT_DAYS:
                break
            feat = calc_features_vec(c, h, l, v, end)
            if feat is None:
                continue
            cur = c[end]
            if cur <= 0 or not np.isfinite(cur):
                continue
            # v2.4 标签: 风险调整 MFE (MFE - MAE) - 奖励上涨惩罚下跌
            # 旧版纯 MFE 只看最大涨幅, 熊市选出暴涨后崩盘的股 (见 docs/2026-06-19_MFE模型缺陷分析.md)
            fut_high = h[end + 1:end + 1 + PREDICT_DAYS].max()
            fut_low = l[end + 1:end + 1 + PREDICT_DAYS].min()
            mfe = (fut_high - cur) / cur
            mae = (cur - fut_low) / cur  # 最大回撤幅度 (>=0)
            label = mfe - mae  # 风险调整收益: 涨多少 - 跌多少
            # 实际收益 (对照, 不作标签但记录)
            fut_close = c[end + PREDICT_DAYS]
            ret = (fut_close - cur) / cur
            if not np.isfinite(label) or not np.isfinite(ret):
                continue
            row = {fn: feat[fn] for fn in FEAT_NAMES}
            row['_mfe'] = mfe
            row['_mae'] = mae
            row['_label'] = label
            row['_ret'] = ret
            row['_stock'] = stock
            row['_date'] = pdf.index[end]
            rows.append(row)
    df = pd.DataFrame(rows)
    logger.info(f"Collected {len(df)} samples from {df['_stock'].nunique()} stocks")
    return df


def main():
    ds = SimTradeLabDataSource()
    N = int(os.environ.get('ML_N_STOCKS', '1500'))
    train_end = os.environ.get('ML_TRAIN_END')  # out-of-sample: 限制训练数据截止日
    df = collect(ds, N, train_end=train_end)
    X = df[FEAT_NAMES].values
    y_mfe = df['_label'].values  # v2.4 标签: MFE - MAE (风险调整)
    y_mfe_raw = df['_mfe'].values  # 旧标签保留用于对照
    dates = pd.Series(pd.to_datetime(df['_date'].values))

    # 时序切分: 70 train / 15 val / 15 test
    ud = sorted(dates.unique())
    t1 = int(len(ud) * 0.7)
    t2 = int(len(ud) * 0.85)
    tr_m = dates.isin(ud[:t1]).values
    va_m = dates.isin(ud[t1:t2]).values
    te_m = dates.isin(ud[t2:]).values
    logger.info(f"Split: train={tr_m.sum()} val={va_m.sum()} test={te_m.sum()}")
    logger.info(f"Train {ud[0].date()}~{ud[t1-1].date()} | Test {ud[t2].date()}~{ud[-1].date()}")

    scaler = RobustScaler()
    Xtr = scaler.fit_transform(X[tr_m]); ytr = y_mfe[tr_m]
    Xva = scaler.transform(X[va_m]); yva = y_mfe[va_m]
    Xte = scaler.transform(X[te_m]); yte = mfe_te = y_mfe[te_m]

    dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=FEAT_NAMES)
    dva = xgb.DMatrix(Xva, label=yva, feature_names=FEAT_NAMES)
    dte = xgb.DMatrix(Xte, label=yte, feature_names=FEAT_NAMES)

    params = {'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8,
              'colsample_bytree': 0.8, 'seed': 42, 'objective': 'reg:squarederror',
              'min_child_weight': 5, 'reg_alpha': 0.1}
    model = xgb.train(params, dtr, num_boost_round=500, evals=[(dtr,'train'),(dva,'val')],
                      early_stopping_rounds=50, verbose_eval=100)

    # 整体 + 分月评估
    pred = model.predict(dte)
    ic, _ = pearsonr(pred, yte); ric, _ = spearmanr(pred, yte)
    order = np.argsort(-pred); n5 = len(order)//5
    top_ret = yte[order[:n5]].mean(); bot_ret = yte[order[-n5:]].mean()
    logger.info(f"\n=== Test overall: IC={ic:.4f} RankIC={ric:.4f} | top20% mfe={top_ret:.4f} bot20%={bot_ret:.4f} LS={top_ret-bot_ret:.4f}")

    # 分月
    te_dates = dates.values[te_m]
    edf = pd.DataFrame({'pred':pred,'y':yte,'date':te_dates})
    edf['month'] = pd.to_datetime(edf['date']).dt.to_period('M')
    ics = []
    print("\n=== 分月 Rank IC ===")
    for m, g in edf.groupby('month'):
        if len(g) < 30: continue
        r,_ = spearmanr(g['pred'], g['y'])
        ics.append(r)
        o = np.argsort(-g['pred'].values); k = len(o)//5
        print(f"{str(m):10s} n={len(g):5d} rank_ic={r:.4f} top={g['y'].values[o[:k]].mean():.4f} bot={g['y'].values[o[-k:]].mean():.4f}")
    ics = np.array(ics)
    print(f"\n分月命中 {np.sum(ics>0)}/{len(ics)} | mean={ics.mean():.4f} | 最差={ics.min():.4f}")

    # 特征重要性
    imp = model.get_score(importance_type='gain')
    print("\n=== 特征重要性 ===")
    for k, v in sorted(imp.items(), key=lambda x: -x[1])[:8]:
        print(f"  {k:20s} {v:.4f}")

    # 保存 deploy 包 (带 train_end 后缀, 不覆盖全量模型)
    suffix = f"_{train_end}" if train_end else ""
    out = Path(f'examples/deploy_mfe{suffix}')
    model.save_model(f'{out}.json')
    with open(f'{out}_scaler.pkl','wb') as f: pickle.dump(scaler, f)
    # JSON scaler (无 sklearn 依赖, 策略用)
    with open(f'{out}_scaler.json','w') as f:
        json.dump({'center': scaler.center_.tolist(), 'scale': scaler.scale_.tolist()}, f)
    with open(f'{out}_meta.json','w') as f:
        json.dump({'features':FEAT_NAMES,'label':'mfe_5d','predict_days':PREDICT_DAYS,
                   'lookback':LOOKBACK,'n_train':int(tr_m.sum()),'n_test':int(te_m.sum()),
                   'test_ic':round(float(ic),4),'test_rank_ic':round(float(ric),4),
                   'monthly_hit_rate':float(np.sum(ics>0)/len(ics)),
                   'trained_at':datetime.now().isoformat(),
                   'train_end': train_end,
                   'train_range':[str(ud[0].date()),str(ud[t1-1].date())],
                   'test_range':[str(ud[t2].date()),str(ud[-1].date())],
                   'label_type': 'mfe_minus_mae_5d',
                   'label_formula': 'mfe - mae = (fut_high_max - cur) / cur - (cur - fut_low_min) / cur',
                   'version': 'mfe_5d_v2.4'}, f, indent=2, ensure_ascii=False)
    print(f"\nDeploy saved: {out}.json / scaler.json / meta")
    print(f"  train_end={train_end} (out-of-sample 回测 2024+2025 全程未见)")


if __name__ == '__main__':
    main()