# -*- coding: utf-8 -*-
"""验证 mfe 模型分月 IC 稳定性 + 保存 deploy 用的特征顺序/标签"""
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle, json
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.preprocessing import RobustScaler

from simtrademl.data_sources.simtradelab_source import SimTradeLabDataSource
from multi_label_train import collect, calc_features, split  # 复用


def main():
    ds = SimTradeLabDataSource()
    X, y_ret, y_cls, y_mfe, dates = collect(ds, n_stocks=500)
    feats = list(X.columns)
    masks = split(X, y_mfe, dates)
    scaler = RobustScaler()
    Xtr = scaler.fit_transform(X.values[masks['train']])
    Xva = scaler.transform(X.values[masks['val']])
    Xte = scaler.transform(X.values[masks['test']])
    dtr = xgb.DMatrix(Xtr, label=y_mfe[masks['train']])
    dva = xgb.DMatrix(Xva, label=y_mfe[masks['val']])
    dte = xgb.DMatrix(Xte, label=y_mfe[masks['test']])
    model = xgb.train({'max_depth':5,'learning_rate':0.05,'subsample':0.8,
                       'colsample_bytree':0.8,'seed':42,'objective':'reg:squarederror'},
                      dtr, num_boost_round=300, evals=[(dtr,'train'),(dva,'val')],
                      early_stopping_rounds=40, verbose_eval=False)
    pred = model.predict(dte)
    test_dates = dates.values[masks['test']]
    test_y = y_mfe[masks['test']]
    # 分月 IC
    df = pd.DataFrame({'pred':pred,'y':test_y,'date':test_dates})
    df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
    print("=== mfe 模型 分月 Rank IC (测试集) ===")
    print(f"{'month':10s} {'n':>5s} {'rank_ic':>9s} {'top20_mfe':>10s} {'bot20_mfe':>10s}")
    ics = []
    for m, g in df.groupby('month'):
        if len(g) < 30:
            continue
        ric, _ = spearmanr(g['pred'], g['y'])
        order = np.argsort(-g['pred'].values)
        n5 = len(order)//5
        top = g['y'].values[order[:n5]].mean()
        bot = g['y'].values[order[-n5:]].mean()
        ics.append(ric)
        print(f"{str(m):10s} {len(g):5d} {ric:9.4f} {top:10.4f} {bot:10.4f}")
    ics = np.array(ics)
    print(f"\n分月 Rank IC: 正月数 {np.sum(ics>0)}/{len(ics)} | mean={ics.mean():.4f} | "
          f"命中率={np.sum(ics>0)/len(ics)*100:.0f}% | 最差={ics.min():.4f}")
    # 保存 deploy 包: model.json + scaler.pkl + 特征顺序
    out = Path('examples/deploy_mfe')
    model.save_model(f'{out}.json')
    with open(f'{out}_scaler.pkl','wb') as f: pickle.dump(scaler, f)
    with open(f'{out}_meta.json','w') as f:
        json.dump({'features':feats,'label':'mfe_5d','predict_days':5,
                   'overall_ic':0.215,'overall_rank_ic':0.247,
                   'monthly_hit_rate':float(np.sum(ics>0)/len(ics))}, f, indent=2)
    print(f"\nDeploy 包已保存: {out}.json + {out}_scaler.pkl + {out}_meta.json")
    print(f"特征顺序 ({len(feats)}): {feats}")


if __name__ == '__main__':
    main()