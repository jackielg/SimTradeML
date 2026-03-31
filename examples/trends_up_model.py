# -*- coding: utf-8 -*-
"""
trends_up 策略模型训练示例
"""

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import RobustScaler

from simtrademl.core.models import ModelMetadata, PTradeModelPackage, create_model_id
from simtrademl.core.utils.logger import setup_logger
from simtrademl.data_sources.simtradelab_source import SimTradeLabDataSource

logger = setup_logger(
    "trends_up_model", level="INFO", log_file="examples/trends_up_model.log"
)


def safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    """安全计算比值。"""
    if denominator is None or abs(denominator) < 1e-12:
        return default
    return float(numerator / denominator)


def calculate_rsi(closes: np.ndarray, period: int = 14) -> float:
    """计算 RSI。"""
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1) :])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def calculate_atr_ratio(price_df: pd.DataFrame, period: int = 14) -> float:
    """计算 ATR 占收盘价比例。"""
    if len(price_df) < period + 1:
        return 0.0
    high = price_df["high"]
    low = price_df["low"]
    close = price_df["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(period).mean().iloc[-1]
    last_close = close.iloc[-1]
    return safe_ratio(float(atr), float(last_close))


def resolve_index(df: pd.DataFrame, target_date: pd.Timestamp) -> Optional[int]:
    """按交易日解析索引位置。"""
    if df.empty:
        return None
    position = int(df.index.searchsorted(target_date, side="right") - 1)
    if position < 0:
        return None
    return position


def calculate_trends_up_features(
    hist_df: pd.DataFrame,
    benchmark_hist_df: pd.DataFrame,
) -> Optional[Dict[str, float]]:
    """计算贴合 trends_up 策略的特征。"""
    if len(hist_df) < 120 or len(benchmark_hist_df) < 120:
        return None

    closes = hist_df["close"].to_numpy(dtype=float)
    highs = hist_df["high"].to_numpy(dtype=float)
    lows = hist_df["low"].to_numpy(dtype=float)
    volumes = hist_df["volume"].to_numpy(dtype=float)
    benchmark_closes = benchmark_hist_df["close"].to_numpy(dtype=float)

    ma5 = float(np.mean(closes[-5:]))
    ma20 = float(np.mean(closes[-20:]))
    ma60 = float(np.mean(closes[-60:]))
    ma120 = float(np.mean(closes[-120:]))
    close = float(closes[-1])
    high_20 = float(np.max(highs[-20:]))
    high_60 = float(np.max(highs[-60:]))
    low_20 = float(np.min(lows[-20:]))
    low_60 = float(np.min(lows[-60:]))
    avg_vol_5 = float(np.mean(volumes[-5:]))
    avg_vol_20 = float(np.mean(volumes[-20:]))
    avg_vol_60 = float(np.mean(volumes[-60:]))
    ma20_prev = float(np.mean(closes[-25:-5]))
    ma60_prev = float(np.mean(closes[-70:-10]))

    stock_return_20d = safe_ratio(close - float(closes[-21]), float(closes[-21]))
    stock_return_60d = safe_ratio(close - float(closes[-61]), float(closes[-61]))
    benchmark_return_20d = safe_ratio(
        float(benchmark_closes[-1] - benchmark_closes[-21]),
        float(benchmark_closes[-21]),
    )
    benchmark_return_60d = safe_ratio(
        float(benchmark_closes[-1] - benchmark_closes[-61]),
        float(benchmark_closes[-61]),
    )

    returns_20d = np.diff(closes[-21:]) / closes[-21:-1]

    features = {
        "atr_ratio_14": calculate_atr_ratio(hist_df, 14),
        "breakout_distance_20d": safe_ratio(close, high_20) - 1.0,
        "breakout_distance_60d": safe_ratio(close, high_60) - 1.0,
        "close_to_ma20": safe_ratio(close, ma20) - 1.0,
        "close_to_ma60": safe_ratio(close, ma60) - 1.0,
        "ma20_slope_5d": safe_ratio(ma20, ma20_prev) - 1.0,
        "ma20_to_ma60": safe_ratio(ma20, ma60) - 1.0,
        "ma5_to_ma20": safe_ratio(ma5, ma20) - 1.0,
        "ma60_slope_10d": safe_ratio(ma60, ma60_prev) - 1.0,
        "ma60_to_ma120": safe_ratio(ma60, ma120) - 1.0,
        "price_position_20d": safe_ratio(close - low_20, high_20 - low_20, 0.5),
        "price_position_60d": safe_ratio(close - low_60, high_60 - low_60, 0.5),
        "relative_strength_20d": stock_return_20d - benchmark_return_20d,
        "relative_strength_60d": stock_return_60d - benchmark_return_60d,
        "return_5d": safe_ratio(close - float(closes[-6]), float(closes[-6])),
        "return_10d": safe_ratio(close - float(closes[-11]), float(closes[-11])),
        "return_20d": stock_return_20d,
        "return_60d": stock_return_60d,
        "rsi14": calculate_rsi(closes, 14),
        "volatility_20d": float(np.std(returns_20d)),
        "volume_ratio_20d": safe_ratio(float(volumes[-1]), avg_vol_20, 1.0),
        "volume_ratio_5d": safe_ratio(float(volumes[-1]), avg_vol_5, 1.0),
        "volume_trend_5_20": safe_ratio(avg_vol_5, avg_vol_20) - 1.0,
        "volume_trend_20_60": safe_ratio(avg_vol_20, avg_vol_60) - 1.0,
    }

    if not all(np.isfinite(value) for value in features.values()):
        return None

    return features


def is_candidate_window(features: Dict[str, float]) -> bool:
    """筛选更接近策略观察池的样本窗口。"""
    return all(
        [
            -0.15 <= features["breakout_distance_20d"] <= 0.04,
            features["ma20_slope_5d"] > -0.02,
            features["ma20_to_ma60"] > -0.05,
            features["relative_strength_20d"] > -0.10,
            features["price_position_20d"] > 0.45,
        ]
    )


def calculate_trends_up_label(
    price_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    sample_idx: int,
    benchmark_idx: int,
    predict_days: int,
) -> Optional[Tuple[int, Dict[str, float]]]:
    """构建牛股早发现标签。"""
    if sample_idx < 60 or benchmark_idx + predict_days >= len(benchmark_df):
        return None

    current_close = float(price_df["close"].iloc[sample_idx])
    current_benchmark = float(benchmark_df["close"].iloc[benchmark_idx])
    if current_close <= 0 or current_benchmark <= 0:
        return None

    future_window = price_df.iloc[sample_idx + 1 : sample_idx + predict_days + 1]
    if len(future_window) < predict_days:
        return None

    future_benchmark = benchmark_df.iloc[benchmark_idx + predict_days]
    past_high_20 = float(price_df["close"].iloc[sample_idx - 20 : sample_idx].max())
    future_end_close = float(future_window["close"].iloc[-1])
    future_max_close = float(future_window["close"].max())
    future_min_close = float(future_window["close"].min())

    future_return = safe_ratio(future_end_close - current_close, current_close)
    future_max_return = safe_ratio(future_max_close - current_close, current_close)
    future_min_return = safe_ratio(future_min_close - current_close, current_close)
    benchmark_return = safe_ratio(
        float(future_benchmark["close"]) - current_benchmark, current_benchmark
    )
    excess_return = future_return - benchmark_return

    breakout_success = future_max_close >= past_high_20 * 1.01
    label = int(
        breakout_success
        and future_max_return >= 0.08
        and future_return >= 0.03
        and excess_return >= 0.02
        and future_min_return >= -0.08
    )

    return label, {
        "future_return": future_return,
        "future_max_return": future_max_return,
        "future_min_return": future_min_return,
        "benchmark_return": benchmark_return,
        "excess_return": excess_return,
    }


def collect_samples(
    data_source: SimTradeLabDataSource,
    n_stocks: int,
    lookback: int,
    predict_days: int,
    sample_step: int,
    benchmark: str,
) -> Tuple[pd.DataFrame, np.ndarray, pd.Series, pd.DataFrame]:
    """采集 trends_up 模型训练样本。"""
    logger.info("开始采集 trends_up 样本")
    stock_list = data_source.get_stock_list()[:n_stocks]
    benchmark_df = data_source.get_market_data(benchmark)
    if benchmark_df.empty:
        benchmark_df = data_source.api.data_context.benchmark_data.get(
            benchmark, pd.DataFrame()
        )
    if benchmark_df.empty:
        raise RuntimeError(f"无法获取基准数据: {benchmark}")
    samples = []
    labels = []
    sample_dates = []
    sample_meta = []

    logger.info("股票数量: %s", len(stock_list))
    logger.info("基准数据范围: %s -> %s", benchmark_df.index[0], benchmark_df.index[-1])

    for idx, stock in enumerate(stock_list, start=1):
        price_df = data_source.get_price_data(stock)
        if price_df.empty or len(price_df) < lookback + predict_days + 20:
            continue

        max_sample_idx = len(price_df) - predict_days
        sample_range = range(lookback - 1, max_sample_idx, sample_step)

        for sample_idx in sample_range:
            sample_date = price_df.index[sample_idx]
            benchmark_idx = resolve_index(benchmark_df, sample_date)
            if benchmark_idx is None or benchmark_idx < lookback - 1:
                continue

            hist_df = price_df.iloc[sample_idx - lookback + 1 : sample_idx + 1]
            benchmark_hist_df = benchmark_df.iloc[
                benchmark_idx - lookback + 1 : benchmark_idx + 1
            ]
            if len(hist_df) < lookback or len(benchmark_hist_df) < lookback:
                continue

            features = calculate_trends_up_features(hist_df, benchmark_hist_df)
            if features is None or not is_candidate_window(features):
                continue

            label_result = calculate_trends_up_label(
                price_df,
                benchmark_df,
                sample_idx,
                benchmark_idx,
                predict_days,
            )
            if label_result is None:
                continue

            label, extra = label_result
            samples.append(features)
            labels.append(label)
            sample_dates.append(sample_date)
            sample_meta.append(
                {
                    "stock": stock,
                    "sample_date": sample_date.isoformat(),
                    **extra,
                }
            )

        if idx % 20 == 0:
            logger.info("已处理股票: %s/%s", idx, len(stock_list))

    if not samples:
        raise RuntimeError("未采集到有效样本，请放宽候选条件或增加股票数量")

    X = pd.DataFrame(samples)
    X = X[sorted(X.columns)]
    y = np.asarray(labels, dtype=int)
    dates = pd.Series(pd.to_datetime(sample_dates))
    meta_df = pd.DataFrame(sample_meta)

    logger.info("样本数: %s", len(X))
    logger.info("正样本占比: %.2f%%", y.mean() * 100)
    logger.info("特征数: %s", X.shape[1])

    return X, y, dates, meta_df


def split_by_time(
    X: pd.DataFrame,
    y: np.ndarray,
    sample_dates: pd.Series,
    meta_df: pd.DataFrame,
) -> Dict[str, object]:
    """按时间切分训练集、验证集和测试集。"""
    unique_dates = sorted(sample_dates.dt.normalize().unique())
    train_end = int(len(unique_dates) * 0.7)
    val_end = int(len(unique_dates) * 0.85)

    train_dates = unique_dates[:train_end]
    val_dates = unique_dates[train_end:val_end]
    test_dates = unique_dates[val_end:]

    masks = {
        "train": sample_dates.dt.normalize().isin(train_dates).to_numpy(),
        "val": sample_dates.dt.normalize().isin(val_dates).to_numpy(),
        "test": sample_dates.dt.normalize().isin(test_dates).to_numpy(),
    }

    return {
        "X_train": X.loc[masks["train"]].reset_index(drop=True),
        "y_train": y[masks["train"]],
        "meta_train": meta_df.loc[masks["train"]].reset_index(drop=True),
        "X_val": X.loc[masks["val"]].reset_index(drop=True),
        "y_val": y[masks["val"]],
        "meta_val": meta_df.loc[masks["val"]].reset_index(drop=True),
        "X_test": X.loc[masks["test"]].reset_index(drop=True),
        "y_test": y[masks["test"]],
        "meta_test": meta_df.loc[masks["test"]].reset_index(drop=True),
        "train_dates": train_dates,
        "test_dates": test_dates,
    }


def evaluate_classifier(
    model: xgb.Booster,
    dmatrix: xgb.DMatrix,
    y_true: np.ndarray,
    meta_df: pd.DataFrame,
    dataset_name: str,
) -> Dict[str, float]:
    """评估分类模型表现。"""
    probabilities = model.predict(dmatrix)
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
    }

    if len(np.unique(y_true)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, probabilities))
    else:
        metrics["auc"] = 0.5

    top_count = max(1, int(len(probabilities) * 0.1))
    top_indices = np.argsort(probabilities)[-top_count:]
    top_meta = meta_df.iloc[top_indices]

    metrics["top10_hit_rate"] = float(np.mean(y_true[top_indices]))
    metrics["top10_avg_future_return"] = float(top_meta["future_return"].mean())
    metrics["top10_avg_excess_return"] = float(top_meta["excess_return"].mean())

    logger.info("%s AUC: %.4f", dataset_name, metrics["auc"])
    logger.info(
        "%s Average Precision: %.4f", dataset_name, metrics["average_precision"]
    )
    logger.info("%s Accuracy: %.4f", dataset_name, metrics["accuracy"])
    logger.info("%s Top10 命中率: %.4f", dataset_name, metrics["top10_hit_rate"])
    logger.info(
        "%s Top10 平均未来收益: %.4f", dataset_name, metrics["top10_avg_future_return"]
    )
    logger.info(
        "%s Top10 平均超额收益: %.4f", dataset_name, metrics["top10_avg_excess_return"]
    )

    return metrics


def train_model(
    X: pd.DataFrame,
    y: np.ndarray,
    sample_dates: pd.Series,
    meta_df: pd.DataFrame,
) -> Tuple[xgb.Booster, RobustScaler, Dict[str, float], pd.DataFrame]:
    """训练 trends_up 分类模型。"""
    split_data = split_by_time(X, y, sample_dates, meta_df)
    X_train = split_data["X_train"]
    X_val = split_data["X_val"]
    X_test = split_data["X_test"]
    y_train = split_data["y_train"]
    y_val = split_data["y_val"]
    y_test = split_data["y_test"]
    meta_test = split_data["meta_test"]

    logger.info(
        "训练集: %s, 验证集: %s, 测试集: %s", len(X_train), len(X_val), len(X_test)
    )

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    positive = max(1, int(np.sum(y_train == 1)))
    negative = max(1, int(np.sum(y_train == 0)))

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "logloss"],
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "lambda": 1.0,
        "seed": 42,
        "scale_pos_weight": negative / positive,
    }

    dtrain = xgb.DMatrix(X_train_scaled, label=y_train, feature_names=list(X.columns))
    dval = xgb.DMatrix(X_val_scaled, label=y_val, feature_names=list(X.columns))
    dtest = xgb.DMatrix(X_test_scaled, label=y_test, feature_names=list(X.columns))

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=400,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=40,
        verbose_eval=False,
    )

    logger.info("最优迭代轮次: %s", model.best_iteration)

    val_metrics = evaluate_classifier(
        model, dval, y_val, split_data["meta_val"], "验证集"
    )
    test_metrics = evaluate_classifier(model, dtest, y_test, meta_test, "测试集")

    test_probabilities = model.predict(dtest)
    ranked_test = X_test.copy()
    ranked_test["prediction"] = test_probabilities
    ranked_test["label"] = y_test
    ranked_test["stock"] = meta_test["stock"].values
    ranked_test["sample_date"] = meta_test["sample_date"].values
    ranked_test = ranked_test.sort_values("prediction", ascending=False).reset_index(
        drop=True
    )

    metrics = {
        **{f"val_{key}": value for key, value in val_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }

    return model, scaler, metrics, ranked_test


def save_artifacts(
    model: xgb.Booster,
    scaler: RobustScaler,
    feature_names: list,
    metrics: Dict[str, float],
    sample_dates: pd.Series,
    ranked_test: pd.DataFrame,
) -> None:
    """保存模型产物。"""
    output_dir = Path("examples")
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = ModelMetadata(
        model_id=create_model_id(prefix="trends_up"),
        version="1.0",
        created_at=datetime.now().isoformat(),
        model_type="xgboost",
        model_library_version=xgb.__version__,
        features=feature_names,
        n_features=len(feature_names),
        scaler_type="RobustScaler",
        train_start_date=str(sample_dates.min().date()),
        train_end_date=str(sample_dates.max().date()),
        n_samples=len(sample_dates),
        hyperparameters={
            "objective": "binary:logistic",
            "target": "未来10日牛股启动概率",
            "label_definition": "10日内突破近20日高点、未来收益为正、相对沪深300有超额且最大回撤受控",
        },
        description="面向 trends_up 策略观察池和买前过滤的牛股早发现概率模型",
        tags=["trends_up", "breakout", "early_discovery", "classification"],
    )

    for name, value in metrics.items():
        metadata.add_metric(name, value)

    package_path = output_dir / "trends_up_model.ptp"
    json_model_path = output_dir / "trends_up_model.json"
    scaler_path = output_dir / "trends_up_model_scaler.pkl"
    features_path = output_dir / "trends_up_model_features.json"
    metadata_path = output_dir / "trends_up_model_metadata.json"
    report_path = output_dir / "trends_up_model_report.json"
    sample_path = output_dir / "trends_up_model_sample.json"

    package = PTradeModelPackage(model=model, scaler=scaler, metadata=metadata)
    package.save(str(package_path))
    model.save_model(str(json_model_path))

    with scaler_path.open("wb") as file:
        pickle.dump(scaler, file)

    with features_path.open("w", encoding="utf-8") as file:
        json.dump(feature_names, file, ensure_ascii=False, indent=2)

    metadata.add_file("package", package_path.name)
    metadata.add_file("model", json_model_path.name)
    metadata.add_file("scaler", scaler_path.name)
    metadata.add_file("features", features_path.name)
    metadata.save(str(metadata_path))

    top_sample = ranked_test.iloc[0]
    sample_features = {
        feature_name: float(top_sample[feature_name]) for feature_name in feature_names
    }
    with sample_path.open("w", encoding="utf-8") as file:
        json.dump(sample_features, file, ensure_ascii=False, indent=2)

    report = {
        "model_id": metadata.model_id,
        "created_at": metadata.created_at,
        "metrics": metrics,
        "top_test_sample": {
            "stock": top_sample["stock"],
            "sample_date": top_sample["sample_date"],
            "prediction": float(top_sample["prediction"]),
            "label": int(top_sample["label"]),
        },
        "files": {
            "package": package_path.name,
            "model": json_model_path.name,
            "scaler": scaler_path.name,
            "features": features_path.name,
            "metadata": metadata_path.name,
            "sample": sample_path.name,
        },
    }
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    logger.info("\n%s", metadata.summary())
    logger.info("已保存产物: %s", report["files"])


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""
    parser = argparse.ArgumentParser(
        description="训练 trends_up 策略专用的牛股早发现模型"
    )
    parser.add_argument("--n-stocks", type=int, default=160, help="训练使用的股票数量")
    parser.add_argument("--lookback", type=int, default=120, help="特征回看窗口")
    parser.add_argument("--predict-days", type=int, default=10, help="标签预测窗口")
    parser.add_argument("--sample-step", type=int, default=7, help="采样步长")
    parser.add_argument("--benchmark", type=str, default="000300.SS", help="基准代码")
    return parser


def main() -> None:
    """执行训练流程。"""
    args = build_parser().parse_args()

    logger.info("=" * 60)
    logger.info("trends_up 模型训练开始")
    logger.info("=" * 60)

    data_source = SimTradeLabDataSource()
    X, y, sample_dates, meta_df = collect_samples(
        data_source=data_source,
        n_stocks=args.n_stocks,
        lookback=args.lookback,
        predict_days=args.predict_days,
        sample_step=args.sample_step,
        benchmark=args.benchmark,
    )

    model, scaler, metrics, ranked_test = train_model(X, y, sample_dates, meta_df)
    save_artifacts(
        model=model,
        scaler=scaler,
        feature_names=list(X.columns),
        metrics=metrics,
        sample_dates=sample_dates,
        ranked_test=ranked_test,
    )

    logger.info("=" * 60)
    logger.info("trends_up 模型训练完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
