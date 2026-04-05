# -*- coding: utf-8 -*-
"""
trends_up 策略模型训练脚本 v2 - 多维度标签版本

主要改进：
1. 5 只样例牛股配置（主板大牛股）
2. 多维度标签评分系统（突破强度、收益强度、稳定性、持续性）
3. 分层训练数据集支持（基础层、核心层、强化层、新鲜层）
4. 样例股追踪支持

版本：v2.0
日期：2026-04-02
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

from simtrademl.core.models import ModelMetadata, create_model_id
from simtrademl.core.utils.logger import setup_logger
from simtrademl.data_sources.simtradelab_source import SimTradeLabDataSource

logger = setup_logger(
    "trends_up_model_v2", level="INFO", log_file="examples/trends_up_model_v2.log"
)


# ============================================================================
# 配置常量
# ============================================================================

RUNTIME_THRESHOLD_POLICY = {
    "buy_threshold": 0.58,
    "buy_threshold_by_market": {
        "bull": 0.52,
        "weak_bull": 0.55,
        "neutral": 0.58,
        "weak_bear": 0.62,
        "bear": 1.0,
    },
    "watchlist_weight": 0.18,
    "select_weight": 0.12,
}

# 样例股票池：主板大牛股（用户提供的牛股）
# 说明：
# - 002815.SZ: 崇达技术（深圳主板）
# - 300476.SZ: 胜宏科技（创业板）
# - 301232.SZ: 飞沃科技（创业板）
# - 301377.SZ: 鼎泰高科（创业板）
# - 000975.SZ: 银泰黄金（深圳主板）
# - 603226.SH: 菲林格尔（上海主板）
SAMPLE_STOCKS = [
    "002815.SZ",  # 崇达技术
    "300476.SZ",  # 胜宏科技
    "301232.SZ",  # 飞沃科技
    "301377.SZ",  # 鼎泰高科
    "000975.SZ",  # 银泰黄金
    "603226.SH",  # 菲林格尔
]

# 样例股票总开关
USE_SAMPLE_STOCKS = True

# 多维度标签权重配置
LABEL_WEIGHTS = {
    "breakout_strength": 0.30,
    "return_strength": 0.30,
    "stability": 0.20,
    "persistence": 0.20,
}

# 牛股标签阈值
BULL_STOCK_THRESHOLD = 0.6


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class MultiDimensionalLabel:
    """多维度标签数据结构"""
    breakout_strength: float = 0.0  # 突破强度（0-1 分）
    return_strength: float = 0.0    # 收益强度（0-1 分）
    stability: float = 0.0          # 稳定性（0-1 分）
    persistence: float = 0.0        # 持续性（0-1 分）
    composite_score: float = 0.0    # 综合评分
    bull_stock_label: int = 0       # 牛股标签（综合评分>=0.6）
    
    # 原始指标
    future_return: float = 0.0
    future_max_return: float = 0.0
    future_min_return: float = 0.0
    benchmark_return: float = 0.0
    excess_return: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            "breakout_strength": self.breakout_strength,
            "return_strength": self.return_strength,
            "stability": self.stability,
            "persistence": self.persistence,
            "composite_score": self.composite_score,
            "bull_stock_label": float(self.bull_stock_label),
            "future_return": self.future_return,
            "future_max_return": self.future_max_return,
            "future_min_return": self.future_min_return,
            "benchmark_return": self.benchmark_return,
            "excess_return": self.excess_return,
        }


# ============================================================================
# 工具函数
# ============================================================================

def configure_stdio():
    """配置标准输出编码。"""
    for stream in (sys.stdout, sys.stderr):
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            continue


def configure_training_runtime():
    """配置训练阶段运行环境。"""
    configure_stdio()
    os.environ["PTRADE_MULTIPROCESSING"] = "false"


def safe_ratio(
    numerator: float, 
    denominator: float, 
    default: float = 0.0
) -> float:
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


def resolve_index(
    df: pd.DataFrame, 
    target_date: pd.Timestamp
) -> Optional[int]:
    """按交易日解析索引位置。"""
    if df.empty:
        return None
    position = int(df.index.searchsorted(target_date, side="right") - 1)
    if position < 0:
        return None
    return position


# ============================================================================
# 特征计算
# ============================================================================

def calculate_trends_up_features(
    hist_df: pd.DataFrame,
    benchmark_hist_df: pd.DataFrame,
) -> Optional[Dict[str, float]]:
    """
    计算贴合 trends_up 策略的特征。
    
    Args:
        hist_df: 个股历史数据
        benchmark_hist_df: 基准指数历史数据
    
    Returns:
        特征字典，如果计算失败返回 None
    """
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
    """
    筛选更接近策略观察池的样本窗口。
    
    Args:
        features: 特征字典
    
    Returns:
        是否为候选窗口
    """
    return all(
        [
            -0.15 <= features["breakout_distance_20d"] <= 0.04,
            features["ma20_slope_5d"] > -0.02,
            features["ma20_to_ma60"] > -0.05,
            features["relative_strength_20d"] > -0.10,
            features["price_position_20d"] > 0.45,
        ]
    )


# ============================================================================
# 多维度标签计算
# ============================================================================

class MultiDimensionalLabelCalculator:
    """多维度标签计算器"""
    
    @staticmethod
    def calculate_breakout_strength(
        future_max_return: float,
        breakout_success: bool,
        past_high_20: float,
        current_close: float
    ) -> float:
        """
        计算突破强度评分（0-1 分）
        
        评分标准：
        - 突破成功且涨幅>15%: 1.0 分
        - 突破成功且涨幅>10%: 0.8 分
        - 突破成功且涨幅>5%: 0.6 分
        - 突破成功但涨幅<5%: 0.4 分
        - 未突破但涨幅>3%: 0.2 分
        - 其他：0.0 分
        """
        if not breakout_success:
            if future_max_return > 0.03:
                return 0.2
            return 0.0
        
        if future_max_return > 0.15:
            return 1.0
        elif future_max_return > 0.10:
            return 0.8
        elif future_max_return > 0.05:
            return 0.6
        else:
            return 0.4
    
    @staticmethod
    def calculate_return_strength(
        future_return: float,
        benchmark_return: float,
        excess_return: float
    ) -> float:
        """
        计算收益强度评分（0-1 分）
        
        评分标准：
        - 未来收益>15% 且超额>5%: 1.0 分
        - 未来收益>10% 且超额>3%: 0.8 分
        - 未来收益>5% 且超额>2%: 0.6 分
        - 未来收益>3% 且超额>1%: 0.4 分
        - 未来收益>0%: 0.2 分
        - 其他：0.0 分
        """
        if future_return > 0.15 and excess_return > 0.05:
            return 1.0
        elif future_return > 0.10 and excess_return > 0.03:
            return 0.8
        elif future_return > 0.05 and excess_return > 0.02:
            return 0.6
        elif future_return > 0.03 and excess_return > 0.01:
            return 0.4
        elif future_return > 0.0:
            return 0.2
        return 0.0
    
    @staticmethod
    def calculate_stability(
        future_min_return: float,
        volatility: float,
        max_drawdown: Optional[float] = None
    ) -> float:
        """
        计算稳定性评分（0-1 分）
        
        评分标准：
        - 最大回撤>-3% 且波动率低：1.0 分
        - 最大回撤>-5% 且波动率中：0.8 分
        - 最大回撤>-8% 且波动率可控：0.6 分
        - 最大回撤>-10%: 0.4 分
        - 最大回撤>-15%: 0.2 分
        - 其他：0.0 分
        """
        if max_drawdown is not None:
            if max_drawdown > -0.03 and volatility < 0.02:
                return 1.0
            elif max_drawdown > -0.05 and volatility < 0.03:
                return 0.8
            elif max_drawdown > -0.08:
                return 0.6
            elif max_drawdown > -0.10:
                return 0.4
            elif max_drawdown > -0.15:
                return 0.2
        else:
            if future_min_return > -0.03 and volatility < 0.02:
                return 1.0
            elif future_min_return > -0.05 and volatility < 0.03:
                return 0.8
            elif future_min_return > -0.08:
                return 0.6
            elif future_min_return > -0.10:
                return 0.4
            elif future_min_return > -0.15:
                return 0.2
        return 0.0
    
    @staticmethod
    def calculate_persistence(
        future_window: pd.Series,
        positive_days_ratio: Optional[float] = None,
        trend_consistency: Optional[float] = None
    ) -> float:
        """
        计算持续性评分（0-1 分）
        
        评分标准：
        - 上涨天数占比>80% 且趋势一致性强：1.0 分
        - 上涨天数占比>70% 且趋势一致性中：0.8 分
        - 上涨天数占比>60%: 0.6 分
        - 上涨天数占比>50%: 0.4 分
        - 其他：0.0 分
        """
        if positive_days_ratio is None:
            positive_days_ratio = (future_window > 0).sum() / len(future_window)
        
        if trend_consistency is None:
            returns = future_window.values
            positive_runs = 0
            current_run = 0
            for i, r in enumerate(returns):
                if r > 0:
                    current_run += 1
                    positive_runs = max(positive_runs, current_run)
                else:
                    current_run = 0
            trend_consistency = positive_runs / len(returns) if len(returns) > 0 else 0
        
        if positive_days_ratio > 0.80 and trend_consistency > 0.6:
            return 1.0
        elif positive_days_ratio > 0.70 and trend_consistency > 0.4:
            return 0.8
        elif positive_days_ratio > 0.60:
            return 0.6
        elif positive_days_ratio > 0.50:
            return 0.4
        return 0.0
    
    def calculate_composite_score(
        self,
        breakout_strength: float,
        return_strength: float,
        stability: float,
        persistence: float,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        计算综合评分（加权平均）
        
        Args:
            breakout_strength: 突破强度
            return_strength: 收益强度
            stability: 稳定性
            persistence: 持续性
            weights: 权重配置，默认使用 LABEL_WEIGHTS
        
        Returns:
            综合评分（0-1 分）
        """
        if weights is None:
            weights = LABEL_WEIGHTS
        
        composite = (
            breakout_strength * weights["breakout_strength"] +
            return_strength * weights["return_strength"] +
            stability * weights["stability"] +
            persistence * weights["persistence"]
        )
        
        return float(np.clip(composite, 0.0, 1.0))
    
    def calculate_multi_dimensional_label(
        self,
        price_df: pd.DataFrame,
        benchmark_df: pd.DataFrame,
        sample_idx: int,
        benchmark_idx: int,
        predict_days: int,
    ) -> Optional[MultiDimensionalLabel]:
        """
        计算多维度标签
        
        Args:
            price_df: 个股价格数据
            benchmark_df: 基准指数数据
            sample_idx: 样本索引
            benchmark_idx: 基准索引
            predict_days: 预测天数
        
        Returns:
            MultiDimensionalLabel 对象，如果计算失败返回 None
        """
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
            float(future_benchmark["close"]) - current_benchmark, 
            current_benchmark
        )
        excess_return = future_return - benchmark_return

        breakout_success = future_max_close >= past_high_20 * 1.01
        
        future_returns = future_window["close"].pct_change().dropna()
        volatility = float(future_returns.std()) if len(future_returns) > 0 else 0.0
        max_drawdown = future_min_return

        breakout_strength = self.calculate_breakout_strength(
            future_max_return, breakout_success, past_high_20, current_close
        )
        
        return_strength = self.calculate_return_strength(
            future_return, benchmark_return, excess_return
        )
        
        stability = self.calculate_stability(
            future_min_return, volatility, max_drawdown
        )
        
        persistence = self.calculate_persistence(future_returns)
        
        composite_score = self.calculate_composite_score(
            breakout_strength,
            return_strength,
            stability,
            persistence,
        )
        
        bull_stock_label = int(composite_score >= BULL_STOCK_THRESHOLD)

        return MultiDimensionalLabel(
            breakout_strength=breakout_strength,
            return_strength=return_strength,
            stability=stability,
            persistence=persistence,
            composite_score=composite_score,
            bull_stock_label=bull_stock_label,
            future_return=future_return,
            future_max_return=future_max_return,
            future_min_return=future_min_return,
            benchmark_return=benchmark_return,
            excess_return=excess_return,
        )


# ============================================================================
# 样本采集
# ============================================================================

def collect_samples(
    data_source: SimTradeLabDataSource,
    n_stocks: int,
    lookback: int,
    predict_days: int,
    sample_step: int,
    benchmark: str,
    train_end_date: Optional[pd.Timestamp] = None,
    sample_stocks: Optional[List[str]] = None,
    use_sample_stocks: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    采集 trends_up 模型训练样本（支持多维度标签）
    
    Args:
        data_source: 数据源对象
        n_stocks: 股票数量（当 use_sample_stocks=False 时使用）
        lookback: 回看天数
        predict_days: 预测天数
        sample_step: 采样步长
        benchmark: 基准指数代码
        train_end_date: 训练截止日期
        sample_stocks: 样例股票池
        use_sample_stocks: 是否使用样例股票
    
    Returns:
        (X, y, dates, meta_df, labels_df) 
        特征、标签、日期、元数据、多维度标签
    """
    logger.info("开始采集 trends_up 样本（多维度标签版本）")
    
    if use_sample_stocks and sample_stocks:
        stock_list = sample_stocks
        logger.info("使用样例股票池（5 只牛股）：%s", stock_list)
    else:
        stock_list = data_source.get_stock_list()[:n_stocks]
        logger.info("使用前%s只股票：%s", n_stocks, stock_list[:5])
    
    benchmark_df = data_source.get_market_data(benchmark)
    if benchmark_df.empty:
        benchmark_df = data_source.api.data_context.benchmark_data.get(
            benchmark, pd.DataFrame()
        )
    if benchmark_df.empty:
        raise RuntimeError(f"无法获取基准数据：{benchmark}")
    
    if train_end_date is not None:
        train_end_date = pd.Timestamp(train_end_date).normalize()
        benchmark_df = benchmark_df.loc[benchmark_df.index <= train_end_date]
        if benchmark_df.empty:
            raise RuntimeError(
                f"训练截止日 {train_end_date.date()} 之前无基准数据"
            )
    
    samples = []
    labels = []
    sample_dates = []
    sample_meta = []
    labels_data = []
    
    label_calculator = MultiDimensionalLabelCalculator()

    logger.info("股票数量：%s", len(stock_list))
    logger.info("基准数据范围：%s -> %s", benchmark_df.index[0], benchmark_df.index[-1])

    for idx, stock in enumerate(stock_list, start=1):
        price_df = data_source.get_price_data(stock)
        if train_end_date is not None and not price_df.empty:
            price_df = price_df.loc[price_df.index <= train_end_date]
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

            label_result = label_calculator.calculate_multi_dimensional_label(
                price_df,
                benchmark_df,
                sample_idx,
                benchmark_idx,
                predict_days,
            )
            if label_result is None:
                continue

            samples.append(features)
            labels.append(label_result.bull_stock_label)
            sample_dates.append(sample_date)
            sample_meta.append(
                {
                    "stock": stock,
                    "sample_date": sample_date.isoformat(),
                    "future_return": label_result.future_return,
                    "future_max_return": label_result.future_max_return,
                    "future_min_return": label_result.future_min_return,
                    "benchmark_return": label_result.benchmark_return,
                    "excess_return": label_result.excess_return,
                }
            )
            labels_data.append(label_result.to_dict())

        if idx % 20 == 0:
            logger.info("已处理股票：%s/%s", idx, len(stock_list))

    if not samples:
        raise RuntimeError("未采集到有效样本，请放宽候选条件或增加股票数量")

    X = pd.DataFrame(samples)
    X = X[sorted(X.columns)]
    y = np.asarray(labels, dtype=int)
    dates = pd.Series(pd.to_datetime(sample_dates))
    meta_df = pd.DataFrame(sample_meta)
    labels_df = pd.DataFrame(labels_data)

    logger.info("样本数：%s", len(X))
    logger.info("正样本占比：%.2f%%", y.mean() * 100)
    logger.info("特征数：%s", X.shape[1])
    logger.info(
        "牛股标签占比：%.2f%%", 
        labels_df["bull_stock_label"].mean() * 100
    )

    return X, y, dates, meta_df, labels_df


def split_by_time(
    X: pd.DataFrame,
    y: np.ndarray,
    sample_dates: pd.Series,
    meta_df: pd.DataFrame,
    labels_df: pd.DataFrame,
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
        "labels_train": labels_df.loc[masks["train"]].reset_index(drop=True),
        "X_val": X.loc[masks["val"]].reset_index(drop=True),
        "y_val": y[masks["val"]],
        "meta_val": meta_df.loc[masks["val"]].reset_index(drop=True),
        "labels_val": labels_df.loc[masks["val"]].reset_index(drop=True),
        "X_test": X.loc[masks["test"]].reset_index(drop=True),
        "y_test": y[masks["test"]],
        "meta_test": meta_df.loc[masks["test"]].reset_index(drop=True),
        "labels_test": labels_df.loc[masks["test"]].reset_index(drop=True),
        "train_dates": train_dates,
        "test_dates": test_dates,
    }


# ============================================================================
# 模型训练与评估
# ============================================================================

def evaluate_classifier(
    model: xgb.Booster,
    dmatrix: xgb.DMatrix,
    y_true: np.ndarray,
    meta_df: pd.DataFrame,
    labels_df: pd.DataFrame,
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
    top_labels = labels_df.iloc[top_indices]

    metrics["top10_hit_rate"] = float(np.mean(y_true[top_indices]))
    metrics["top10_avg_future_return"] = float(top_meta["future_return"].mean())
    metrics["top10_avg_excess_return"] = float(top_meta["excess_return"].mean())
    metrics["top10_avg_composite_score"] = float(top_labels["composite_score"].mean())

    logger.info("%s AUC: %.4f", dataset_name, metrics["auc"])
    logger.info("%s Average Precision: %.4f", dataset_name, metrics["average_precision"])
    logger.info("%s Accuracy: %.4f", dataset_name, metrics["accuracy"])
    logger.info("%s Top10 命中率：%.4f", dataset_name, metrics["top10_hit_rate"])
    logger.info(
        "%s Top10 平均未来收益：%.4f", dataset_name, metrics["top10_avg_future_return"]
    )
    logger.info(
        "%s Top10 平均超额收益：%.4f", dataset_name, metrics["top10_avg_excess_return"]
    )
    logger.info(
        "%s Top10 平均综合评分：%.4f", dataset_name, metrics["top10_avg_composite_score"]
    )

    return metrics


def train_model(
    X: pd.DataFrame,
    y: np.ndarray,
    sample_dates: pd.Series,
    meta_df: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> Tuple[xgb.Booster, RobustScaler, Dict[str, float], pd.DataFrame]:
    """训练 trends_up 分类模型（多维度标签版本）。"""
    split_data = split_by_time(X, y, sample_dates, meta_df, labels_df)
    X_train = split_data["X_train"]
    X_val = split_data["X_val"]
    X_test = split_data["X_test"]
    y_train = split_data["y_train"]
    y_val = split_data["y_val"]
    y_test = split_data["y_test"]
    meta_test = split_data["meta_test"]
    labels_test = split_data["labels_test"]

    logger.info(
        "训练集：%s, 验证集：%s, 测试集：%s", 
        len(X_train), len(X_val), len(X_test)
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

    logger.info("最优迭代轮次：%s", model.best_iteration)

    val_metrics = evaluate_classifier(
        model, dval, y_val, split_data["meta_val"], 
        split_data["labels_val"], "验证集"
    )
    test_metrics = evaluate_classifier(
        model, dtest, y_test, meta_test, labels_test, "测试集"
    )

    test_probabilities = model.predict(dtest)
    ranked_test = X_test.copy()
    ranked_test["prediction"] = test_probabilities
    ranked_test["label"] = y_test
    ranked_test["stock"] = meta_test["stock"].values
    ranked_test["sample_date"] = meta_test["sample_date"].values
    ranked_test["composite_score"] = labels_test["composite_score"].values
    ranked_test = ranked_test.sort_values("prediction", ascending=False).reset_index(
        drop=True
    )

    metrics = {
        **{f"val_{key}": value for key, value in val_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }

    return model, scaler, metrics, ranked_test


# ============================================================================
# 模型导出
# ============================================================================

def export_scaler_payload(
    scaler: RobustScaler,
    feature_names: list,
) -> Dict[str, object]:
    """导出 JSON-only scaler 契约。"""
    center = getattr(scaler, "center_", None)
    scale = getattr(scaler, "scale_", None)
    if center is None or scale is None:
        raise RuntimeError("RobustScaler 尚未拟合，无法导出 JSON 契约")
    return {
        "enabled": True,
        "type": "RobustScaler",
        "feature_order": list(feature_names),
        "center": [float(value) for value in center],
        "scale": [float(value) for value in scale],
    }


def save_artifacts(
    model: xgb.Booster,
    scaler: RobustScaler,
    feature_names: list,
    metrics: Dict[str, float],
    sample_dates: pd.Series,
    ranked_test: pd.DataFrame,
    labels_df: pd.DataFrame,
    output_dir: Path,
    model_version: str,
    model_id_prefix: str,
    requested_cutoff_date: Optional[str] = None,
    data_cutoff_date: Optional[str] = None,
) -> Dict[str, object]:
    """保存模型产物。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    actual_cutoff_date = data_cutoff_date or str(sample_dates.max().date())
    requested_cutoff_date = requested_cutoff_date or actual_cutoff_date
    fallback_reason = ""
    if requested_cutoff_date != actual_cutoff_date:
        fallback_reason = "trade_calendar_backtrack"

    metadata = ModelMetadata(
        model_id=create_model_id(prefix=model_id_prefix),
        version=model_version,
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
            "target": "未来 10 日牛股启动概率",
            "label_definition": "多维度评分系统（突破强度、收益强度、稳定性、持续性）",
            "composite_threshold": BULL_STOCK_THRESHOLD,
            "label_weights": LABEL_WEIGHTS,
            "requested_cutoff_date": requested_cutoff_date,
            "data_cutoff_date": actual_cutoff_date,
            "fallback_reason": fallback_reason,
            "thresholds": RUNTIME_THRESHOLD_POLICY,
        },
        description="面向 trends_up 策略观察池和买前过滤的牛股早发现概率模型（多维度标签 v2）",
        tags=[
            "trends_up", 
            "breakout", 
            "early_discovery", 
            "classification",
            "multi_dimensional_label",
            "v2",
        ],
    )

    for name, value in metrics.items():
        metadata.add_metric(name, value)

    json_model_path = output_dir / "trends_up_model_v2.json"
    scaler_path = output_dir / "trends_up_model_v2_scaler.json"
    features_path = output_dir / "trends_up_model_v2_features.json"
    metadata_path = output_dir / "trends_up_model_v2_metadata.json"
    report_path = output_dir / "trends_up_model_v2_report.json"
    sample_path = output_dir / "trends_up_model_v2_sample.json"
    contract_path = output_dir / "trends_up_model_v2_contract.json"

    model.save_model(str(json_model_path))
    scaler_payload = export_scaler_payload(scaler, feature_names)
    with scaler_path.open("w", encoding="utf-8") as file:
        json.dump(scaler_payload, file, ensure_ascii=False, indent=2)

    with features_path.open("w", encoding="utf-8") as file:
        json.dump(feature_names, file, ensure_ascii=False, indent=2)

    metadata.add_file("model", json_model_path.name)
    metadata.add_file("scaler", scaler_path.name)
    metadata.add_file("features", features_path.name)
    metadata.add_file("report", report_path.name)
    metadata.add_file("contract", contract_path.name)
    metadata.add_file("sample", sample_path.name)
    metadata.save(str(metadata_path))

    top_sample = ranked_test.iloc[0]
    sample_features = {
        feature_name: float(top_sample[feature_name]) 
        for feature_name in feature_names
    }
    with sample_path.open("w", encoding="utf-8") as file:
        json.dump(sample_features, file, ensure_ascii=False, indent=2)

    report = {
        "model_id": metadata.model_id,
        "model_version": metadata.version,
        "created_at": metadata.created_at,
        "requested_cutoff_date": requested_cutoff_date,
        "data_cutoff_date": actual_cutoff_date,
        "fallback_reason": fallback_reason,
        "train_date_range": {
            "sample_start": str(sample_dates.min().date()),
            "sample_end": str(sample_dates.max().date()),
        },
        "metrics": metrics,
        "thresholds": RUNTIME_THRESHOLD_POLICY,
        "top_test_sample": {
            "stock": top_sample["stock"],
            "sample_date": top_sample["sample_date"],
            "prediction": float(top_sample["prediction"]),
            "label": int(top_sample["label"]),
            "composite_score": float(top_sample.get("composite_score", 0.0)),
        },
        "files": {
            "model": json_model_path.name,
            "scaler": scaler_path.name,
            "features": features_path.name,
            "metadata": metadata_path.name,
            "contract": contract_path.name,
            "sample": sample_path.name,
        },
    }
    contract = {
        "model_version": metadata.version,
        "model_id": metadata.model_id,
        "requested_cutoff_date": requested_cutoff_date,
        "data_cutoff_date": actual_cutoff_date,
        "fallback_reason": fallback_reason,
        "feature_order": list(feature_names),
        "thresholds": RUNTIME_THRESHOLD_POLICY,
        "files": report["files"],
        "runtime": {
            "predictor": "xgboost_booster_json",
            "scaler": scaler_payload,
        },
    }
    with contract_path.open("w", encoding="utf-8") as file:
        json.dump(contract, file, ensure_ascii=False, indent=2)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    if fallback_reason:
        logger.info(
            "训练截止日发生交易日回溯：requested_cutoff=%s actual_cutoff=%s reason=%s",
            requested_cutoff_date,
            actual_cutoff_date,
            fallback_reason,
        )
    logger.info("\n%s", metadata.summary())
    logger.info("已保存产物：%s", report["files"])
    return report


# ============================================================================
# 命令行接口
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""
    parser = argparse.ArgumentParser(
        description="训练 trends_up 策略专用的牛股早发现模型（多维度标签 v2）"
    )
    parser.add_argument(
        "--n-stocks", 
        type=int, 
        default=160, 
        help="训练使用的股票数量（当不使用样例股票时）"
    )
    parser.add_argument(
        "--lookback", 
        type=int, 
        default=120, 
        help="特征回看窗口"
    )
    parser.add_argument(
        "--predict-days", 
        type=int, 
        default=10, 
        help="标签预测窗口"
    )
    parser.add_argument(
        "--sample-step", 
        type=int, 
        default=7, 
        help="采样步长"
    )
    parser.add_argument(
        "--benchmark", 
        type=str, 
        default="000300.SS", 
        help="基准代码"
    )
    parser.add_argument(
        "--train-end-date",
        type=str,
        default=None,
        help="训练数据截止日，格式 YYYY-MM-DD，默认使用当前可用数据",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="examples",
        help="模型产物输出目录",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="2.0",
        help="写入 metadata 的模型版本号",
    )
    parser.add_argument(
        "--model-id-prefix",
        type=str,
        default="trends_up_v2",
        help="模型 ID 前缀",
    )
    parser.add_argument(
        "--use-sample-stocks",
        type=bool,
        default=True,
        help="是否使用样例股票池",
    )
    return parser


def main() -> None:
    """执行训练流程。"""
    configure_training_runtime()
    args = build_parser().parse_args()
    train_end_date = (
        pd.Timestamp(args.train_end_date).normalize()
        if args.train_end_date
        else None
    )
    output_dir = Path(args.output_dir)

    logger.info("=" * 60)
    logger.info("trends_up 模型训练开始（多维度标签 v2）")
    logger.info("=" * 60)
    if train_end_date is not None:
        logger.info("训练数据截止日：%s", train_end_date.date())
    logger.info("模型输出目录：%s", output_dir)
    logger.info("使用样例股票：%s", args.use_sample_stocks)

    data_source = SimTradeLabDataSource(required_data={"price"})
    requested_cutoff_date = str(train_end_date.date()) if train_end_date is not None else None
    actual_cutoff_ts = train_end_date
    if train_end_date is not None:
        trade_dates = data_source.get_trading_dates(end_date=train_end_date)
        valid_dates = [
            pd.Timestamp(day).normalize() 
            for day in trade_dates 
            if pd.Timestamp(day).normalize() <= train_end_date
        ]
        if valid_dates:
            actual_cutoff_ts = valid_dates[-1]
    actual_cutoff_date = str(actual_cutoff_ts.date()) if actual_cutoff_ts is not None else None
    if requested_cutoff_date and actual_cutoff_date and requested_cutoff_date != actual_cutoff_date:
        logger.info(
            "训练截止日按交易日历回溯：requested_cutoff=%s actual_cutoff=%s reason=trade_calendar_backtrack",
            requested_cutoff_date,
            actual_cutoff_date,
        )
    elif actual_cutoff_date:
        logger.info(
            "训练截止日确认：requested_cutoff=%s actual_cutoff=%s", 
            requested_cutoff_date or actual_cutoff_date, 
            actual_cutoff_date
        )
    
    X, y, sample_dates, meta_df, labels_df = collect_samples(
        data_source=data_source,
        n_stocks=args.n_stocks,
        lookback=args.lookback,
        predict_days=args.predict_days,
        sample_step=args.sample_step,
        benchmark=args.benchmark,
        train_end_date=train_end_date,
        sample_stocks=SAMPLE_STOCKS if args.use_sample_stocks else None,
        use_sample_stocks=args.use_sample_stocks and USE_SAMPLE_STOCKS,
    )

    model, scaler, metrics, ranked_test = train_model(
        X, y, sample_dates, meta_df, labels_df
    )
    report = save_artifacts(
        model=model,
        scaler=scaler,
        feature_names=list(X.columns),
        metrics=metrics,
        sample_dates=sample_dates,
        ranked_test=ranked_test,
        labels_df=labels_df,
        output_dir=output_dir,
        model_version=args.model_version,
        model_id_prefix=args.model_id_prefix,
        requested_cutoff_date=requested_cutoff_date,
        data_cutoff_date=actual_cutoff_date,
    )
    logger.info("模型版本：%s", report["model_version"])
    logger.info("模型 ID: %s", report["model_id"])

    logger.info("=" * 60)
    logger.info("trends_up 模型训练完成（多维度标签 v2）")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
