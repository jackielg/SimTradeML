# -*- coding: utf-8 -*-
"""
模型自进化系统 - 每周一自动进化模型

功能：
1. 收集上周实战数据
2. 分析失败案例（假阳性、假阴性）
3. 构建分层训练数据集
4. 训练新模型并验证改进
5. 部署或回退

版本：v2.1
日期：2026-04-02
"""

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import RobustScaler

from simtrademl.core.utils.logger import setup_logger

logger = setup_logger(
    "model_evolution", 
    level="INFO", 
    log_file="examples/model_evolution.log"
)


# ============================================================================
# 配置常量
# ============================================================================

# 样例股票池
SAMPLE_STOCKS = [
    "002815.SZ",  # 崇达技术
    "300476.SZ",  # 胜宏科技
    "301232.SZ",  # 飞沃科技
    "301377.SZ",  # 鼎泰高科
    "000975.SZ",  # 银泰黄金
    "603226.SH",  # 菲林格尔
]

# 分层权重
LAYER_WEIGHTS = {
    "base": 1.0,           # 基础层权重
    "core": 3.0,           # 核心层权重（样例股）
    "reinforcement": 5.0,  # 强化层权重（失败案例）
    "fresh": 2.0,          # 新鲜层权重（上周数据）
}

# 模型改进验证阈值
IMPROVEMENT_THRESHOLDS = {
    "auc_min_improvement": -0.05,      # AUC 最小允许下降幅度
    "hit_rate_min_improvement": -0.10, # Top10 命中率最小允许下降幅度
    "sample_stock_recall_min": 0.8,    # 样例股召回率最低要求
}

# 模型产物目录
MODEL_ARTIFACTS_DIR = Path("examples")
MODEL_BACKUP_DIR = Path("examples/model_backups")


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class FailureCase:
    """失败案例"""
    stock: str                        # 股票代码
    date: pd.Timestamp                # 日期
    prediction: float                 # 预测值
    actual_label: int                 # 实际标签
    failure_type: str                 # 失败类型（假阳性/假阴性）
    features: Dict[str, float]        # 特征
    composite_score: float = 0.0      # 综合评分
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "stock": self.stock,
            "date": str(self.date),
            "prediction": self.prediction,
            "actual_label": self.actual_label,
            "failure_type": self.failure_type,
            "features": self.features,
            "composite_score": self.composite_score,
        }


@dataclass
class LayeredDataset:
    """分层数据集"""
    base_layer: Optional[pd.DataFrame] = None           # 基础层
    core_layer: Optional[pd.DataFrame] = None           # 核心层
    reinforcement_layer: Optional[pd.DataFrame] = None  # 强化层
    fresh_layer: Optional[pd.DataFrame] = None          # 新鲜层
    
    def get_all_data(self) -> pd.DataFrame:
        """获取加权合并后的数据集"""
        layers = []
        weights = []
        
        if self.base_layer is not None and not self.base_layer.empty:
            layers.append(self.base_layer)
            weights.extend([LAYER_WEIGHTS["base"]] * len(self.base_layer))
        
        if self.core_layer is not None and not self.core_layer.empty:
            layers.append(self.core_layer)
            weights.extend([LAYER_WEIGHTS["core"]] * len(self.core_layer))
        
        if self.reinforcement_layer is not None and not self.reinforcement_layer.empty:
            layers.append(self.reinforcement_layer)
            weights.extend([LAYER_WEIGHTS["reinforcement"]] * len(self.reinforcement_layer))
        
        if self.fresh_layer is not None and not self.fresh_layer.empty:
            layers.append(self.fresh_layer)
            weights.extend([LAYER_WEIGHTS["fresh"]] * len(self.fresh_layer))
        
        if not layers:
            return pd.DataFrame()
        
        # 通过重复采样实现权重
        weighted_dfs = []
        for df, weight in zip(layers, weights):
            n_repeats = int(np.ceil(weight))
            weighted_df = df.sample(n=len(df) * n_repeats, replace=True, random_state=42)
            weighted_dfs.append(weighted_df)
        
        return pd.concat(weighted_dfs, ignore_index=True)


@dataclass
class EvolutionResult:
    """进化结果"""
    success: bool                       # 是否成功
    new_model_path: Optional[str] = None  # 新模型路径
    improvement_metrics: Dict[str, float] = field(default_factory=dict)  # 改进指标
    rollback_reason: Optional[str] = None  # 回退原因
    message: str = ""                   # 消息
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "new_model_path": self.new_model_path,
            "improvement_metrics": self.improvement_metrics,
            "rollback_reason": self.rollback_reason,
            "message": self.message,
        }


# ============================================================================
# 模型自进化系统
# ============================================================================

class ModelEvolutionSystem:
    """模型自进化系统"""
    
    def __init__(
        self,
        model_dir: Path = MODEL_ARTIFACTS_DIR,
        backup_dir: Path = MODEL_BACKUP_DIR,
    ):
        """
        初始化自进化系统
        
        Args:
            model_dir: 模型产物目录
            backup_dir: 模型备份目录
        """
        self.model_dir = model_dir
        self.backup_dir = backup_dir
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_model: Optional[xgb.Booster] = None
        self.current_scaler: Optional[RobustScaler] = None
        self.current_metadata: Optional[Dict[str, Any]] = None
    
    def load_current_model(self, model_version: Optional[str] = None):
        """
        加载当前模型
        
        Args:
            model_version: 模型版本号，默认加载最新版
        """
        logger.info("开始加载当前模型")
        
        if model_version is None:
            metadata_path = self.model_dir / "trends_up_model_v2_metadata.json"
        else:
            metadata_path = self.model_dir / f"trends_up_model_v2_{model_version}_metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"未找到模型元数据文件：{metadata_path}")
        
        with metadata_path.open("r", encoding="utf-8") as f:
            self.current_metadata = json.load(f)
        
        model_path = self.model_dir / self.current_metadata["files"]["model"]
        scaler_path = self.model_dir / self.current_metadata["files"]["scaler"]
        
        self.current_model = xgb.Booster()
        self.current_model.load_model(str(model_path))
        
        with scaler_path.open("r", encoding="utf-8") as f:
            scaler_data = json.load(f)
        self.current_scaler = RobustScaler()
        self.current_scaler.center_ = np.array(scaler_data["center"])
        self.current_scaler.scale_ = np.array(scaler_data["scale"])
        
        logger.info("模型加载成功：%s", self.current_metadata["model_id"])
    
    def analyze_failures(
        self,
        predictions_df: pd.DataFrame,
        actual_labels_df: pd.DataFrame,
    ) -> List[FailureCase]:
        """
        分析失败案例
        
        Args:
            predictions_df: 预测结果 DataFrame
            actual_labels_df: 实际标签 DataFrame
        
        Returns:
            失败案例列表
        """
        logger.info("开始分析失败案例")
        
        failures = []
        
        for idx in range(len(predictions_df)):
            pred_row = predictions_df.iloc[idx]
            actual_row = actual_labels_df.iloc[idx] if idx < len(actual_labels_df) else None
            
            if actual_row is None:
                continue
            
            prediction = float(pred_row["prediction"])
            actual_label = int(actual_row["bull_stock_label"])
            predicted_label = 1 if prediction >= 0.5 else 0
            
            if predicted_label != actual_label:
                failure_type = (
                    "false_positive" if predicted_label > actual_label 
                    else "false_negative"
                )
                
                features = {
                    col: float(pred_row[col]) 
                    for col in predictions_df.columns 
                    if col not in ["prediction", "label", "stock", "sample_date"]
                }
                
                failure = FailureCase(
                    stock=pred_row.get("stock", "UNKNOWN"),
                    date=pd.Timestamp(pred_row.get("sample_date", datetime.now())),
                    prediction=prediction,
                    actual_label=actual_label,
                    failure_type=failure_type,
                    features=features,
                    composite_score=float(actual_row.get("composite_score", 0.0)),
                )
                failures.append(failure)
        
        logger.info(
            "分析完成：共发现 %s 个失败案例（假阳性：%s, 假阴性：%s）",
            len(failures),
            sum(1 for f in failures if f.failure_type == "false_positive"),
            sum(1 for f in failures if f.failure_type == "false_negative"),
        )
        
        return failures
    
    def collect_sample_stocks(
        self,
        data_source: Any,
        lookback: int = 120,
        predict_days: int = 10,
    ) -> pd.DataFrame:
        """
        采集样例牛股样本
        
        Args:
            data_source: 数据源对象
            lookback: 回看天数
            predict_days: 预测天数
        
        Returns:
            样例股数据 DataFrame
        """
        logger.info("开始采集样例牛股样本：%s", SAMPLE_STOCKS)
        
        from trends_up_model_v2 import (
            MultiDimensionalLabelCalculator,
            calculate_trends_up_features,
            is_candidate_window,
            resolve_index,
        )
        
        samples = []
        label_calculator = MultiDimensionalLabelCalculator()
        
        benchmark_df = data_source.get_market_data("000300.SS")
        if benchmark_df.empty:
            raise RuntimeError("无法获取基准数据")
        
        for stock in SAMPLE_STOCKS:
            price_df = data_source.get_price_data(stock)
            if price_df.empty or len(price_df) < lookback + predict_days + 20:
                logger.warning("股票 %s 数据不足，跳过", stock)
                continue
            
            max_sample_idx = len(price_df) - predict_days
            sample_range = range(lookback - 1, max_sample_idx, 7)
            
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
                
                sample_data = {
                    "stock": stock,
                    "sample_date": sample_date,
                    **features,
                    **label_result.to_dict(),
                }
                samples.append(sample_data)
        
        if not samples:
            raise RuntimeError("未采集到样例股样本")
        
        df = pd.DataFrame(samples)
        logger.info("样例股样本数：%s", len(df))
        
        return df
    
    def collect_failure_cases(
        self,
        failures: List[FailureCase],
    ) -> pd.DataFrame:
        """
        采集失败案例数据
        
        Args:
            failures: 失败案例列表
        
        Returns:
            失败案例 DataFrame
        """
        logger.info("开始构建失败案例数据集")
        
        if not failures:
            return pd.DataFrame()
        
        records = []
        for failure in failures:
            record = {
                "stock": failure.stock,
                "sample_date": failure.date,
                **failure.features,
                "bull_stock_label": failure.actual_label,
                "composite_score": failure.composite_score,
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        logger.info("失败案例数据集：%s 条记录", len(df))
        
        return df
    
    def collect_fresh_data(
        self,
        data_source: Any,
        last_week_start: pd.Timestamp,
        last_week_end: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        采集上周新鲜数据
        
        Args:
            data_source: 数据源对象
            last_week_start: 上周开始日期
            last_week_end: 上周结束日期
        
        Returns:
            新鲜数据 DataFrame
        """
        logger.info(
            "开始采集上周新鲜数据：%s - %s", 
            last_week_start.date(), last_week_end.date()
        )
        
        from trends_up_model_v2 import (
            MultiDimensionalLabelCalculator,
            calculate_trends_up_features,
            is_candidate_window,
            resolve_index,
        )
        
        samples = []
        label_calculator = MultiDimensionalLabelCalculator()
        benchmark_df = data_source.get_market_data("000300.SS")
        
        stock_list = data_source.get_stock_list()[:100]
        
        for stock in stock_list:
            price_df = data_source.get_price_data(stock)
            if price_df.empty:
                continue
            
            mask = (price_df.index >= last_week_start) & (price_df.index <= last_week_end)
            week_df = price_df.loc[mask]
            if len(week_df) < 10:
                continue
            
            for sample_idx in range(120, len(price_df), 7):
                sample_date = price_df.index[sample_idx]
                if sample_date < last_week_start or sample_date > last_week_end:
                    continue
                
                benchmark_idx = resolve_index(benchmark_df, sample_date)
                if benchmark_idx is None:
                    continue
                
                hist_df = price_df.iloc[sample_idx - 119 : sample_idx + 1]
                benchmark_hist_df = benchmark_df.iloc[
                    benchmark_idx - 119 : benchmark_idx + 1
                ]
                if len(hist_df) < 120 or len(benchmark_hist_df) < 120:
                    continue
                
                features = calculate_trends_up_features(hist_df, benchmark_hist_df)
                if features is None or not is_candidate_window(features):
                    continue
                
                label_result = label_calculator.calculate_multi_dimensional_label(
                    price_df,
                    benchmark_df,
                    sample_idx,
                    benchmark_idx,
                    10,
                )
                if label_result is None:
                    continue
                
                sample_data = {
                    "stock": stock,
                    "sample_date": sample_date,
                    **features,
                    **label_result.to_dict(),
                }
                samples.append(sample_data)
        
        if not samples:
            return pd.DataFrame()
        
        df = pd.DataFrame(samples)
        logger.info("新鲜数据：%s 条记录", len(df))
        
        return df
    
    def build_layered_dataset(
        self,
        base_df: pd.DataFrame,
        core_df: pd.DataFrame,
        reinforcement_df: pd.DataFrame,
        fresh_df: pd.DataFrame,
    ) -> LayeredDataset:
        """
        构建分层训练数据集
        
        Args:
            base_df: 基础层数据
            core_df: 核心层数据（样例股）
            reinforcement_df: 强化层数据（失败案例）
            fresh_df: 新鲜层数据（上周）
        
        Returns:
            LayeredDataset 对象
        """
        logger.info("开始构建分层训练数据集")
        
        feature_cols = sorted([
            col for col in base_df.columns 
            if col not in ["stock", "sample_date", "bull_stock_label", "composite_score"]
        ])
        
        def prepare_layer(df: pd.DataFrame) -> pd.DataFrame:
            """准备数据集层"""
            if df is None or df.empty:
                return pd.DataFrame()
            
            df = df.copy()
            df = df[feature_cols + ["bull_stock_label"]]
            df = df.dropna()
            return df
        
        dataset = LayeredDataset(
            base_layer=prepare_layer(base_df),
            core_layer=prepare_layer(core_df),
            reinforcement_layer=prepare_layer(reinforcement_df),
            fresh_layer=prepare_layer(fresh_df),
        )
        
        merged_df = dataset.get_all_data()
        logger.info(
            "分层数据集构建完成：总计 %s 条记录",
            len(merged_df),
        )
        
        return dataset
    
    def verify_improvement(
        self,
        old_model: xgb.Booster,
        new_model: xgb.Booster,
        test_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        验证模型改进
        
        Args:
            old_model: 旧模型
            new_model: 新模型
            test_df: 测试数据集
        
        Returns:
            改进指标字典
        """
        logger.info("开始验证模型改进")
        
        from sklearn.metrics import roc_auc_score, accuracy_score
        
        feature_cols = sorted([
            col for col in test_df.columns 
            if col not in ["stock", "sample_date", "bull_stock_label", "composite_score"]
        ])
        
        X_test = test_df[feature_cols].values
        y_test = test_df["bull_stock_label"].values
        
        dtest = xgb.DMatrix(X_test, feature_names=feature_cols)
        
        old_probs = old_model.predict(dtest)
        new_probs = new_model.predict(dtest)
        
        old_auc = roc_auc_score(y_test, old_probs) if len(np.unique(y_test)) > 1 else 0.5
        new_auc = roc_auc_score(y_test, new_probs) if len(np.unique(y_test)) > 1 else 0.5
        
        old_preds = (old_probs >= 0.5).astype(int)
        new_preds = (new_probs >= 0.5).astype(int)
        
        old_acc = accuracy_score(y_test, old_preds)
        new_acc = accuracy_score(y_test, new_preds)
        
        top_n = max(1, int(len(y_test) * 0.1))
        old_top_idx = np.argsort(old_probs)[-top_n:]
        new_top_idx = np.argsort(new_probs)[-top_n:]
        
        old_hit_rate = np.mean(y_test[old_top_idx])
        new_hit_rate = np.mean(y_test[new_top_idx])
        
        sample_stock_mask = test_df["stock"].isin(SAMPLE_STOCKS)
        if sample_stock_mask.sum() > 0:
            sample_y = y_test[sample_stock_mask]
            sample_probs = new_probs[sample_stock_mask]
            sample_preds = (sample_probs >= 0.5).astype(int)
            sample_recall = sample_y[sample_preds == 1].sum() / max(1, sample_y.sum())
        else:
            sample_recall = 0.0
        
        improvement_metrics = {
            "old_auc": float(old_auc),
            "new_auc": float(new_auc),
            "auc_improvement": float(new_auc - old_auc),
            "old_accuracy": float(old_acc),
            "new_accuracy": float(new_acc),
            "accuracy_improvement": float(new_acc - old_acc),
            "old_top10_hit_rate": float(old_hit_rate),
            "new_top10_hit_rate": float(new_hit_rate),
            "hit_rate_improvement": float(new_hit_rate - old_hit_rate),
            "sample_stock_recall": float(sample_recall),
        }
        
        logger.info("AUC 改进：%.4f -> %.4f (%+.4f)", old_auc, new_auc, new_auc - old_auc)
        logger.info("准确率改进：%.4f -> %.4f (%+.4f)", old_acc, new_acc, new_acc - old_acc)
        logger.info("Top10 命中率改进：%.4f -> %.4f (%+.4f)", old_hit_rate, new_hit_rate, new_hit_rate - old_hit_rate)
        logger.info("样例股召回率：%.4f", sample_recall)
        
        return improvement_metrics
    
    def deploy_new_model(
        self,
        model: xgb.Booster,
        scaler: RobustScaler,
        metadata: Dict[str, Any],
        model_version: str,
    ) -> str:
        """
        部署新模型
        
        Args:
            model: 模型对象
            scaler: 标准化器
            metadata: 元数据
            model_version: 版本号
        
        Returns:
            模型路径
        """
        logger.info("开始部署新模型：%s", model_version)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"trends_up_model_v2_{model_version}_{timestamp}.json"
        model_path = self.model_dir / model_filename
        
        model.save_model(str(model_path))
        
        scaler_filename = f"trends_up_model_v2_{model_version}_{timestamp}_scaler.json"
        scaler_path = self.model_dir / scaler_filename
        
        from trends_up_model_v2 import export_scaler_payload
        scaler_payload = export_scaler_payload(scaler, list(model.get_score().keys()))
        with scaler_path.open("w", encoding="utf-8") as f:
            json.dump(scaler_payload, f, ensure_ascii=False, indent=2)
        
        metadata["files"] = {
            "model": model_filename,
            "scaler": scaler_filename,
        }
        metadata_filename = f"trends_up_model_v2_{model_version}_{timestamp}_metadata.json"
        metadata_path = self.model_dir / metadata_filename
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info("新模型部署成功：%s", model_path)
        
        return str(model_path)
    
    def rollback_to_previous_model(self, reason: str) -> Optional[str]:
        """
        回退到旧模型
        
        Args:
            reason: 回退原因
        
        Returns:
            回退模型路径，如果无可用备份返回 None
        """
        logger.info("开始回退模型，原因：%s", reason)
        
        backup_models = list(self.backup_dir.glob("trends_up_model_v2_*.json"))
        if not backup_models:
            logger.error("无可用备份模型")
            return None
        
        latest_backup = max(backup_models, key=lambda p: p.stat().st_mtime)
        
        logger.info("回退到模型：%s", latest_backup)
        
        return str(latest_backup)
    
    def weekly_evolution(
        self,
        data_source: Any,
        base_df: pd.DataFrame,
        last_week_start: pd.Timestamp,
        last_week_end: pd.Timestamp,
        model_version: str,
    ) -> EvolutionResult:
        """
        执行每周进化流程
        
        Args:
            data_source: 数据源对象
            base_df: 基础数据集
            last_week_start: 上周开始日期
            last_week_end: 上周结束日期
            model_version: 新版本号
        
        Returns:
            进化结果
        """
        logger.info("=" * 60)
        logger.info("开始执行周一模型进化流程")
        logger.info("=" * 60)
        
        try:
            if self.current_model is None:
                self.load_current_model()
            
            sample_df = self.collect_sample_stocks(data_source)
            
            fresh_df = self.collect_fresh_data(
                data_source, last_week_start, last_week_end
            )
            
            predictions_df = pd.DataFrame()
            actual_labels_df = pd.DataFrame()
            failures = self.analyze_failures(predictions_df, actual_labels_df)
            reinforcement_df = self.collect_failure_cases(failures)
            
            layered_dataset = self.build_layered_dataset(
                base_df, sample_df, reinforcement_df, fresh_df
            )
            merged_df = layered_dataset.get_all_data()
            
            if merged_df.empty:
                return EvolutionResult(
                    success=False,
                    rollback_reason="dataset_empty",
                    message="分层数据集为空，无法训练",
                )
            
            from trends_up_model_v2 import train_model
            
            feature_cols = sorted([
                col for col in merged_df.columns 
                if col not in ["stock", "sample_date", "bull_stock_label", "composite_score"]
            ])
            
            X = merged_df[feature_cols]
            y = merged_df["bull_stock_label"].values
            sample_dates = pd.to_datetime(merged_df["sample_date"])
            meta_df = merged_df[["stock", "sample_date"]]
            labels_df = merged_df[["composite_score"]]
            
            new_model, new_scaler, metrics, _ = train_model(
                X, y, sample_dates, meta_df, labels_df
            )
            
            improvement_metrics = self.verify_improvement(
                self.current_model, new_model, merged_df.sample(frac=0.2, random_state=42)
            )
            
            should_deploy = (
                improvement_metrics["auc_improvement"] >= IMPROVEMENT_THRESHOLDS["auc_min_improvement"]
                and improvement_metrics["hit_rate_improvement"] >= IMPROVEMENT_THRESHOLDS["hit_rate_min_improvement"]
                and improvement_metrics["sample_stock_recall"] >= IMPROVEMENT_THRESHOLDS["sample_stock_recall_min"]
            )
            
            if should_deploy:
                model_path = self.deploy_new_model(
                    new_model, new_scaler, 
                    {"version": model_version, "metrics": metrics},
                    model_version
                )
                
                return EvolutionResult(
                    success=True,
                    new_model_path=model_path,
                    improvement_metrics=improvement_metrics,
                    message="模型改进验证通过，已部署新版本",
                )
            else:
                rollback_path = self.rollback_to_previous_model(
                    "模型改进未达阈值"
                )
                
                return EvolutionResult(
                    success=False,
                    new_model_path=rollback_path,
                    improvement_metrics=improvement_metrics,
                    rollback_reason="improvement_below_threshold",
                    message="模型改进未达阈值，已回退到旧版本",
                )
        
        except Exception as e:
            logger.exception("模型进化失败：%s", str(e))
            
            rollback_path = self.rollback_to_previous_model(f"异常：{str(e)}")
            
            return EvolutionResult(
                success=False,
                new_model_path=rollback_path,
                rollback_reason="exception",
                message=f"模型进化异常：{str(e)}",
            )


# ============================================================================
# 命令行接口
# ============================================================================

def main():
    """执行周进化流程示例"""
    logger.info("模型自进化系统示例")
    
    system = ModelEvolutionSystem()
    
    last_monday = datetime.now() - timedelta(days=datetime.now().weekday() + 7)
    last_friday = last_monday + timedelta(days=4)
    
    logger.info("上周时间范围：%s - %s", last_monday.date(), last_friday.date())
    
    logger.info("注意：完整进化流程需要在真实回测环境中运行")
    logger.info("本脚本仅提供架构参考")


if __name__ == "__main__":
    main()
