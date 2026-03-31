# -*- coding: utf-8 -*-
"""
PTrade 部署验证脚本
"""

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Dict

import pandas as pd
import xgboost as xgb

from simtrademl.core.models import ModelMetadata, PTradeModelPackage


def load_features(features_path: Path) -> Dict[str, float]:
    """加载单条预测特征。"""
    with features_path.open("r", encoding="utf-8") as file:
        features = json.load(file)
    if not isinstance(features, dict):
        raise ValueError("特征文件必须是 JSON 字典")
    return features


def predict_with_raw_json(
    model_path: Path,
    metadata_path: Path,
    scaler_path: Path,
    features: Dict[str, float],
) -> float:
    """使用 JSON 模型和原生 XGBoost 接口预测。"""
    model = xgb.Booster(model_file=str(model_path))
    metadata = ModelMetadata.from_json(metadata_path.read_text(encoding="utf-8"))
    metadata.validate_features(list(features.keys()))

    with scaler_path.open("rb") as file:
        scaler = pickle.load(file)

    vector = pd.DataFrame([[features[name] for name in metadata.features]], columns=metadata.features)
    transformed = scaler.transform(vector)
    dmatrix = xgb.DMatrix(transformed, feature_names=metadata.features)
    return float(model.predict(dmatrix)[0])


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="验证 SimTradeML 产物能否按 PTrade 风格完成部署预测")
    parser.add_argument(
        "--mode",
        choices=["all", "ptp", "json", "raw_json"],
        default="all",
        help="验证模式",
    )
    parser.add_argument(
        "--features-json",
        default="examples/ptrade_sample_features.json",
        help="特征 JSON 文件路径",
    )
    parser.add_argument(
        "--ptp-path",
        default="examples/mvp_model.ptp",
        help=".ptp 模型包路径",
    )
    parser.add_argument(
        "--json-model-path",
        default="examples/mvp_model.json",
        help="JSON 模型路径",
    )
    parser.add_argument(
        "--metadata-path",
        default="examples/mvp_metadata.json",
        help="模型元数据路径",
    )
    parser.add_argument(
        "--scaler-path",
        default="examples/mvp_scaler.pkl",
        help="缩放器路径",
    )
    return parser


def main() -> None:
    """执行部署验证。"""
    args = build_parser().parse_args()

    features_path = Path(args.features_json)
    ptp_path = Path(args.ptp_path)
    json_model_path = Path(args.json_model_path)
    metadata_path = Path(args.metadata_path)
    scaler_path = Path(args.scaler_path)

    features = load_features(features_path)
    results = {}

    if args.mode in {"all", "ptp"}:
        package = PTradeModelPackage.load(str(ptp_path))
        results["ptp"] = package.predict(features)

    if args.mode in {"all", "json"}:
        package = PTradeModelPackage.load_from_files(
            str(json_model_path),
            str(metadata_path),
            str(scaler_path),
        )
        results["json_package"] = package.predict(features)

    if args.mode in {"all", "raw_json"}:
        results["raw_json"] = predict_with_raw_json(
            json_model_path,
            metadata_path,
            scaler_path,
            features,
        )

    if len(results) > 1:
        baseline = next(iter(results.values()))
        for name, value in results.items():
            if not math.isclose(baseline, value, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"预测结果不一致: {results}")

    print(json.dumps({
        "features_path": str(features_path),
        "results": results,
        "validated": True,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
