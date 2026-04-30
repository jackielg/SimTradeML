# CLAUDE.md — SimTradeML

## Project
ML 框架 v0.2.0，训练 XGBoost 预测模型，打包为 .ptp 文件部署到 SimTradeLab 和 PTrade 实盘。

## Stack
Python >=3.10, Poetry, XGBoost 1.7.4, scikit-learn, src/ layout

## Commands
- Train: `poetry run python examples/mvp_train.py`
- Test all: `poetry run pytest`
- Test by marker: `poetry run pytest -m unit`

## Architecture
- `src/simtrademl/core/data/` → DataSource ABC; `data_sources/simtradelab_source.py` 对接 SimTradeLab
- `src/simtrademl/core/models/` → PTradeModelPackage (.ptp = XGBoost + scaler + metadata)
- `src/simtrademl/features/` → FeatureRegistry (装饰器注册); `technical.py` 注册 32 个技术指标
- `src/simtrademl/core/utils/metrics.py` → IC, Rank IC, ICIR, quantile returns, direction accuracy

## Rules
- XGBoost 版本：本框架 1.7.4 / PTrade 生产 0.90 / macOS ARM64 3.2.0（API 兼容）
- .ptp 文件格式包含模型 + scaler + 元数据，SimTradeLab 和 PTrade 通用
- FeatureRegistry 使用装饰器模式，新增指标需在 technical.py 注册

## Out of Scope
- 不自动推送到远程仓库
