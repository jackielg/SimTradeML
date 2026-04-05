# SimTradeML

[English](./README.md) | 中文 | [Deutsch](./README_DE.md)

**PTrade 兼容的量化交易 ML 框架**，帮助用户快速训练出可在 SimTradeLab 和 Ptrade 中使用的预测模型。

## 核心定位

SimTradeML 是 [SimTradeLab](https://github.com/kay-ou/SimTradeLab) 的 **机器学习工具链**：
- 🎯 **专为 PTrade 优化**：训练产出的模型可直接在 SimTradeLab 回测, Ptrade 实盘使用
- ⚡ **快速训练**：5分钟从数据到可用模型
- 📊 **量化金融指标**：IC/ICIR/分位收益等专业评估
- 🔧 **A 股生态集成**：深度绑定 SimTradeLab 数据源

## 快速开始

### 安装

```bash
cd /path/to/SimTradeML
python3 -m poetry lock
python3 -m poetry install
```

如果本地使用同级目录中的 SimTradeLab 源码与数据，建议安装到 Poetry 虚拟环境：

```bash
python3 -m poetry run pip install -e ../SimTradeLab
export SIMTRADELAB_DATA_PATH="/path/to/SimTradeLab/data"
```

### 5分钟训练第一个模型

```bash
# 1. 准备数据（复制 SimTradeData 的 parquet 数据到 data/ 目录）
mkdir -p data
cp -r /path/to/SimTradeData/data/* data/

# 2. 运行完整训练流程
python3 -m poetry run python examples/mvp_train.py
```

### 完整示例

参考 `examples/` 目录：
- **mvp_train.py** - 完整训练流程（数据收集、训练、导出）
- **complete_example.py** - 推荐用法演示（JSON-only 契约）
- **validate_ptrade_deploy.py** - 验证 `model.json + metadata.json + scaler.json` 加载链路

### 部署验证

训练完成后，可用示例特征文件验证 PTrade 风格部署链路：

```bash
python3 -m poetry run python examples/validate_ptrade_deploy.py
```

如需只验证单一路径：

```bash
python3 -m poetry run python examples/validate_ptrade_deploy.py --mode json
```

### 推荐用法（JSON-only）

```python
import json
import xgboost as xgb
import numpy as np

# 训练后保存
model.save_model('my_model.json')
with open('my_model_scaler.json', 'w', encoding='utf-8') as f:
    json.dump({'center': [...], 'scale': [...]}, f, ensure_ascii=False, indent=2)

# PTrade / SimTradeLab 中加载和预测
booster = xgb.Booster()
booster.load_model('my_model.json')
dmatrix = xgb.DMatrix(np.array([[...]], dtype=float))
prediction = booster.predict(dmatrix)[0]
```

## 核心特性

### PTrade 兼容性
- ✅ **XGBoost 1.7.4**：与当前 PTrade 环境对齐
- ✅ **JSON-only 运行契约**：模型、metadata、特征顺序、scaler 参数均可文本审计
- ✅ **即插即用**：训练的模型可直接在 SimTradeLab 中使用

### ML 能力
- **数据源抽象**：轻松切换不同数据源
- **特征工程**：内置技术指标，支持自定义
- **评估指标**：IC/ICIR/分位收益/方向准确率
- **并行处理**：自动多进程采样加速

### 量化金融特化
- **时间序列严谨性**：防止未来数据泄露
- **日度再平衡**：模拟真实交易场景
- **分位数收益**：策略收益模拟
- **方向准确率**：涨跌判断评估

## 项目结构

```
src/simtrademl/
├── core/
│   ├── data/          # 数据层（DataSource, DataCollector）
│   └── utils/         # 工具（Config, Logger, Metrics）
└── data_sources/      # 数据源实现
    └── simtradelab_source.py

examples/
└── mvp_train.py       # 完整训练示例
```

## API 文档

### 配置管理

```python
from simtrademl import Config

# 从字典创建
config = Config.from_dict({'data': {'lookback_days': 60}})

# 从 YAML 加载
config = Config.from_yaml('config.yml')

# 点号访问
lookback = config.get('data.lookback_days', default=30)
config.set('model.type', 'xgboost')
```

### 数据收集

```python
from simtrademl.core.data.collector import DataCollector

collector = DataCollector(data_source, config)

# 收集所有股票
X, y, dates = collector.collect()

# 过滤股票
X, y, dates = collector.collect(
    stock_filter=lambda s: s.startswith('60')
)

# 自定义特征
def custom_features(stock, price_df, idx, date, ds):
    return {'my_feature': price_df['close'].iloc[idx-1]}

collector = DataCollector(data_source, config,
                          feature_calculator=custom_features)
```

### 评估指标

```python
from simtrademl import (
    calculate_ic, calculate_rank_ic, calculate_icir,
    calculate_quantile_returns, calculate_direction_accuracy
)

# IC 指标
ic, p_value = calculate_ic(predictions, actuals)
rank_ic, p_value = calculate_rank_ic(predictions, actuals)
icir, ic_std = calculate_icir(predictions, actuals)

# 分位收益（日度再平衡）
quantile_returns, long_short = calculate_quantile_returns(
    predictions, actuals, dates=sample_dates
)

# 方向准确率
accuracy = calculate_direction_accuracy(predictions, actuals)
```

## 测试

```bash
# 运行所有测试
poetry run pytest

# 查看覆盖率
poetry run pytest --cov=simtrademl --cov-report=html
open htmlcov/index.html
```

## 配置示例

完整配置（`config.yml`）:

```yaml
data:
  lookback_days: 60
  predict_days: 5
  sampling_window_days: 15

model:
  type: xgboost
  params:
    max_depth: 4
    learning_rate: 0.04
    subsample: 0.7
    colsample_bytree: 0.7

training:
  train_ratio: 0.70
  val_ratio: 0.15
  parallel_jobs: -1  # -1 = 使用所有 CPU
```

## 扩展数据源

```python
from simtrademl.core.data.base import DataSource

class MyDataSource(DataSource):
    def get_stock_list(self) -> List[str]:
        return ['600519.SS', '000858.SZ']

    def get_price_data(self, stock, start_date, end_date, fields):
        # 返回 DataFrame，index 为日期
        return pd.DataFrame({
            'open': [...], 'high': [...], 'low': [...],
            'close': [...], 'volume': [...]
        })

    # 实现其他必需方法...
```

## 依赖

**核心**: Python 3.9+, numpy, pandas, scikit-learn, **xgboost 1.7.4** (与当前 PTrade 环境对齐)
**可选**: simtradelab (数据), optuna (超参优化), mlflow (实验追踪)

> ⚠️ **重要**：真实 PTrade 环境当前安装的是 XGBoost 1.7.4，本项目默认按该版本对齐。

## PTrade 集成说明

### 模型导出格式
推荐在策略运行侧统一使用 JSON-only 契约：

```python
import xgboost as xgb
import json

model = xgb.train(params, dtrain, ...)

# 方式1: 模型 JSON（推荐，人类可读）
model.save_model('my_model.json')

# 方式2: 配套 scaler / metadata / features 也保存为 JSON
with open('my_model_scaler.json', 'w', encoding='utf-8') as f:
    json.dump({'center': [...], 'scale': [...]}, f, ensure_ascii=False, indent=2)
```

### 在 SimTradeLab 中使用
```python
# 方式1: 加载 JSON-only 模型
import xgboost as xgb
model = xgb.Booster(model_file='my_model.json')

# 预测
features = [...]
dmatrix = xgb.DMatrix([features])
prediction = model.predict(dmatrix)[0]
```

### 特征一致性
确保训练和推理时使用相同的特征顺序：
```python
# 训练时记录特征顺序
feature_names = ['ma5', 'ma10', 'rsi14', ...]

# 推理时按相同顺序构造特征
features = [ma5, ma10, rsi14, ...]  # 顺序必须一致
```

## 开发计划

### 当前版本 (v0.2.0) - MVP ✅
- [x] SimTradeLab 数据源集成
- [x] XGBoost 1.7.4 训练流程
- [x] 量化金融评估指标
- [x] 并行数据收集

### 下一阶段 (v0.3.0) - PTrade 增强 🚧
- [ ] **模型元数据系统** (P0) - 特征一致性保证
- [ ] **统一模型导出器** (P0) - 一键生成 PTrade 模型包
- [ ] **特征注册表** (P0) - 特征复用和版本管理
- [ ] **快速训练管道** (P1) - 简化训练流程

详见 [TODO.md](TODO.md)

## 许可证

MIT License

---

**文档**: 参考 `examples/mvp_train.py` 获取完整示例
**问题**: 提交 Issue 到 GitHub
**测试覆盖率**: 88% | 66 个测试全部通过
