# SimTradeML

English | [中文](./README_CN.md) | [Deutsch](./README_DE.md)

**PTrade-compatible quantitative trading ML framework** - helps users quickly train prediction models for use in SimTradeLab and PTrade.

## Core Positioning

SimTradeML is the **machine learning toolchain** for [SimTradeLab](https://github.com/kay-ou/SimTradeLab):
- 🎯 **Optimized for PTrade**: Trained models can be used directly in SimTradeLab backtesting and PTrade live trading
- ⚡ **Fast Training**: From data to usable model in 5 minutes
- 📊 **Quantitative Finance Metrics**: Professional evaluation with IC/ICIR/quantile returns
- 🔧 **A-share Ecosystem Integration**: Deep integration with SimTradeLab data sources

## Quick Start

### Installation

```bash
cd /path/to/SimTradeML
poetry install
pip install simtradelab  # If you need to use SimTradeLab data source
```

### Train Your First Model in 5 Minutes

```bash
# 1. Prepare data (copy SimTradeLab h5 files to data/ directory)
mkdir -p data
cp /path/to/ptrade_data.h5 data/
cp /path/to/ptrade_fundamentals.h5 data/

# 2. Run complete training pipeline
poetry run python examples/mvp_train.py
```

### Complete Examples

See `examples/` directory:
- **mvp_train.py** - Complete training pipeline (data collection, training, export)
- **complete_example.py** - Recommended usage demonstration (single-file package)

### Recommended Usage (Single-File Package)

```python
from simtrademl.core.models import PTradeModelPackage

# Save after training (one file contains everything)
package = PTradeModelPackage(model=model, scaler=scaler, metadata=metadata)
package.save('my_model.ptp')

# Load and predict in PTrade
package = PTradeModelPackage.load('my_model.ptp')
prediction = package.predict(features_dict)  # Auto validation + scaling
```

## Core Features

### PTrade Compatibility
- ✅ **XGBoost 0.90**: PTrade supported version
- ✅ **Flexible Save Formats**: Supports JSON, Pickle, XGBoost native formats
- ✅ **Plug and Play**: Trained models can be used directly in SimTradeLab

### ML Capabilities
- **Data Source Abstraction**: Easily switch between different data sources
- **Feature Engineering**: Built-in technical indicators, supports custom features
- **Evaluation Metrics**: IC/ICIR/quantile returns/directional accuracy
- **Parallel Processing**: Automatic multi-process sampling acceleration

### Quantitative Finance Specialization
- **Time Series Rigor**: Prevents future data leakage
- **Daily Rebalancing**: Simulates real trading scenarios
- **Quantile Returns**: Strategy return simulation
- **Directional Accuracy**: Up/down judgment evaluation

## Project Structure

```
src/simtrademl/
├── core/
│   ├── data/          # Data layer (DataSource, DataCollector)
│   └── utils/         # Utilities (Config, Logger, Metrics)
└── data_sources/      # Data source implementations
    └── simtradelab_source.py

examples/
└── mvp_train.py       # Complete training example
```

## API Documentation

### Configuration Management

```python
from simtrademl import Config

# Create from dictionary
config = Config.from_dict({'data': {'lookback_days': 60}})

# Load from YAML
config = Config.from_yaml('config.yml')

# Dot notation access
lookback = config.get('data.lookback_days', default=30)
config.set('model.type', 'xgboost')
```

### Data Collection

```python
from simtrademl.core.data.collector import DataCollector

collector = DataCollector(data_source, config)

# Collect all stocks
X, y, dates = collector.collect()

# Filter stocks
X, y, dates = collector.collect(
    stock_filter=lambda s: s.startswith('60')
)

# Custom features
def custom_features(stock, price_df, idx, date, ds):
    return {'my_feature': price_df['close'].iloc[idx-1]}

collector = DataCollector(data_source, config,
                          feature_calculator=custom_features)
```

### Evaluation Metrics

```python
from simtrademl import (
    calculate_ic, calculate_rank_ic, calculate_icir,
    calculate_quantile_returns, calculate_direction_accuracy
)

# IC metrics
ic, p_value = calculate_ic(predictions, actuals)
rank_ic, p_value = calculate_rank_ic(predictions, actuals)
icir, ic_std = calculate_icir(predictions, actuals)

# Quantile returns (daily rebalancing)
quantile_returns, long_short = calculate_quantile_returns(
    predictions, actuals, dates=sample_dates
)

# Directional accuracy
accuracy = calculate_direction_accuracy(predictions, actuals)
```

## Testing

```bash
# Run all tests
poetry run pytest

# View coverage
poetry run pytest --cov=simtrademl --cov-report=html
open htmlcov/index.html
```

## Configuration Example

Complete configuration (`config.yml`):

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
  parallel_jobs: -1  # -1 = use all CPUs
```

## Extending Data Sources

```python
from simtrademl.core.data.base import DataSource

class MyDataSource(DataSource):
    def get_stock_list(self) -> List[str]:
        return ['600519.SS', '000858.SZ']

    def get_price_data(self, stock, start_date, end_date, fields):
        # Return DataFrame, index is date
        return pd.DataFrame({
            'open': [...], 'high': [...], 'low': [...],
            'close': [...], 'volume': [...]
        })

    # Implement other required methods...
```

## Dependencies

**Core**: Python 3.9+, numpy, pandas, scikit-learn, **xgboost 0.90** (PTrade compatible version)
**Optional**: simtradelab (data), optuna (hyperparameter optimization), mlflow (experiment tracking)

> ⚠️ **Important**: XGBoost version is locked at 0.90 to ensure PTrade compatibility. Do not upgrade.

## PTrade Integration Guide

### Model Export Formats
PTrade supports multiple model save formats as long as compatible libraries can read them:

```python
import xgboost as xgb

model = xgb.train(params, dtrain, ...)

# Method 1: JSON format (recommended, human-readable)
model.save_model('my_model.json')

# Method 2: XGBoost native format
model.save_model('my_model.model')

# Method 3: Pickle format (universal)
import pickle
with open('my_model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

### Using in SimTradeLab
```python
# Method 1: Load JSON/Model format
import xgboost as xgb
model = xgb.Booster(model_file='my_model.json')

# Method 2: Load Pickle format
import pickle
with open('my_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict
features = [...]
dmatrix = xgb.DMatrix([features])
prediction = model.predict(dmatrix)[0]
```

### Feature Consistency
Ensure the same feature order is used during training and inference:
```python
# Record feature order during training
feature_names = ['ma5', 'ma10', 'rsi14', ...]

# Construct features in same order during inference
features = [ma5, ma10, rsi14, ...]  # Order must be consistent
```

## Development Roadmap

### Current Version (v0.2.0) - MVP ✅
- [x] SimTradeLab data source integration
- [x] XGBoost 0.90 training pipeline
- [x] Quantitative finance evaluation metrics
- [x] Parallel data collection

### Next Phase (v0.3.0) - PTrade Enhancement 🚧
- [ ] **Model Metadata System** (P0) - Feature consistency guarantee
- [ ] **Unified Model Exporter** (P0) - One-click PTrade model package generation
- [ ] **Feature Registry** (P0) - Feature reuse and version management
- [ ] **Fast Training Pipeline** (P1) - Simplify training workflow

See [TODO.md](TODO.md) for details

## License

MIT License

---

**Documentation**: See `examples/mvp_train.py` for complete examples
**Issues**: Submit issues to GitHub
**Test Coverage**: 88% | All 66 tests passed
