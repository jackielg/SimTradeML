# trends_up 模型自进化系统设计文档 (optimize_21)

## 1. 设计背景

### 1.1 核心需求
用户需要模型具备**自进化能力**，通过以下机制持续提升模型质量：
1. **5 只样例牛股双重角色**：
   - 模型学习的核心对象
   - 模型能力的检测器（通过日志观察策略是否发现和买入）
   - 禁止硬编码内定买入

2. **标签定义优化**：当前标签定义过于简单（仅 0/1 分类），需要多维度评分

3. **周一自进化机制**：收集上周实战数据，分析失败案例，训练新模型并部署

### 1.2 样例牛股配置
```python
SAMPLE_STOCKS = [
    "002815.SZ",  # 源达技术
    "300476.SZ",  # 胜宏科技
    "301232.SZ",  # 飞沃科技
    "000975.SZ",  # 银泰黄金
    "603226.SH",  # 菲格尔
]
```

---

## 2. 文件结构定义

```
SimTradeML/
├── examples/
│   ├── trends_up_model_v2.py           # 新版模型训练脚本
│   ├── model_evolution_system.py       # 自进化系统
│   ├── sample_stock_tracker.py         # 样例追踪系统
│   └── ...
├── docs/
│   └── model_evolution_design.md       # 本设计文档
└── src/simtrademl/
    └── ...
```

---

## 3. 类/函数清单

### 3.1 新版模型训练脚本 (`trends_up_model_v2.py`)

#### 类
- `MultiDimensionalLabelCalculator` - 多维度标签计算器
  - `calculate_breakout_strength()` - 计算突破强度
  - `calculate_return_strength()` - 计算收益强度
  - `calculate_stability()` - 计算稳定性
  - `calculate_persistence()` - 计算持续性
  - `calculate_composite_score()` - 计算综合评分

#### 函数
- `calculate_trends_up_features()` - 计算特征（保持与 v1 兼容）
- `is_candidate_window()` - 候选窗口过滤
- `collect_samples()` - 样本采集（支持样例股加权）
- `train_model()` - 模型训练
- `export_model_artifacts()` - 导出模型产物

### 3.2 自进化系统 (`model_evolution_system.py`)

#### 类
- `ModelEvolutionSystem` - 模型自进化系统
  - `weekly_evolution()` - 每周一执行完整进化流程
  - `analyze_failures()` - 分析失败案例
  - `collect_sample_stocks()` - 采集样例牛股样本
  - `collect_failure_cases()` - 采集失败案例
  - `build_layered_dataset()` - 构建分层训练数据集
  - `verify_improvement()` - 验证模型改进
  - `deploy_new_model()` - 部署新模型
  - `rollback_to_previous_model()` - 回退到旧模型

#### 数据类
- `FailureCase` - 失败案例
  - `stock: str` - 股票代码
  - `date: pd.Timestamp` - 日期
  - `prediction: float` - 预测值
  - `actual_label: int` - 实际标签
  - `failure_type: str` - 失败类型（假阳性/假阴性）
  - `features: Dict[str, float]` - 特征

- `LayeredDataset` - 分层数据集
  - `base_layer: pd.DataFrame` - 基础层（权重 1.0）
  - `core_layer: pd.DataFrame` - 核心层（权重 3.0）
  - `reinforcement_layer: pd.DataFrame` - 强化层（权重 5.0）
  - `fresh_layer: pd.DataFrame` - 新鲜层（权重 2.0）

### 3.3 样例追踪系统 (`sample_stock_tracker.py`)

#### 类
- `SampleStockTracker` - 样例股票追踪器
  - `track_filter_stage()` - 追踪过滤阶段
  - `track_watchlist_stage()` - 追踪观察池阶段
  - `track_entry_queue_stage()` - 追踪入场队列阶段
  - `track_order_stage()` - 追踪买入下单
  - `track_execution_stage()` - 追踪成交确认
  - `generate_report()` - 生成追踪报告

#### 数据类
- `StockJourney` - 股票完整路径
  - `stock: str` - 股票代码
  - `date: pd.Timestamp` - 日期
  - `passed_filter: bool` - 是否通过过滤
  - `entered_watchlist: bool` - 是否进入观察池
  - `entered_entry_queue: bool` - 是否进入入场队列
  - `order_placed: bool` - 是否下单
  - `order_executed: bool` - 是否成交
  - `model_score: float` - 模型评分

---

## 4. 特征清单

### 4.1 技术特征（保持与 v1 兼容）
共 24 个基础特征：
- `atr_ratio_14` - ATR 占收盘价比例
- `breakout_distance_20d` - 距 20 日高点距离
- `breakout_distance_60d` - 距 60 日高点距离
- `close_to_ma20` - 收盘价相对 MA20 位置
- `close_to_ma60` - 收盘价相对 MA60 位置
- `ma20_slope_5d` - MA20 斜率
- `ma20_to_ma60` - MA20 相对 MA60 位置
- `ma5_to_ma20` - MA5 相对 MA20 位置
- `ma60_slope_10d` - MA60 斜率
- `ma60_to_ma120` - MA60 相对 MA120 位置
- `price_position_20d` - 价格在 20 日区间位置
- `price_position_60d` - 价格在 60 日区间位置
- `relative_strength_20d` - 相对强度 20 日
- `relative_strength_60d` - 相对强度 60 日
- `return_5d` - 5 日收益
- `return_10d` - 10 日收益
- `return_20d` - 20 日收益
- `return_60d` - 60 日收益
- `rsi14` - RSI 指标
- `volatility_20d` - 20 日波动率
- `volume_ratio_20d` - 成交量比率
- `volume_ratio_5d` - 成交量比率
- `volume_trend_5_20` - 成交量趋势
- `volume_trend_20_60` - 成交量趋势

### 4.2 标签特征（新增多维度评分）
- `breakout_strength` - 突破强度（0-1 分）
- `return_strength` - 收益强度（0-1 分）
- `stability` - 稳定性（0-1 分）
- `persistence` - 持续性（0-1 分）
- `composite_score` - 综合评分（加权平均）
- `bull_stock_label` - 牛股标签（综合评分≥0.6）

---

## 5. 标签定义

### 5.1 多维度评分系统

#### 突破强度（0-1 分）
```python
def calculate_breakout_strength(
    future_max_return: float,
    breakout_success: bool,
    past_high_20: float,
    current_close: float
) -> float:
    """
    计算突破强度评分
    
    评分标准：
    - 突破成功且涨幅>15%: 1.0 分
    - 突破成功且涨幅>10%: 0.8 分
    - 突破成功且涨幅>5%: 0.6 分
    - 突破成功但涨幅<5%: 0.4 分
    - 未突破但涨幅>3%: 0.2 分
    - 其他：0.0 分
    """
```

#### 收益强度（0-1 分）
```python
def calculate_return_strength(
    future_return: float,
    benchmark_return: float,
    excess_return: float
) -> float:
    """
    计算收益强度评分
    
    评分标准：
    - 未来收益>15% 且超额>5%: 1.0 分
    - 未来收益>10% 且超额>3%: 0.8 分
    - 未来收益>5% 且超额>2%: 0.6 分
    - 未来收益>3% 且超额>1%: 0.4 分
    - 未来收益>0%: 0.2 分
    - 其他：0.0 分
    """
```

#### 稳定性（0-1 分）
```python
def calculate_stability(
    future_min_return: float,
    volatility: float,
    max_drawdown: float
) -> float:
    """
    计算稳定性评分
    
    评分标准：
    - 最大回撤>-3% 且波动率低：1.0 分
    - 最大回撤>-5% 且波动率中：0.8 分
    - 最大回撤>-8% 且波动率可控：0.6 分
    - 最大回撤>-10%: 0.4 分
    - 最大回撤>-15%: 0.2 分
    - 其他：0.0 分
    """
```

#### 持续性（0-1 分）
```python
def calculate_persistence(
    future_window: pd.Series,
    positive_days_ratio: float,
    trend_consistency: float
) -> float:
    """
    计算持续性评分
    
    评分标准：
    - 上涨天数占比>80% 且趋势一致性强：1.0 分
    - 上涨天数占比>70% 且趋势一致性中：0.8 分
    - 上涨天数占比>60%: 0.6 分
    - 上涨天数占比>50%: 0.4 分
    - 其他：0.0 分
    """
```

#### 综合评分（加权平均）
```python
def calculate_composite_score(
    breakout_strength: float,
    return_strength: float,
    stability: float,
    persistence: float,
    weights: Dict[str, float] = None
) -> float:
    """
    计算综合评分
    
    默认权重：
    - 突破强度：0.30
    - 收益强度：0.30
    - 稳定性：0.20
    - 持续性：0.20
    
    牛股标签：综合评分 >= 0.6
    """
```

---

## 6. 分层训练数据集

### 6.1 数据分层设计

| 层级 | 数据来源 | 权重 | 说明 |
|-----|---------|------|------|
| **基础层** | 历史全市场数据 | 1.0 | 提供广泛的市场模式 |
| **核心层** | 5 只样例牛股完整历史 | 3.0 | 重点学习样例股特征 |
| **强化层** | 失败案例 | 5.0 | 重点修正模型错误 |
| **新鲜层** | 上周实战数据 | 2.0 | 保持模型时效性 |

### 6.2 加权采样策略
```python
def build_layered_dataset(
    base_df: pd.DataFrame,
    core_df: pd.DataFrame,
    reinforcement_df: pd.DataFrame,
    fresh_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    构建分层加权数据集
    
    通过重复采样实现权重：
    - 基础层：采样 1 次
    - 核心层：采样 3 次
    - 强化层：采样 5 次
    - 新鲜层：采样 2 次
    """
```

---

## 7. 自进化流程

### 7.1 周一完整重训流程

```python
def weekly_evolution(self, previous_week_data: pd.DataFrame):
    """
    每周一执行完整进化流程
    
    步骤：
    1. 收集上周实战数据
    2. 分析失败案例（假阳性、假阴性）
    3. 构建分层训练数据集
    4. 训练新模型
    5. 验证改进（对比上周模型）
    6. 部署或回退
    """
```

### 7.2 失败案例分析
```python
def analyze_failures(self, predictions: pd.DataFrame, actuals: pd.DataFrame):
    """
    分析失败案例
    
    失败类型：
    - 假阳性（False Positive）：模型预测为牛股但实际不是
    - 假阴性（False Negative）：模型未识别出牛股
    
    输出：
    - 失败案例列表
    - 失败模式分析
    - 特征重要性变化
    """
```

### 7.3 模型验证
```python
def verify_improvement(
    old_model: xgb.Booster,
    new_model: xgb.Booster,
    test_dataset: pd.DataFrame
) -> Dict[str, float]:
    """
    验证新模型是否改进
    
    对比指标：
    - AUC
    - Average Precision
    - Top10 命中率
    - Top10 平均收益
    - 样例股识别率
    """
```

### 7.4 回退机制
```python
def rollback_to_previous_model(self, reason: str):
    """
    回退到旧模型
    
    回退条件：
    - 新模型 AUC 下降 > 5%
    - Top10 命中率下降 > 10%
    - 样例股识别率下降
    - 训练失败或验证失败
    
    回退策略：
    1. 先回退到上周模型
    2. 再失败则退回原规则链
    """
```

---

## 8. 样例追踪系统

### 8.1 追踪阶段

```python
class StockJourney:
    """股票完整路径追踪"""
    
    def __init__(self, stock: str, date: pd.Timestamp):
        self.stock = stock
        self.date = date
        self.passed_filter = False           # 过滤阶段
        self.entered_watchlist = False       # 观察池阶段
        self.entered_entry_queue = False     # 入场队列阶段
        self.order_placed = False            # 买入下单
        self.order_executed = False          # 成交确认
        self.model_score = 0.0               # 模型评分
```

### 8.2 追踪日志
```python
def track_filter_stage(self, stock: str, date: pd.Timestamp, passed: bool):
    """追踪过滤阶段"""
    
def track_watchlist_stage(self, stock: str, date: pd.Timestamp, entered: bool, score: float):
    """追踪观察池阶段"""
    
def track_entry_queue_stage(self, stock: str, date: pd.Timestamp, entered: bool, rank: int):
    """追踪入场队列阶段"""
    
def track_order_stage(self, stock: str, date: pd.Timestamp, placed: bool):
    """追踪买入下单"""
    
def track_execution_stage(self, stock: str, date: pd.Timestamp, executed: bool):
    """追踪成交确认"""
```

### 8.3 追踪报告
```python
def generate_report(self) -> Dict[str, Any]:
    """
    生成追踪报告
    
    包含：
    - 样例股总数
    - 各阶段通过率
    - 平均模型评分
    - 买入成功率
    - 问题诊断建议
    """
```

---

## 9. 集成边界

### 9.1 与策略主体的边界
- **模型负责**：
  - 训练和导出模型产物
  - 提供多维度标签评分
  - 提供阈值建议
  - 提供回退方案

- **策略负责**：
  - 加载模型并推理
  - 集成到观察池排序
  - 集成到买前过滤
  - 实现回退逻辑

### 9.2 接入点
```python
# 策略侧接入示例
def before_trading_start(context):
    """盘前加载模型"""
    # 1. 加载最新模型
    # 2. 验证模型文件完整性
    # 3. 失败则回退到上周模型
    # 4. 再失败则退回原规则链
    
def on_bar(context, data):
    """盘中推理"""
    # 1. 构造特征（按 metadata 顺序）
    # 2. 模型推理
    # 3. 应用阈值
    # 4. 排序增强/买前过滤
```

---

## 10. 验收检查点清单

### 10.1 代码验收
- [ ] 5 只样例牛股全部纳入训练
- [ ] 标签定义改为多维度评分
- [ ] 自进化系统架构清晰
- [ ] 分层训练数据集设计合理
- [ ] 样例追踪系统可完整追踪

### 10.2 功能验收
- [ ] 周一重训流程可执行
- [ ] 失败案例分析可输出
- [ ] 模型验证对比可量化
- [ ] 回退机制可触发
- [ ] 样例股追踪可日志

### 10.3 集成验收
- [ ] 模型产物可加载
- [ ] 特征顺序与 metadata 一致
- [ ] 阈值建议可配置
- [ ] 回退方案可实现

---

## 11. 版本命名规则

### 11.1 模型版本
- `v2.0` - 多维度标签版本
- `v2.1` - 自进化系统版本
- `v2.1.WXX` - 第 WXX 周版本（如 `v2.1.W42`）

### 11.2 周内复用规则
- **周一**：完整重训（训练截止上一个完整交易日）
- **周二到周五**：复用周一模型
- **失败回退**：
  - 先回退到上周模型（`v2.1.W(XX-1)`）
  - 再失败退回原规则链

---

## 12. 下一步工作

1. 实现 `trends_up_model_v2.py` - 新版模型训练脚本
2. 实现 `model_evolution_system.py` - 自进化系统
3. 实现 `sample_stock_tracker.py` - 样例追踪系统
4. 运行训练验证 5 只样例股
5. 输出模型训练报告
6. 输出集成交底文档
