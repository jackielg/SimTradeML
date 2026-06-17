# AGENTS.md — SimTradeML

## 定位
ML 框架 v0.2.0，训练 XGBoost 预测模型，打包为 .ptp 文件部署到 SimTradeLab 和 PTrade 实盘。
从 SimTradeLab 读数据 → 训练模型 → 产出 .ptp 部署包。

## 关键命令
- 训练 MVP: `poetry run python examples/mvp_train.py`
- 训练 trends_up 模型: `poetry run python examples/trends_up_model.py`
- 验证 PTrade 部署: `poetry run python examples/validate_ptrade_deploy.py`
- 测试全部: `poetry run pytest`
- 按 marker 测试: `poetry run pytest -m unit`

## 核心架构
- `src/simtrademl/core/data/` — 数据源
  - `data_source.py` (DataSource ABC)
  - `data_sources/simtradelab_source.py` — 对接 SimTradeLab 数据
- `src/simtrademl/core/models/` — 模型打包
  - `package.py` (PTradeModelPackage) — .ptp = XGBoost + scaler + metadata
  - `predict()` / `save()` / `load()` / `summary()` — 部署包核心 API
- `src/simtrademl/core/utils/metrics.py` — 评估指标
  - IC, Rank IC, ICIR, quantile returns, direction accuracy
- `src/simtrademl/features/` — 特征工程
  - `registry.py` (FeatureRegistry) — 装饰器注册模式
  - `technical.py` — 32 个技术指标注册
- `examples/` — 训练/部署样例
  - `mvp_train.py` — 最小可运行训练
  - `trends_up_model.py` / `trends_up_model_v2.py` — trends_up 策略专用模型
  - `validate_ptrade_deploy.py` — PTrade 部署验证
  - `model_evolution_system.py` — 模型演进系统
  - `sample_stock_tracker.py` / `test_sample_stocks.py` — 样本股跟踪

## 规则

### 模型版本兼容
- XGBoost 版本：本框架 >=1.7,<3.0 / PTrade 生产 0.90 / macOS ARM64 3.2.0（API 兼容）
- .ptp 文件格式包含模型 + scaler + 元数据，SimTradeLab 和 PTrade 通用
- 加载时检查 XGBoost 版本兼容性，必要时回退

### 特征注册
- FeatureRegistry 使用装饰器模式：`@registry.register("name")`
- 新增指标需在 `features/technical.py` 注册
- 注册名与 SimTradeLab 引用名保持一致（跨项目契约）

### 跨项目契约
- `SimTradeLabDataSource` 是 SimTradeLab 数据消费的官方入口
- .ptp 部署包格式必须与 SimTradeLab `ptrade/model_loader.py` 兼容
- 训练样本股由 SimTradeLab 策略目录提供（如 `SimTradeLab/strategies/<name>/`）

### 评估标准
- 模型上线前必须验证 IC、Rank IC、ICIR、quantile returns
- direction accuracy 决定策略可不可用
- 模型退化（< 阈值）时通过 `model_evolution_system.py` 自动重训

## Out of Scope
- 🚫 永远不要推送到 upstream (kay-ou) 仓库
- 不自动推送到远程仓库（需用户确认后才能 push origin）


<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **SimTradeML** (1697 symbols, 2611 relationships, 35 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/SimTradeML/context` | Codebase overview, check index freshness |
| `gitnexus://repo/SimTradeML/clusters` | All functional areas |
| `gitnexus://repo/SimTradeML/processes` | All execution flows |
| `gitnexus://repo/SimTradeML/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
