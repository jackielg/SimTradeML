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

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **SimTradeML** (1668 symbols, 2235 relationships, 9 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

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
