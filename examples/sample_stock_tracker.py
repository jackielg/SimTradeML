# -*- coding: utf-8 -*-
"""
样例股票追踪系统 - 追踪样例股在策略中的完整路径

功能：
1. 追踪过滤阶段
2. 追踪观察池阶段
3. 追踪入场队列阶段
4. 追踪买入下单
5. 追踪成交确认
6. 生成追踪报告

版本：v2.2
日期：2026-04-03
修改：移除独立 logger，使用策略的 log 对象（符合 PTrade 规范）
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import pandas as pd

# 移除独立的 logger 创建，使用策略注入的 log 对象
# from simtrademl.core.utils.logger import setup_logger

if TYPE_CHECKING:
    # 仅在类型检查时导入，避免运行时依赖
    from logging import Logger


# ============================================================================
# 配置常量
# ============================================================================

SAMPLE_STOCKS = [
    "002815.SZ",  # 崇达技术
    "300476.SZ",  # 胜宏科技
    "301232.SZ",  # 飞沃科技
    "301377.SZ",  # 鼎泰高科
    "000975.SZ",  # 银泰黄金
    "603226.SH",  # 菲林格尔
]

TRACKING_LOG_DIR = Path("examples/tracking_logs")


# ============================================================================
# 数据类
# ============================================================================


@dataclass
class StockJourney:
    """股票完整路径追踪"""

    stock: str  # 股票代码
    date: pd.Timestamp  # 日期
    passed_filter: bool = False  # 是否通过过滤
    entered_watchlist: bool = False  # 是否进入观察池
    entered_entry_queue: bool = False  # 是否进入入场队列
    order_placed: bool = False  # 是否下单
    order_executed: bool = False  # 是否成交
    model_score: float = 0.0  # 模型评分
    filter_reason: Optional[str] = None  # 过滤原因
    watchlist_rank: Optional[int] = None  # 观察池排名
    entry_queue_rank: Optional[int] = None  # 入场队列排名
    order_price: Optional[float] = None  # 下单价格
    exec_price: Optional[float] = None  # 成交价格
    exec_volume: Optional[int] = None  # 成交数量

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "stock": self.stock,
            "date": str(self.date),
            "passed_filter": self.passed_filter,
            "entered_watchlist": self.entered_watchlist,
            "entered_entry_queue": self.entered_entry_queue,
            "order_placed": self.order_placed,
            "order_executed": self.order_executed,
            "model_score": self.model_score,
            "filter_reason": self.filter_reason,
            "watchlist_rank": self.watchlist_rank,
            "entry_queue_rank": self.entry_queue_rank,
            "order_price": self.order_price,
            "exec_price": self.exec_price,
            "exec_volume": self.exec_volume,
        }

    def get_stage_summary(self) -> str:
        """获取阶段摘要"""
        stages = []
        if self.passed_filter:
            stages.append("✓过滤")
        else:
            stages.append(f"✗过滤 ({self.filter_reason})")

        if self.entered_watchlist:
            stages.append(f"✓观察池 (#{self.watchlist_rank})")
        else:
            stages.append("✗观察池")

        if self.entered_entry_queue:
            stages.append(f"✓入场队列 (#{self.entry_queue_rank})")
        else:
            stages.append("✗入场队列")

        if self.order_placed:
            stages.append(f"✓下单 (@{self.order_price})")
        else:
            stages.append("✗下单")

        if self.order_executed:
            stages.append(f"✓成交 (@{self.exec_price} x{self.exec_volume})")
        else:
            stages.append("✗成交")

        return " -> ".join(stages)


@dataclass
class TrackingReport:
    """追踪报告"""

    start_date: pd.Timestamp  # 开始日期
    end_date: pd.Timestamp  # 结束日期
    total_samples: int  # 样例股总数
    journeys: List[StockJourney]  # 路径列表

    # 各阶段统计
    filter_pass_count: int = 0
    filter_pass_rate: float = 0.0

    watchlist_entry_count: int = 0
    watchlist_entry_rate: float = 0.0
    avg_watchlist_rank: float = 0.0

    entry_queue_count: int = 0
    entry_queue_rate: float = 0.0
    avg_entry_queue_rank: float = 0.0

    order_count: int = 0
    order_rate: float = 0.0

    execution_count: int = 0
    execution_rate: float = 0.0

    # 模型评分统计
    avg_model_score: float = 0.0
    avg_model_score_passed_filter: float = 0.0

    # 问题诊断
    common_filter_reasons: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "start_date": str(self.start_date),
            "end_date": str(self.end_date),
            "total_samples": self.total_samples,
            "filter_pass_count": self.filter_pass_count,
            "filter_pass_rate": self.filter_pass_rate,
            "watchlist_entry_count": self.watchlist_entry_count,
            "watchlist_entry_rate": self.watchlist_entry_rate,
            "avg_watchlist_rank": self.avg_watchlist_rank,
            "entry_queue_count": self.entry_queue_count,
            "entry_queue_rate": self.entry_queue_rate,
            "avg_entry_queue_rank": self.avg_entry_queue_rank,
            "order_count": self.order_count,
            "order_rate": self.order_rate,
            "execution_count": self.execution_count,
            "execution_rate": self.execution_rate,
            "avg_model_score": self.avg_model_score,
            "avg_model_score_passed_filter": self.avg_model_score_passed_filter,
            "common_filter_reasons": self.common_filter_reasons,
            "journeys": [j.to_dict() for j in self.journeys],
        }


# ============================================================================
# 样例股票追踪器
# ============================================================================


class SampleStockTracker:
    """样例股票追踪器"""

    def __init__(
        self,
        sample_stocks: Optional[List[str]] = None,
        log=None,  # 使用策略的 log 对象（PTrade 内置）
        log_dir: Path = TRACKING_LOG_DIR,
    ):
        """
        初始化追踪器

        Args:
            sample_stocks: 样例股票池，默认使用全局配置
            log: 日志对象（策略注入，符合 PTrade 规范）
            log_dir: 日志目录
        """
        self.sample_stocks = sample_stocks or SAMPLE_STOCKS
        self.log = log  # 使用策略的 log 对象
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.journeys: Dict[str, StockJourney] = {}
        self.daily_logs: List[Dict[str, Any]] = {}
        self.daily_logs: List[Dict[str, Any]] = []

    def _get_journey_key(self, stock: str, date: pd.Timestamp) -> str:
        """获取路径唯一键"""
        return f"{stock}_{str(date.date())}"

    def track_filter_stage(
        self,
        stock: str,
        date: pd.Timestamp,
        passed: bool,
        reason: Optional[str] = None,
        model_score: float = 0.0,
    ):
        """
        追踪过滤阶段

        Args:
            stock: 股票代码
            date: 日期
            passed: 是否通过过滤
            reason: 过滤原因（未通过时）
            model_score: 模型评分
        """
        if stock not in self.sample_stocks:
            return

        key = self._get_journey_key(stock, date)

        if key not in self.journeys:
            self.journeys[key] = StockJourney(stock=stock, date=date)

        journey = self.journeys[key]
        journey.passed_filter = passed
        journey.filter_reason = reason
        journey.model_score = model_score

        # 样例追踪日志格式（使用策略的 log 对象）
        if passed:
            self.log.info("[样例追踪·过滤] %s - 通过 (评分：%.4f)", stock, model_score)
        else:
            self.log.info(
                "[样例追踪·过滤] %s - 排除 (原因：%s)", stock, reason or "未知"
            )

    def track_filter(
        self,
        stock: str,
        passed: bool,
        reason: Optional[str] = None,
    ):
        """
        简化版过滤追踪方法（兼容旧接口）

        Args:
            stock: 股票代码
            passed: 是否通过过滤
            reason: 过滤原因
        """
        # 使用当前日期（策略调用时会传入 context.current_dt）
        self.track_filter_stage(
            stock=stock,
            date=pd.Timestamp.now(),
            passed=passed,
            reason=reason,
            model_score=0.0,
        )

    def track_watchlist_stage(
        self,
        stock: str,
        date: pd.Timestamp,
        entered: bool,
        rank: Optional[int] = None,
        model_score: Optional[float] = None,
    ):
        """
        追踪观察池阶段

        Args:
            stock: 股票代码
            date: 日期
            entered: 是否进入观察池
            rank: 观察池排名
            model_score: 模型评分
        """
        if stock not in self.sample_stocks:
            return

        key = self._get_journey_key(stock, date)

        if key not in self.journeys:
            self.journeys[key] = StockJourney(stock=stock, date=date)

        journey = self.journeys[key]
        journey.entered_watchlist = entered
        journey.watchlist_rank = rank

        if model_score is not None:
            journey.model_score = model_score

        self.log.debug(
            "观察池阶段：%s %s - %s (排名：%s)",
            stock,
            date.date(),
            "进入" if entered else "未进入",
            rank,
        )

    def track_entry_queue_stage(
        self,
        stock: str,
        date: pd.Timestamp,
        entered: bool,
        rank: Optional[int] = None,
    ):
        """
        追踪入场队列阶段

        Args:
            stock: 股票代码
            date: 日期
            entered: 是否进入入场队列
            rank: 入场队列排名
        """
        if stock not in self.sample_stocks:
            return

        key = self._get_journey_key(stock, date)

        if key not in self.journeys:
            self.journeys[key] = StockJourney(stock=stock, date=date)

        journey = self.journeys[key]
        journey.entered_entry_queue = entered
        journey.entry_queue_rank = rank

        self.log.debug(
            "入场队列阶段：%s %s - %s (排名：%s)",
            stock,
            date.date(),
            "进入" if entered else "未进入",
            rank,
        )

    def track_order_stage(
        self,
        stock: str,
        date: pd.Timestamp,
        placed: bool,
        order_price: Optional[float] = None,
    ):
        """
        追踪买入下单

        Args:
            stock: 股票代码
            date: 日期
            placed: 是否下单
            order_price: 下单价格
        """
        if stock not in self.sample_stocks:
            return

        key = self._get_journey_key(stock, date)

        if key not in self.journeys:
            self.journeys[key] = StockJourney(stock=stock, date=date)

        journey = self.journeys[key]
        journey.order_placed = placed
        journey.order_price = order_price

        self.log.debug(
            "买入下单阶段：%s %s - %s (价格：%s)",
            stock,
            date.date(),
            "下单" if placed else "未下单",
            order_price,
        )

    def track_execution_stage(
        self,
        stock: str,
        date: pd.Timestamp,
        executed: bool,
        exec_price: Optional[float] = None,
        exec_volume: Optional[int] = None,
    ):
        """
        追踪成交确认

        Args:
            stock: 股票代码
            date: 日期
            executed: 是否成交
            exec_price: 成交价格
            exec_volume: 成交数量
        """
        if stock not in self.sample_stocks:
            return

        key = self._get_journey_key(stock, date)

        if key not in self.journeys:
            self.journeys[key] = StockJourney(stock=stock, date=date)

        journey = self.journeys[key]
        journey.order_executed = executed
        journey.exec_price = exec_price
        journey.exec_volume = exec_volume

        self.log.info(
            "成交确认阶段：%s %s - %s (价格：%s, 数量：%s)",
            stock,
            date.date(),
            "成交" if executed else "未成交",
            exec_price,
            exec_volume,
        )

    def generate_report(
        self,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> TrackingReport:
        """
        生成追踪报告

        Args:
            start_date: 开始日期，默认使用所有数据
            end_date: 结束日期，默认使用所有数据

        Returns:
            TrackingReport 对象
        """
        self.log.info("开始生成追踪报告")

        journeys = list(self.journeys.values())

        if start_date is not None:
            journeys = [j for j in journeys if j.date >= start_date]
        if end_date is not None:
            journeys = [j for j in journeys if j.date <= end_date]

        if not journeys:
            return TrackingReport(
                start_date=start_date or pd.Timestamp.now(),
                end_date=end_date or pd.Timestamp.now(),
                total_samples=0,
                journeys=[],
            )

        total = len(journeys)
        filter_pass = sum(1 for j in journeys if j.passed_filter)
        watchlist_entry = sum(1 for j in journeys if j.entered_watchlist)
        entry_queue = sum(1 for j in journeys if j.entered_entry_queue)
        order = sum(1 for j in journeys if j.order_placed)
        execution = sum(1 for j in journeys if j.order_executed)

        avg_model_score = (
            sum(j.model_score for j in journeys) / total if total > 0 else 0.0
        )
        avg_model_score_passed = (
            sum(j.model_score for j in journeys if j.passed_filter) / filter_pass
            if filter_pass > 0
            else 0.0
        )

        watchlist_ranks = [
            j.watchlist_rank for j in journeys if j.watchlist_rank is not None
        ]
        avg_watchlist_rank = (
            sum(watchlist_ranks) / len(watchlist_ranks) if watchlist_ranks else 0.0
        )

        entry_queue_ranks = [
            j.entry_queue_rank for j in journeys if j.entry_queue_rank is not None
        ]
        avg_entry_queue_rank = (
            sum(entry_queue_ranks) / len(entry_queue_ranks)
            if entry_queue_ranks
            else 0.0
        )

        filter_reasons = {}
        for j in journeys:
            if j.filter_reason:
                filter_reasons[j.filter_reason] = (
                    filter_reasons.get(j.filter_reason, 0) + 1
                )

        report = TrackingReport(
            start_date=min(j.date for j in journeys),
            end_date=max(j.date for j in journeys),
            total_samples=total,
            journeys=journeys,
            filter_pass_count=filter_pass,
            filter_pass_rate=filter_pass / total if total > 0 else 0.0,
            watchlist_entry_count=watchlist_entry,
            watchlist_entry_rate=(
                watchlist_entry / filter_pass if filter_pass > 0 else 0.0
            ),
            avg_watchlist_rank=avg_watchlist_rank,
            entry_queue_count=entry_queue,
            entry_queue_rate=entry_queue / filter_pass if filter_pass > 0 else 0.0,
            avg_entry_queue_rank=avg_entry_queue_rank,
            order_count=order,
            order_rate=order / total if total > 0 else 0.0,
            execution_count=execution,
            execution_rate=execution / order if order > 0 else 0.0,
            avg_model_score=avg_model_score,
            avg_model_score_passed_filter=avg_model_score_passed,
            common_filter_reasons=filter_reasons,
        )

        self.log.info("追踪报告生成完成")
        self.log.info("样例股总数：%s", total)
        self.log.info("过滤通过率：%.2f%%", report.filter_pass_rate * 100)
        self.log.info("观察池进入率：%.2f%%", report.watchlist_entry_rate * 100)
        self.log.info("入场队列进入率：%.2f%%", report.entry_queue_rate * 100)
        self.log.info("下单率：%.2f%%", report.order_rate * 100)
        self.log.info("成交率：%.2f%%", report.execution_rate * 100)

        return report

    def save_report(
        self,
        report: TrackingReport,
        filename: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ) -> str:
        """
        保存追踪报告

        Args:
            report: 追踪报告对象
            filename: 文件名，默认自动生成
            output_dir: 输出目录，默认使用 log_dir

        Returns:
            文件路径
        """
        if output_dir is None:
            output_dir = self.log_dir
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sample_stock_report_{timestamp}.json"

        filepath = output_dir / filename

        with filepath.open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

        self.log.info("追踪报告已保存：%s", filepath)

        return str(filepath)

    def print_summary(self, report: TrackingReport):
        """
        打印摘要信息

        Args:
            report: 追踪报告对象
        """
        print("\n" + "=" * 60)
        print("样例股票追踪报告摘要")
        print("=" * 60)
        print(f"时间范围：{report.start_date.date()} - {report.end_date.date()}")
        print(f"样例股总数：{report.total_samples}")
        print()
        print("各阶段通过率:")
        print(
            f"  - 过滤阶段：{report.filter_pass_count}/{report.total_samples} ({report.filter_pass_rate*100:.2f}%)"
        )
        print(
            f"  - 观察池阶段：{report.watchlist_entry_count}/{report.filter_pass_count} ({report.watchlist_entry_rate*100:.2f}%)"
        )
        print(
            f"  - 入场队列阶段：{report.entry_queue_count}/{report.filter_pass_count} ({report.entry_queue_rate*100:.2f}%)"
        )
        print(
            f"  - 买入下单：{report.order_count}/{report.total_samples} ({report.order_rate*100:.2f}%)"
        )
        print(
            f"  - 成交确认：{report.execution_count}/{report.order_count} ({report.execution_rate*100:.2f}%)"
        )
        print()
        print("模型评分:")
        print(f"  - 平均评分：{report.avg_model_score:.4f}")
        print(f"  - 通过过滤的平均评分：{report.avg_model_score_passed_filter:.4f}")
        print()

        if report.common_filter_reasons:
            print("常见过滤原因:")
            for reason, count in sorted(
                report.common_filter_reasons.items(), key=lambda x: x[1], reverse=True
            )[:5]:
                print(f"  - {reason}: {count}次")
        print()

        print("路径示例:")
        for journey in report.journeys[:5]:
            print(
                f"  {journey.stock} ({journey.date.date()}): {journey.get_stage_summary()}"
            )
        print("=" * 60)


# ============================================================================
# 策略集成示例
# ============================================================================


def create_tracker(log=None) -> SampleStockTracker:
    """创建追踪器实例（策略集成用）

    Args:
        log: 策略的 log 对象（PTrade 内置）
    """
    tracker = SampleStockTracker(log=log)
    if log:
        log.info("样例股票追踪器已初始化")
    return tracker


# ============================================================================
# 命令行接口
# ============================================================================


def main():
    """测试追踪器功能"""
    # 测试环境：创建一个简单的 logger
    import logging

    test_log = logging.getLogger("test")
    test_log.setLevel(logging.INFO)
    if not test_log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        test_log.addHandler(handler)

    test_log.info("样例股票追踪器测试")

    tracker = SampleStockTracker(log=test_log)

    test_date = pd.Timestamp("2026-04-01")

    for stock in SAMPLE_STOCKS:
        tracker.track_filter_stage(
            stock=stock,
            date=test_date,
            passed=True,
            model_score=0.75,
        )

        tracker.track_watchlist_stage(
            stock=stock,
            date=test_date,
            entered=True,
            rank=5,
        )

        tracker.track_entry_queue_stage(
            stock=stock,
            date=test_date,
            entered=True,
            rank=3,
        )

        tracker.track_order_stage(
            stock=stock,
            date=test_date,
            placed=True,
            order_price=10.5,
        )

        tracker.track_execution_stage(
            stock=stock,
            date=test_date,
            executed=True,
            exec_price=10.52,
            exec_volume=100,
        )

    report = tracker.generate_report()
    tracker.print_summary(report)
    tracker.save_report(report)

    test_log.info("测试完成")


if __name__ == "__main__":
    main()
