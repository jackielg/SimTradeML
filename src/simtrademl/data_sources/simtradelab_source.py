# -*- coding: utf-8 -*-
"""
SimTradeLab 数据源实现
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from simtrademl.core.data.base import DataSource

logger = logging.getLogger("simtrademl")


class SimTradeLabDataSource(DataSource):
    """基于当前 SimTradeLab 本地数据接口的数据源。"""

    _FIELD_MAP = {
        "pe_ratio": "pe_ttm",
        "pb_ratio": "pb",
        "ps_ratio": "ps_ttm",
        "pcf_ratio": "pcf",
        "circulating_value": "float_value",
    }

    def __init__(self, data_path: Optional[str] = None):
        """初始化数据源。"""
        self.data_path = self._resolve_data_path(data_path)
        self.api = self._build_api(self.data_path)
        self._stock_cache: Optional[List[str]] = None

    @staticmethod
    def _resolve_data_path(data_path: Optional[str]) -> Optional[str]:
        """解析可用的数据目录。"""
        if data_path:
            return str(Path(data_path).expanduser().resolve())

        env_path = os.environ.get("SIMTRADELAB_DATA_PATH")
        if env_path:
            return str(Path(env_path).expanduser().resolve())

        project_root = Path(__file__).resolve().parents[3]
        candidates = [
            project_root / "data",
            project_root.parent / "SimTradeLab" / "data",
            project_root.parent / "SimTradeLab" / "data" / "cn",
        ]

        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())

        return None

    @staticmethod
    def _build_api(data_path: Optional[str]):
        """构建兼容当前 SimTradeLab 的本地 API。"""
        try:
            from simtradelab.ptrade import PtradeAPI, create_research_context
            from simtradelab.ptrade.data_context import DataContext
            from simtradelab.service.data_server import DataServer
        except ImportError as exc:
            raise ImportError(
                "无法导入 SimTradeLab 当前版本接口，请确认本地 SimTradeLab 已加入 Python 路径。"
            ) from exc

        try:
            data_server = DataServer(
                required_data={"price", "valuation", "fundamentals", "exrights"},
                frequency="1d",
                data_path=data_path,
                market="CN",
            )
            context = create_research_context()
            if getattr(data_server, "trade_days", None) is not None and len(data_server.trade_days) > 0:
                context.current_dt = data_server.trade_days[-1]

            data_context = DataContext(
                stock_data_dict=data_server.stock_data_dict,
                valuation_dict=data_server.valuation_dict,
                fundamentals_dict=data_server.fundamentals_dict,
                exrights_dict=data_server.exrights_dict,
                benchmark_data=data_server.benchmark_data,
                stock_metadata=data_server.stock_metadata,
                index_constituents=data_server.index_constituents,
                stock_status_history=data_server.stock_status_history,
                adj_pre_cache=data_server.adj_pre_cache,
                adj_post_cache=data_server.adj_post_cache,
                dividend_cache=data_server.dividend_cache,
                trade_days=data_server.trade_days,
                stock_data_dict_1m=data_server.stock_data_dict_1m,
            )
            return PtradeAPI(data_context=data_context, context=context, log=logger)
        except Exception as exc:
            raise RuntimeError(
                f"初始化 SimTradeLab 数据接口失败，data_path={data_path!r}: {exc}"
            ) from exc

    def get_stock_list(self) -> List[str]:
        """获取 A 股股票列表。"""
        if self._stock_cache is None:
            try:
                self._stock_cache = self.api.get_Ashares() or []
            except Exception as exc:
                logger.exception("获取股票列表失败")
                raise RuntimeError("获取股票列表失败") from exc
        return self._stock_cache

    def get_trading_dates(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[pd.Timestamp]:
        """获取交易日历。"""
        try:
            trading_dates = self.api.get_trade_days(start_date=start_date, end_date=end_date)
            return list(pd.to_datetime(trading_dates))
        except Exception as exc:
            logger.exception("获取交易日历失败")
            raise RuntimeError("获取交易日历失败") from exc

    def get_price_data(
        self,
        stock: str,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """获取日线行情数据。"""
        default_fields = fields or ["open", "high", "low", "close", "volume"]
        start_str = start_date.strftime("%Y-%m-%d") if start_date is not None else None
        end_str = end_date.strftime("%Y-%m-%d") if end_date is not None else None

        try:
            df = self.api.get_price(
                stock,
                start_date=start_str,
                end_date=end_str,
                frequency="1d",
                fields=default_fields,
                fq="pre",
            )
        except Exception as exc:
            logger.debug("获取行情失败: %s %s", stock, exc)
            return pd.DataFrame(columns=default_fields)

        if df is None or df.empty:
            return pd.DataFrame(columns=default_fields)
        return df

    def get_fundamentals(
        self,
        stock: str,
        date: pd.Timestamp,
        fields: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """获取估值类基本面数据。"""
        requested_fields = fields or [
            "pe_ttm",
            "pb",
            "ps_ttm",
            "pcf",
            "total_value",
            "float_value",
        ]
        mapped_fields = [self._FIELD_MAP.get(field, field) for field in requested_fields]

        try:
            result = self.api.get_fundamentals(
                stock,
                "valuation",
                mapped_fields,
                date.strftime("%Y-%m-%d"),
            )
        except Exception as exc:
            logger.debug("获取基本面失败: %s %s %s", stock, date.strftime("%Y-%m-%d"), exc)
            return None

        if result is None or result.empty:
            return None
        return result.iloc[0].to_dict()

    def get_market_data(
        self,
        benchmark: str = "000300.SS",
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """获取基准行情数据。"""
        return self.get_price_data(benchmark, start_date, end_date)

    def get_history_batch(
        self,
        stock: str,
        count: int,
        field: str,
        end_date: Optional[pd.Timestamp] = None,
    ) -> np.ndarray:
        """按字段批量获取历史序列。"""
        end_str = end_date.strftime("%Y-%m-%d") if end_date is not None else None

        try:
            df = self.api.get_price(
                stock,
                end_date=end_str,
                count=count,
                frequency="1d",
                fields=[field],
                fq="pre",
            )
        except Exception as exc:
            logger.debug("获取历史序列失败: %s %s %s", stock, field, exc)
            return np.array([])

        if df is None or df.empty or field not in df.columns:
            return np.array([])
        return df[field].to_numpy()

    def supports_feature_type(self, feature_type: str) -> bool:
        """判断是否支持指定特征类型。"""
        return feature_type in {"price", "fundamental", "market", "technical"}
