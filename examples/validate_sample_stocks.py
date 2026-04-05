# -*- coding: utf-8 -*-
"""
验证样例牛股配置

功能：
1. 验证样例股票池配置
2. 验证股票代码格式
3. 验证板块分布
4. 验证数据完整性
"""

import sys
from pathlib import Path

import pandas as pd

from simtrademl.data_sources.simtradelab_source import SimTradeLabDataSource


SAMPLE_STOCKS = [
    "002815.SZ",  # 崇达技术
    "300476.SZ",  # 胜宏科技
    "301232.SZ",  # 飞沃科技
    "301377.SZ",  # 鼎泰高科
    "000975.SZ",  # 银泰黄金
    "603226.SH",  # 菲林格尔
]


def get_board_type(stock_code: str) -> str:
    """
    根据股票代码判断板块类型
    
    Args:
        stock_code: 股票代码
    
    Returns:
        板块类型
    """
    if stock_code.startswith("300") or stock_code.startswith("301"):
        return "创业板"
    elif stock_code.startswith("688"):
        return "科创板"
    elif stock_code.startswith("000") or stock_code.startswith("002"):
        return "深圳主板"
    elif stock_code.startswith("600") or stock_code.startswith("601") or stock_code.startswith("603"):
        return "上海主板"
    else:
        return "其他"


def validate_sample_stocks():
    """验证样例股票配置"""
    print("=" * 60)
    print("验证样例牛股配置")
    print("=" * 60)
    
    print(f"\n样例股票池 ({len(SAMPLE_STOCKS)}只):")
    for stock in SAMPLE_STOCKS:
        board = get_board_type(stock)
        print(f"  - {stock}: {board}")
    
    print("\n板块分布统计:")
    boards = {}
    for stock in SAMPLE_STOCKS:
        board = get_board_type(stock)
        boards[board] = boards.get(board, 0) + 1
    
    for board, count in sorted(boards.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {board}: {count}只 ({count/len(SAMPLE_STOCKS)*100:.1f}%)")
    
    print("\n数据完整性验证:")
    data_source = SimTradeLabDataSource(required_data={"price"})
    
    valid_count = 0
    for stock in SAMPLE_STOCKS:
        try:
            price_df = data_source.get_price_data(stock)
            if price_df.empty:
                print(f"  ✗ {stock}: 无数据")
            else:
                date_range = f"{price_df.index[0].date()} - {price_df.index[-1].date()}"
                print(f"  ✓ {stock}: {len(price_df)}条记录 ({date_range})")
                valid_count += 1
        except Exception as e:
            print(f"  ✗ {stock}: 获取失败 - {str(e)}")
    
    print(f"\n验证结果：{valid_count}/{len(SAMPLE_STOCKS)} 只股票数据完整")
    
    if valid_count == len(SAMPLE_STOCKS):
        print("\n✓ 所有样例股票数据完整，配置有效")
        return True
    else:
        print(f"\n✗ 有 {len(SAMPLE_STOCKS) - valid_count} 只股票数据不完整")
        return False


def main():
    """主函数"""
    try:
        success = validate_sample_stocks()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n验证过程出错：{str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
