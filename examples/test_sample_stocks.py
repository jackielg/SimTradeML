# -*- coding: utf-8 -*-
"""
验证样例股票配置
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.trends_up_model import SAMPLE_STOCKS, USE_SAMPLE_STOCKS


def test_sample_stocks():
    """测试样例股票配置"""
    print("=" * 60)
    print("样例股票配置验证")
    print("=" * 60)
    
    print(f"\n样例股票开关：USE_SAMPLE_STOCKS = {USE_SAMPLE_STOCKS}")
    print(f"\n样例股票池 ({len(SAMPLE_STOCKS)}只):")
    
    for stock in SAMPLE_STOCKS:
        # 判断股票类型
        if stock.startswith('6'):
            market = "上海主板"
        elif stock.startswith('0'):
            market = "深圳主板"
        elif stock.startswith('3'):
            market = "创业板 (不推荐)"
        elif stock.startswith('68'):
            market = "科创板 (不推荐)"
        else:
            market = "未知"
        
        print(f"  - {stock}: {market}")
    
    # 验证是否都是主板股票
    main_board_stocks = [
        s for s in SAMPLE_STOCKS 
        if s.startswith('6') or s.startswith('0')
    ]
    
    print(f"\n主板股票数量：{len(main_board_stocks)}/{len(SAMPLE_STOCKS)}")
    
    if len(main_board_stocks) == len(SAMPLE_STOCKS):
        print("✓ 所有样例股票都是主板股票")
    else:
        print("✗ 警告：样例股票中包含非主板股票")
    
    print("\n" + "=" * 60)
    print("验证完成")
    print("=" * 60)


if __name__ == "__main__":
    test_sample_stocks()
