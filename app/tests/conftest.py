# tests/conftest.py
import pytest
from pathlib import Path
# 自动创建缺省目录
DEFAULT_DATA_DIR = Path(__file__).parent / "data"
DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
def pytest_addoption(parser):
    parser.addoption("--test-data", 
                    action="store", 
                    default=Path(__file__).parent / "data",
                    help="测试数据目录路径")

def pytest_generate_tests(metafunc):
    if "test_case" in metafunc.fixturenames:
        # 转换为 Path 对象并解析绝对路径
        data_dir = Path(metafunc.config.getoption("--test-data")).resolve()
        
        # 验证目录是否存在
        if not data_dir.exists():
            raise FileNotFoundError(f"测试数据目录不存在: {data_dir}")
            
        # 收集测试用例
        test_cases = [d for d in data_dir.iterdir() if d.is_dir()]
        metafunc.parametrize("test_case", test_cases, ids=[d.name for d in test_cases])