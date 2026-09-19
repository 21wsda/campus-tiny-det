"""
注册自定义模块到 ultralytics
用法: python -c "import models.register"
之后 yaml 里写 - WeightedFuse 就能被解析
"""
from ultralytics.nn.tasks import parse_model  # noqa: F401  确保 parse_model 已加载
from models.weighted_fuse import WeightedFuse

# 把 WeightedFuse 加进 ultralytics 的模块表
import ultralytics.nn.modules as _modules
import sys

_modules.WeightedFuse = WeightedFuse
sys.modules['ultralytics.nn.modules.WeightedFuse'] = WeightedFuse

print("[register] WeightedFuse 注册成功")
