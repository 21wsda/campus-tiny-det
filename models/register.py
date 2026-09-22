"""
注册自定义模块到 ultralytics
用法: python -c "import models.register"
"""
import sys

# 1. 导入我们的模块
from models.weighted_fuse import WeightedFuse

# 2. 加到 ultralytics.nn.modules 的命名空间
import ultralytics.nn.modules as _modules
_modules.WeightedFuse = WeightedFuse

# 3. 加到 sys.modules
sys.modules['ultralytics.nn.modules.WeightedFuse'] = WeightedFuse

print("[register] WeightedFuse 注册成功")
