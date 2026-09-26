"""
注册自定义模块到 ultralytics
用法（训练和推理前都要执行）:
  python -c "import models.register"

必须在 import ultralytics 之后、构建模型之前执行。
"""
import sys

# 1. 导入我们的模块
from models.weighted_fuse import WeightedFuse

# 2. 加到 ultralytics.nn.modules
import ultralytics.nn.modules as _modules
_modules.WeightedFuse = WeightedFuse
_modules.Concat = WeightedFuse

# 3. 关键：替换 ultralytics.nn.tasks 模块里的 Concat 引用
# parse_model 在 tasks 模块里，它用的 Concat 是 tasks 模块级别的导入
import ultralytics.nn.tasks as _tasks
_tasks.Concat = WeightedFuse
_tasks.WeightedFuse = WeightedFuse
# 同时加到 __dict__，确保 globals() 能找到
_tasks.__dict__['WeightedFuse'] = WeightedFuse
_tasks.__dict__['Concat'] = WeightedFuse

# 4. 加到 sys.modules
sys.modules['ultralytics.nn.modules.WeightedFuse'] = WeightedFuse
sys.modules['ultralytics.nn.modules.Concat'] = WeightedFuse

print("[register] WeightedFuse 注册成功（已替换Concat，训练和推理通用）")
