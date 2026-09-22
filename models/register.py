"""
注册自定义模块到 ultralytics
用法: python -c "import models.register"
之后 yaml 里写 - WeightedFuse 就能被解析
"""
import sys

# 1. 导入我们的模块
from models.weighted_fuse import WeightedFuse

# 2. 把 WeightedFuse 加到 ultralytics.nn.modules 的命名空间
import ultralytics.nn.modules as _modules
_modules.WeightedFuse = WeightedFuse

# 3. 关键：把 Concat 替换成 WeightedFuse
# 这样 parse_model 里的 `m is Concat` 判断会为 True，
# 会按 Concat 的方式计算输出通道数（sum of input channels），
# 但实际创建的是 WeightedFuse 实例
_modules.Concat = WeightedFuse

# 4. 加到 sys.modules
sys.modules['ultralytics.nn.modules.WeightedFuse'] = WeightedFuse
sys.modules['ultralytics.nn.modules.Concat'] = WeightedFuse

# 5. 也加到 ultralytics.nn.tasks 的命名空间
import ultralytics.nn.tasks as _tasks
if not hasattr(_tasks, 'WeightedFuse'):
    setattr(_tasks, 'WeightedFuse', WeightedFuse)

print("[register] WeightedFuse 注册成功（已替换Concat）")
