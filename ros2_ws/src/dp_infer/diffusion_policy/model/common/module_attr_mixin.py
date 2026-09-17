"""提供模型参数设备和数据类型访问。

该模块从 model/dp 移植，保持 checkpoint 所用导入路径。
"""

import torch.nn as nn


class ModuleAttrMixin(nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy_variable = nn.Parameter(requires_grad=False)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
