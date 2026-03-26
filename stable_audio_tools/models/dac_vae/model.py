"""
重新导出外部 dac 包的 DAC 和 DACFile 类
这样可以避免在 dac_vae/__init__.py 中直接导入外部包导致的循环导入问题
"""
from dac.model.dac import DAC
from dac import DACFile  # DACFile 在 dac 包的顶层

__all__ = ['DAC', 'DACFile']

