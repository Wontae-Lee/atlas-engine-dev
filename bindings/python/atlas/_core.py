import sys
from types import ModuleType

from ._engine import load_engine


_native = load_engine()
__version__ = _native.__version__
engine = _native.engine
__path__ = []

for _name, _value in vars(_native).items():
    if isinstance(_value, ModuleType):
        globals()[_name] = _value
        sys.modules[f"{__name__}.{_name}"] = _value
