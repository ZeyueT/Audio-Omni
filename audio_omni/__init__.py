from .models.factory import create_model_from_config, create_model_from_config_path
from .models.pretrained import get_pretrained_model

def __getattr__(name):
    if name == "AudioOmni":
        from .api import AudioOmni
        return AudioOmni
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")