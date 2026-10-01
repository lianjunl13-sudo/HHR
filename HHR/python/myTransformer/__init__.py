from . import cache
from . import models
try:
    import KVLib as capi
except (ImportError, OSError):
    # Dataset-only portable backends do not call the fused runtime extension.
    capi = None
