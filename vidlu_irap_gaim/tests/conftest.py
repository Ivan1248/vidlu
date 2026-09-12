import os

# Data-loader workers default to a module-level constant read at import time; worker
# processes are slow to spawn on Windows and unnecessary for the small test datasets.
os.environ.setdefault("VIDLU_NUM_WORKERS", "0")
