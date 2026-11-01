"""Learning project: an AI agent that paper-trades crypto futures on a demo venue."""

import os
import sys

# macOS: torch and LightGBM bundle different OpenMP runtimes, which crash when both run
# multi-threaded in one process. Single-threaded OpenMP avoids it (Linux is unaffected).
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
