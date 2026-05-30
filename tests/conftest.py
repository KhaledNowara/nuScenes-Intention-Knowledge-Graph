"""Put the experiment root on sys.path so tests can import data_layer/domain/graph_exporter."""

from __future__ import annotations

import sys
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parent.parent
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))
