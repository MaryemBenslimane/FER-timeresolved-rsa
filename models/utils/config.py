"""Paths used by utils.load_weights.

Kept separate from the top-level config.py so that importing FaceNet.py (which
pulls in utils.load_weights) does not drag in the whole RSA pipeline config.
"""
import os
from pathlib import Path

# Checkpoints are expected under ./net_weights (override with WEIGHTS_PATH).
weights_path = Path(os.environ.get("WEIGHTS_PATH", "./net_weights"))
