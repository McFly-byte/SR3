import os
import sys
from pathlib import Path

# Make the repository root importable when running pytest from anywhere.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))
