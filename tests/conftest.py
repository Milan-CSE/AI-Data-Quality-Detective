import os
import sys

# Ensure project root is available to pytest regardless of runner environment
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
