import sys
from pathlib import Path

# Ensure the project root is on the path so generate_test_logs can be imported
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
