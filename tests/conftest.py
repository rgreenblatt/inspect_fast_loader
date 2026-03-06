import subprocess
import sys
from pathlib import Path

# Ensure the project root is on the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Ensure tests/ is on the path so the helpers module can be imported
tests_dir = Path(__file__).parent
sys.path.insert(0, str(tests_dir))

# Generate test logs if they don't exist
test_logs_dir = project_root / "test_logs"
if not test_logs_dir.exists() or not list(test_logs_dir.glob("*.eval")):
    subprocess.run(
        [sys.executable, str(project_root / "generate_test_logs.py")],
        cwd=str(project_root),
        check=True,
    )
