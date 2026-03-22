# Geometric loss functions between point clouds, images and volumes

Please check our [website](https://www.kernel-operations.io/geomloss)!

## Packaging and Release Checklist

This project now uses `pyproject.toml` with setuptools (PEP 517).

1. Clean previous build artifacts:

```bash
rm -rf build dist geomloss.egg-info
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
```

2. Build distribution artifacts:

```bash
python -m pip install --upgrade build
python -m build
```

3. Inspect generated metadata (optional quick check):

```bash
python - <<'PY'
import zipfile
from pathlib import Path
wheel = max(Path("dist").glob("geomloss-*.whl"), key=lambda p: p.stat().st_mtime)
with zipfile.ZipFile(wheel) as zf:
	meta = zf.read(next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))).decode()
for line in meta.splitlines():
	if line.startswith(("Name:", "Version:", "Requires-Python:", "Requires-Dist:", "Provides-Extra:")):
		print(line)
PY
```

4. Test local installs:

```bash
python -m pip install .
python -m pip install ".[full]"
```

5. Smoke-test import:

```bash
python -c "import geomloss; print(geomloss.__version__)"
```
