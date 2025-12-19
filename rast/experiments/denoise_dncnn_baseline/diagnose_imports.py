"""
RAST Import Diagnostics Script
===============================

Run this from anywhere in the rast/ project to diagnose import issues.

Usage:
    cd /workspace/rast
    python experiments/denoise_dncnn_baseline/diagnose_imports.py
    
Or:
    cd /workspace/rast/experiments/denoise_dncnn_baseline
    python diagnose_imports.py
"""

import sys
from pathlib import Path

print("="*70)
print("RAST Import Diagnostics")
print("="*70)

# Detect project structure
script_path = Path(__file__).resolve()
print(f"\n[1] Script location: {script_path}")

# Try to find project root
current = script_path.parent
project_root = None
for _ in range(5):  # Search up to 5 levels
    if (current / "core").exists() and (current / "experiments").exists():
        project_root = current
        break
    current = current.parent

if project_root:
    print(f"    [PASSED] Project root found: {project_root}")
else:
    print(f"    [ERROR] Could not find project root (looking for core/ and experiments/)")
    sys.exit(1)

baseline_dir = project_root / "experiments" / "denoise_dncnn_baseline"
if not baseline_dir.exists():
    print(f"    [ERROR] Baseline directory not found: {baseline_dir}")
    sys.exit(1)

print(f"    [PASSED] Baseline directory: {baseline_dir}")

# Check Python path
print(f"\n[2] Python sys.path (first 5):")
for i, p in enumerate(sys.path[:5]):
    marker = "[PASSED]" if Path(p) == project_root else " "
    print(f"    {marker} {i}: {p}")

if str(project_root) not in sys.path:
    print(f"    [WARNING]  Project root NOT in sys.path, adding it...")
    sys.path.insert(0, str(project_root))

# Check core structure
print(f"\n[3] Checking core/ directory:")
core_dir = project_root / "core"
core_files = ['engine.py', 'student.py', 'teacher.py', 'assembler.py', 'student_pipeline.py']

for cf in core_files:
    cf_path = core_dir / cf
    if cf_path.exists():
        size = cf_path.stat().st_size
        print(f"    [PASSED] core/{cf} ({size:,} bytes)")
    else:
        print(f"    [ERROR] core/{cf} - MISSING!")

# Check utils structure
print(f"\n[4] Checking utils/ directory:")
utils_dir = project_root / "utils"
if utils_dir.exists():
    print(f"    [PASSED] utils/ directory found")
    rm_path = utils_dir / "result_manager.py"
    if rm_path.exists():
        print(f"      [PASSED] result_manager.py ({rm_path.stat().st_size:,} bytes)")
    else:
        print(f"      [ERROR] result_manager.py - MISSING!")
else:
    print(f"    [ERROR] utils/ directory not found at: {utils_dir}")

# Check baseline structure
print(f"\n[5] Checking denoise_dncnn_baseline/ structure:")
baseline_subdirs = {
    'pipeline': ['denoise_dncnn_pipeline.py'],
    'module/h_gate': ['denoise_hallucination_gate.py'],
    'module/e_grader': ['e_mc_dropout.py'],
    'networks/dncnn': ['model.py'],
    'networks/vae': ['model.py'],
    'checkpoints': [],
    'data/test': [],
}

for subdir, files in baseline_subdirs.items():
    subdir_path = baseline_dir / subdir
    if subdir_path.exists():
        print(f"    ✓ {subdir}/")
        for f in files:
            f_path = subdir_path / f
            if f_path.exists():
                print(f"      [PASSED] {f} ({f_path.stat().st_size:,} bytes)")
            else:
                print(f"      [ERROR] {f} - MISSING!")
    else:
        print(f"    ✗ {subdir}/ - MISSING!")

# Check for __init__.py files
print(f"\n[6] Checking __init__.py files:")
init_locations = [
    project_root / "core",
    project_root / "utils",
    baseline_dir / "pipeline",
    baseline_dir / "module",
    baseline_dir / "module" / "h_gate",
    baseline_dir / "module" / "e_grader",
    baseline_dir / "module" / "a_grader",
    baseline_dir / "networks",
    baseline_dir / "networks" / "dncnn",
    baseline_dir / "networks" / "vae",
]

missing_init = []
for loc in init_locations:
    init_file = loc / "__init__.py"
    rel_path = init_file.relative_to(project_root)
    if init_file.exists():
        print(f"    [PASSED] {rel_path}")
    else:
        print(f"    [ERROR] {rel_path} - MISSING!")
        missing_init.append(init_file)

# Check test datasets
print(f"\n[7] Checking test datasets:")
test_data_dir = baseline_dir / "data" / "test"
if test_data_dir.exists():
    print(f"    [PASSED] data/test/ found")
    for dataset in ['bsd68', 'set12', 'urban100']:
        ds_path = test_data_dir / dataset
        if ds_path.exists():
            image_count = len(list(ds_path.glob("*.png")) + list(ds_path.glob("*.jpg")))
            print(f"      [PASSED] {dataset}/ ({image_count} images)")
        else:
            print(f"      [ERROR] {dataset}/ - MISSING!")
else:
    print(f"    [ERROR] data/test/ not found at: {test_data_dir}")

# Check checkpoints
print(f"\n[8] Checking model checkpoints:")
ckpt_dir = baseline_dir / "checkpoints"
if ckpt_dir.exists():
    ckpts = list(ckpt_dir.glob("*.pth")) + list(ckpt_dir.glob("*.pt"))
    if ckpts:
        print(f"    [PASSED] Found {len(ckpts)} checkpoint(s):")
        for ckpt in sorted(ckpts)[:5]:  # Show first 5
            size_mb = ckpt.stat().st_size / 1024 / 1024
            print(f"      - {ckpt.name} ({size_mb:.1f} MB)")
    else:
        print(f"    [WARNING]  No .pth or .pt files found in checkpoints/")
else:
    print(f"    [ERROR] checkpoints/ directory not found")

# Test imports
print(f"\n[9] Testing critical imports:")

test_imports = [
    ('core.engine', 'Core module'),
    ('core.teacher', 'Core module'),
    ('core.student', 'Core module'),
    ('utils.result_manager', 'Utils module'),
    ('experiments.denoise_dncnn_baseline.pipeline.denoise_dncnn_pipeline', 'Pipeline'),
    ('experiments.denoise_dncnn_baseline.module.h_gate.denoise_hallucination_gate', 'H-Gate'),
    ('experiments.denoise_dncnn_baseline.networks.dncnn.model', 'DnCNN'),
    ('experiments.denoise_dncnn_baseline.networks.vae.model', 'VAE'),
]

import_errors = []
for module_name, description in test_imports:
    try:
        __import__(module_name)
        print(f"    [PASSED] {module_name}")
        print(f"      ({description})")
    except ImportError as e:
        print(f"    [ERROR] {module_name}")
        print(f"      Error: {e}")
        import_errors.append((module_name, str(e)))
    except Exception as e:
        print(f"    [WARNING]  {module_name}")
        print(f"      Unexpected error: {e}")

# Summary and recommendations
print(f"\n{'='*70}")
print("SUMMARY & RECOMMENDATIONS")
print(f"{'='*70}")

issues = []

if missing_init:
    issues.append("missing __init__.py files")
    print(f"\n[WARNING]  Missing {len(missing_init)} __init__.py file(s)")
    print(f"\nTo fix, run from project root:")
    print(f"    cd {project_root}")
    for init_file in missing_init:
        rel_path = init_file.relative_to(project_root)
        print(f"    touch {rel_path}")

if import_errors:
    issues.append("import errors")
    print(f"\n[WARNING]  {len(import_errors)} import error(s) detected")
    for module, error in import_errors:
        print(f"    - {module}")
        print(f"      {error}")

if not issues:
    print(f"\n[PASSED] All checks passed!")
    print(f"\nYou can now run experiments:")
    print(f"    cd {baseline_dir}")
    print(f"    python script/run_experiment.py --dataset set12 --noise_sigma 25")
else:
    print(f"\n[ERROR] Found issues: {', '.join(issues)}")
    print(f"\nPlease fix the above issues before running experiments.")

print(f"\n{'='*70}")