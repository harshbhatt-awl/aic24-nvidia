# Command reference (runbook)

Full-flag invocations for operating the aic24-nvidia pipeline. Run everything
from the repo root with the project venv active (or call `.venv/bin/python`
directly, as shown). Conventions:

```bash
cd /home/harsh1/github-repos/awl-research/aic24-nvidia
source .venv/bin/activate        # or prefix commands with .venv/bin/python
```

`<CONFIG>` below is a config file, almost always `configs/baseline.yaml`.
`<RUN_ID>` is a run directory name under `outputs/` (auto-generated as
`<config-stem>_YYYYmmdd_HHMMSS` when omitted — second resolution, so same-minute
runs don't collide).

---

## 1. Setup / bootstrap

```bash
# Clone + pin + patch the upstream siblings (YACHIYO @ f881fe0, TrackEval @ 12c8791).
# Idempotent: skips an existing clone. Auto-applies patches/*.patch and verifies them.
bash scripts/bootstrap_external.sh

# Verify the vendored upstream patches are present (run standalone any time).
bash scripts/verify_patches.sh

# Install the package + dev tools (ruff, mypy, pytest) into the venv.
pip install -e ".[dev]"

# Exact, reproducible environment (Python 3.14, torch 2.12+cu130):
pip install -r requirements.lock
```

---

## 2. Pipeline (`pipeline.py`)

Subcommands: `adapt frames detect reid pose sct mct evaluate all viz dashboard bootstrap`.
Stage/`all` subcommands take `--config` (required), `--run-id`, `--force`.

```bash
# Full pipeline, fresh auto-timestamped run-id (all 8 stages from scratch).
python pipeline.py all --config configs/baseline.yaml

# Full pipeline into an explicit run dir (e.g. rebuild the locked baseline).
python pipeline.py all --config configs/baseline.yaml --run-id baseline

# A single stage (manifest-gated: skips if already done, refuses if upstream missing).
python pipeline.py detect --config configs/baseline.yaml --run-id baseline

# Force a stage to re-run even if its manifest says ok (downstream stages then re-run too).
python pipeline.py sct --config configs/baseline.yaml --run-id baseline --force

# Stage order: adapt -> frames -> detect -> reid -> pose -> sct -> mct -> evaluate
# (single source of truth: aic24_nvidia/registry.py)

# Bootstrap via the CLI (delegates to scripts/bootstrap_external.sh).
python pipeline.py bootstrap

# Visualization videos (only detect / sct / mct are wired).
python pipeline.py viz --config configs/baseline.yaml --run-id baseline --stage detect
python pipeline.py viz --config configs/baseline.yaml --run-id baseline --stage sct
python pipeline.py viz --config configs/baseline.yaml --run-id baseline --stage mct

# Read-only Streamlit dashboard (default port 8501).
python pipeline.py dashboard --port 8501
```

### Rebuild the locked baseline (cached detect/reid/pose reused, ~5 min)

```bash
rm -rf outputs/baseline/{sct,mct,evaluate}
python pipeline.py all --config configs/baseline.yaml --run-id baseline
```

> NOTE: this reuses the cached detect/reid/pose stages via manifest gating. To
> re-run the *model* stages (e.g. after a backend change), use a fresh run-id or
> `--force` from `detect` — see §5.

---

## 3. Experiment harness (`experiments/`)

Global flag: `--registry experiments/registry.yaml` (default). Variants inherit
upstream stages (before `rerun_from`) from `outputs/baseline/` via symlinks.

```bash
# Build outputs/baseline/ once (~45 min full run; refuses if it already exists).
python experiments/run.py ensure-baseline
python experiments/run.py ensure-baseline --config configs/baseline.yaml --force   # rebuild from scratch

# List defined experiments (id, #variants, rerun_from, description).
python experiments/run.py list

# Show which variants have completed.
python experiments/run.py status

# Run every variant of one experiment.
python experiments/run.py run eps_mcpt_sweep

# Run a single variant; --force rebuilds; --stop-on-failure halts on first failure.
python experiments/run.py run eps_mcpt_sweep --variant 0.30 --force --stop-on-failure

# Use an alternate registry file.
python experiments/run.py --registry experiments/registry.yaml run eps_mcpt_sweep

# Compare results vs baseline. --sort-by <metric>, --experiment <id> to filter,
# --include-incomplete to show partial runs, --markdown <path> to write a table.
python experiments/compare.py
python experiments/compare.py --sort-by mct_world.HOTA
python experiments/compare.py --experiment eps_mcpt_sweep --sort-by image.HOTA --markdown /tmp/compare.md
python experiments/compare.py --include-incomplete
```

---

## 4. Tests & quality gates

```bash
# Fast unit suite (GPU-free; what CI runs).
.venv/bin/python -m pytest tests/unit -q

# A single test file / test.
.venv/bin/python -m pytest tests/unit/test_stage_registry.py -q
.venv/bin/python -m pytest tests/unit/test_paths.py::test_make_run_id_format -v

# Integration (adapter on a synthetic 2-cam fixture; full-pipeline path is skipped).
.venv/bin/python -m pytest tests/integration -q

# Everything.
.venv/bin/python -m pytest tests/ -q

# Lint (high-signal rules, blocking) + type-check (informational).
.venv/bin/ruff check .
.venv/bin/mypy aic24_nvidia

# Byte-compile a set of modules (quick sanity that nothing is syntactically broken).
.venv/bin/python -m py_compile aic24_nvidia/models/*.py aic24_nvidia/stages/*.py aic24_nvidia/config.py
```

---

## 5. Behavior-preservation check (after a refactor)

A fresh full run exercises every stage from scratch (no cache), so it actually
re-runs the model backends and the tracker — then diff its metrics against the
locked `outputs/baseline/`:

```bash
# Fresh full run (auto-timestamped run-id), logged.
.venv/bin/python pipeline.py all --config configs/baseline.yaml 2>&1 | tee /tmp/aic24_baseline_verify.log

# Compare the new run's metrics against the locked baseline (substitute the run-id printed above).
python - <<'PY'
import json, glob, os
new = sorted(glob.glob("outputs/baseline_*/evaluate/metrics.json"))[-1]
old = "outputs/baseline/evaluate/metrics.json"
a, b = json.load(open(new)), json.load(open(old))
print("new:", new)
for k in sorted(set(a) | set(b)):
    print(f"  {k}: new={a.get(k)}  baseline={b.get(k)}")
PY
```

If the metrics match, the refactor is behavior-preserving. To swap a model
backend instead (Phase 2), set `detect.model_name` / `reid.model_name` /
`pose.model_name` (and optionally `<stage>.weights`) in the config — `load_config`
rejects an unknown name.

---

## 6. Git workflow used this session

```bash
# Feature work on a branch off main; one logical commit per task.
git checkout -b <phaseN-name>
git add <exact paths>            # stage only the task's files
git commit -F - <<'EOF'
<type>(<scope>): <subject>

<body>

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF

# Finish: fast-forward merge the stacked branch into main, verify, delete merged branches.
git checkout main
git merge --ff-only <phaseN-name>
.venv/bin/python -m pytest tests/unit -q     # verify on the merged result
git branch -d <phaseN-name>

# Pushing to harshbhatt-awl/aic24-nvidia returns 403 by default — switch the gh
# account first (see the gh-multi-account memory), then:
# git push origin main
```
