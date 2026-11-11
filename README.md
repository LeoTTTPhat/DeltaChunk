# Chunks of Theseus / EditStream-RAG — reproduction artifact

This directory contains a self-contained reproduction workspace for Chunks of Theseus / EditStream-RAG.

## Contents

| Path | Purpose |
|---|---|
| `P3_DeltaChunk/src/` | Implementation and experiment libraries |
| `P3_DeltaChunk/tests/` | Correctness and regression checks |
| `P3_DeltaChunk/results/` | Saved measurements and derived results |
| `tools/` | Shared resource supervisors and plotting utilities |
| `setup.sh` | Create the Python environment from recorded dependencies |
| `reproduce.sh` | Run the default saved-result analysis, tests, or a chosen driver |
| `MANIFEST.json`, `verify.py` | File sizes, SHA-256 hashes, and integrity checking |

## Quick start

Use Python 3.12, uv, and a POSIX shell. The native measurements were collected on Apple Silicon macOS; CPU-only analyses can also run on compatible environments.

```bash
python3 verify.py --full
bash setup.sh
bash reproduce.sh analysis
bash reproduce.sh test
```

`test` runs the core checks. `test-all` runs the complete included suite and can require native services or engine environments. Use `bash guarded_run.sh run <driver.py> [arguments]` for a supervised fresh experiment.

Run these commands from this directory. `setup.sh` installs the recorded environment; network access is needed for dependency installation. `ARTIFACT_PYTHON=/absolute/path/to/python` can select an existing compatible interpreter for the reproduction launcher. Relative paths are resolved inside the bundled source directory. Analyses can update derived outputs; use a fresh copy to repeat integrity verification against the original inventory.

## Data and full experiments

The bundle contains saved per-edit counts, extracted document streams, runbooks and point maps, generated questions, saved retrieval contexts, and Round 2 results. The default analysis reads these measurements without model inference.

To reconstruct upstream inputs from pinned public sources, run from the source directory:

```bash
bash experiments/get_data.sh all
../.venv/bin/python -m deltachunk.data.wiki_extract scan
../.venv/bin/python -m deltachunk.data.wiki_extract convert
../.venv/bin/python -m deltachunk.data.wiki_extract merge
../.venv/bin/python -m deltachunk.data.git_extract k8s --procs 2
../.venv/bin/python -m deltachunk.data.git_extract cpy --procs 2
```

Source revisions and checksums are in `results/data_versions.json` and `results/cpy_release_tags.json`. Model-based retrieval and timing require the recorded embedding models and local llama-server endpoints: qwen3:8b on port 8301 and the second-model question check on port 8302. Digests and serving parameters are in the bundled provenance. CPU paths can be selected with `--device cpu`; original GPU timing depends on hardware. Full stages are selected with `STEPS` in `experiments/run_all.sh`; the frozen questions can be reused by selecting measurement stages directly. `experiments/round2_long.sh` runs the additional ablation and matched-retrieval grids.

## Resources and outputs

The default analysis uses saved measurements. Full experiments can require local models, native services, significant disk space, and hours of compute. Run one heavy campaign at a time. The supervisors in `tools/` preserve the original process-memory ceiling of 200 GB and host ceiling of 490 GB; select a smaller job limit for smaller machines. Saved timing values describe the measured hardware. Replays preserve the recorded seeds, configurations, and input identities; fresh concurrent or model-server timing measurements can vary.

The source directory retains its original experiment and result paths. New experiments should use separate output directories where the driver supports them. Dependency locks, source revisions, input hashes, model identifiers, and run metadata are retained alongside the corresponding measurements.

Before using a full driver that invokes `tools/memguard.py`, run `bash tools/ensure_memwatch.sh` from this directory on macOS. The supervisor refuses heavy jobs if its accounting watchdog is absent. On other platforms, launch individual drivers through `tools/memory_guard.py` with a memory limit suited to that machine.

## Clone and retrieve data

Large data files use Git LFS. Install Git LFS before cloning, then retrieve all data objects:

```bash
git lfs install
git clone https://github.com/LeoTTTPhat/DeltaChunk.git
cd DeltaChunk
git lfs pull
python3 verify.py --full
```
