# EditStream-RAG v1.0

Artifact for *[Experiment, Analysis, and Benchmark] The Chunks of Theseus: Re-Embedding Cost and Layout Drift of RAG
Chunkers on Real Edit Histories* (ICDE 2027 submission).

## Redistributed

In the repository:

- code: chunkers and adapters, replay harness, extraction scripts, experiment drivers, Round 2 analyses, plots and
  tables (`src/`, `experiments/`), and tests (`tests/`), under Apache-2.0;
- pinned source manifests (`results/data_versions.json`, `results/cpy_release_tags.json`), calibration
  (`results/calibration.json`) and Bian et al.'s frozen parameters (`results/bian_params.json`);
- generated retrieval questions with their evidence passages (`results/qa/questions_*.jsonl`), the second-model check
  (`results/round2/qcheck.csv`, `qcheck_summary.json`) and a blind 100-question audit sheet
  (`results/round2/audit_sheet.csv`);
- every summary file behind the paper's tables and figures (`results/*.csv`, `results/*.json`, `results/round2/`),
  the resynchronization per-edit counts (`results/round2/resync_edits.parquet`) and the figure PDFs.

As release assets (`gh release` attachments, each well below 2 GB):

- `editstream-rag-per-edit-counts-v1.0.tar.gz`: per-edit S1/S2/S3 counts for all configurations, sync intervals and
  size points (`results/e1_edits.parquet`, `results/e1_edits_tierB.parquet`) and the band-only ablation replay
  (`results/round2/band_edits.parquet`);
- `editstream-rag-runbooks-v1.0.tar.gz`: big-ann-benchmarks runbooks and point maps for REC, SentCDC, DIFF[REC] and
  KARA on the three streams (`results/runbooks/`);
- `editstream-rag-contexts-v1.0.tar.gz`: the packed retrieval contexts of every quality run (`results/qa/contexts*.jsonl`),
  from which the span, answer and anywhere criteria are recomputed.

## Regenerated, not redistributed

- the edit streams (`data/streams/*.parquet`, 1.4 GB), from the pinned Wikipedia dump and git commits by
  `experiments/get_data.sh` and `deltachunk.data.*`;
- raw per-edit chunk keys, runbook vectors (`experiments/runbook_vectors.py`), embeddings and index states;
- LLM outputs for fresh runs (questions, answers, paraphrases, contexts, extraction), which need local model servers.

## Excluded

- **Raw LLM response caches** (`data/llm_cache*.sqlite`, `data/llm_cache_r2_qcheck.jsonl`). They hold the full prompts,
  including copied source text, and the raw responses; their redistribution terms (CC BY-SA share-alike for the copied
  Wikipedia text, model terms for the outputs) have not been reviewed by the authors. Without them, the saved question,
  answer and context files still reproduce the reported retrieval and answer numbers; the RQ3 sentence-probe paraphrases
  exist only in these caches, so replaying the probe experiment needs the caches or a qwen3:8b server.
- The downloaded dump and git clones, the content-addressed vector cache, scoring checkpoints (`results/raw/`), smoke
  outputs, logs, and the manuscript source.

## Round 2 results included

The Round 2 long runs (`experiments/round2_long.sh`, finished 2026-11-10) are part of this release:

- band-only ablation of guarded content-defined chunking (`experiments/r2_band.py`): calibration
  (`results/round2/band_calibration.json`), per-edit replay (`band_edits.parquet`, asset), costs relative to REC at the
  common size with paired bootstrap intervals (`results/round2/band_ablation.csv`, Table VIII of the paper) and the
  fidelity check that a REC replay at E1's parameter reproduces E1's per-edit counts on every edit
  (`results/round2/band_fidelity.json`);
- size-matched retrieval on the common REC background (CPU): `results/e4_quality_r2matched.*`,
  `results/e4_stats_r2matched.*` (harness span hit, McNemar with Holm correction, TOST fields), realized against target
  sizes (`results/round2/size_check_r2matched.json`, `results/round2/matched_params.json`), the span, answer and
  anywhere criteria recomputed from the saved contexts (`results/round2/span_*.csv`, tag `_r2matched`), the direct
  comparisons STRUCT-NM vs. REC at its size and KARA at REC's size vs. nominal KARA (`results/round2/size_pairs.csv`),
  and the packed contexts (`results/qa/contexts_r2matched.jsonl`, contexts asset).

The paper reports the three-criteria differences from `span_pairs.csv`; `e4_stats_r2matched.csv` uses the harness's
saved span hit, which agrees with the recomputed span on at least 99.4% of questions per arm (`span_validation.csv`),
so its differences can differ by a few tenths of a point (for example KARA@match +2.9 instead of +3.2 points).

## Known limitations

- Wikimedia keeps dumps online for a limited time; the dump's SHA-1 is recorded so that a mirror copy can be verified.
  Depositing the derived streams (CC BY-SA 4.0 and CC BY 4.0 / PSF parts) on an archive such as Zenodo would remove this
  dependency; this is an author decision.
- Timings were measured on a shared Apple M3 Ultra and will differ on other hardware; counts are hardware-independent.
