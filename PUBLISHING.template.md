# Publishing EditStream-RAG (for the authors)

Nothing in this folder has been pushed or published. The authors publish the release themselves and send the URL to
the orchestrator, who inserts it into the paper (`\artifacturl` in `paper/main.tex`).

## Suggested repository name

`EditStream-RAG` (public repository under the authors' account or organization). A DOI-backed archive (for example the
Zenodo GitHub integration, enabled before creating the release) gives the persistent identifier the ICDE EAB track asks
for.

## Commands

Run from the `P3_DeltaChunk` directory after completing the checklist below:

```bash
gh repo create EditStream-RAG --public --source release/EditStream-RAG --push
cd release/EditStream-RAG
gh release create {{VERSION}} {{ASSET_ARGS}} --notes-file RELEASE_NOTES.md
```

## Release assets

| File | Size | SHA-256 | Contents |
|---|---|---|---|
{{ASSET_TABLE}}

The archives live in `release/assets/`, outside the repository. `CHECKSUMS` lists the repository files and the archives;
`CHECKSUMS.assets` lists the files inside the archives (check with `shasum -a 256 -c CHECKSUMS.assets` after unpacking).

## Pre-publication checklist

- [ ] **Round 2 results.** Wait for `results/round2/LONG_DONE`, then rerun `python3 experiments/build_release.py` so that
      the band-ablation and size-matched retrieval outputs are included (the repository is rebuilt with one commit).
- [ ] **License.** Confirm Apache-2.0 for the code and set the copyright holder in `LICENSE` (currently "The
      EditStream-RAG authors").
- [ ] **Wikipedia attribution and share-alike.** Confirm that `DATA_STATEMENT.md` meets CC BY-SA 4.0: derived WIKI
      text (questions, evidence passages, contexts, audit sheet) is released under CC BY-SA 4.0 with attribution through
      the page ids; keep that license on any redistribution. Kubernetes text is CC BY 4.0 and CPython text PSF-licensed.
- [ ] **Cached model responses.** Decide whether to publish the raw LLM caches (`data/llm_cache*.sqlite`,
      `data/llm_cache_r2_qcheck.jsonl`; excluded so far). They contain full prompts with copied source text and the raw
      qwen3:8b and gemma3:27b responses, including the RQ3 sentence-probe paraphrases, which exist nowhere else. If
      published, attach them as a separate asset under CC BY-SA 4.0 and note the model terms.
- [ ] **Derived streams.** Optionally deposit `data/streams/*.parquet` (1.4 GB) on Zenodo, since Wikimedia removes old
      dumps; otherwise the dump SHA-1 lets users verify a mirror copy.
- [ ] **Author names.** Add the authors to `README.md` (a citation block) and to the copyright line, and replace the
      placeholder author block in the paper.
- [ ] **Paper link.** After publishing, send the repository or DOI URL to the orchestrator for `\artifacturl`.
