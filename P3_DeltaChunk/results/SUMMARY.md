# P3 EditStream-RAG: results summary

Generated 2026-11-11 20:56:23 (+07) by `experiments/summary.py --tag ''` at commit b1a7929. Every number below is read or computed from the named file in results/; sections marked PENDING have no result yet. Hardware: Apple M3 Ultra (shared by three projects), <= 8 worker threads per job (multi-process drivers: 1 thread per worker), bge-small-en-v1.5 fp16 on MPS unless stated; versions in each file's `meta`.

## Paper-facing notes (binding)

- **Streams.** K8S and CPY are main-branch (first-parent) histories: a merged PR is one edit. K8S has **284 files** with >= 20 first-parent changes (not 300), 15,208 versions; CPY 300 files, 26,117 versions (results/e0_k8s_stats.json, results/e0_cpy_stats.json).
- **Size calibration.** T=128 is a nominal target calibrated separately per stream. Achieved sizes vary by chunker and stream, including on the E4 final snapshots. At T=200 SentCDC (WIKI, K8S), STRUCT-M and GCDC (WIKI) and BCDC (K8S) cannot reach the target, so T=200 rows are not size-matched; achieved per-stream mean chunk sizes are printed next to every count. E4/E9 quality runs use T=128 (runner session 1 fix); the sync-interval, scope (S2/S3) and regime analyses exist at T=200 only (calibrated, achieved sizes stated). `results/e1_matched_size.csv` interpolates cost over the T sweep at common sizes.
- **toks_frac** = re-embedded tokens / tokens of the new document version (Bian et al.'s workload fraction only in T-rep as `toks_frac_workload`).
- **CPY release axis** is labelled **'sync at release dates'**: main's file state at the timestamps of `v3.*` tags (mostly maintenance-branch tags), not the release branches' contents.
- **Theorem 1 denominators.** Guard-split edits are excluded from the bound check; every Theorem 1 statement must give `covered_edits` / `edits` per stream and r (table in E2 below).
- **TOST rule.** Equivalence (2 pp margin) is declared only if the Newcombe paired 90% interval lies in [-0.02, 0.02], >= 1 discordant pair exists, and n >= n_required (power 0.8, discordance floored at 5%: n >= 773-897 pairs). With a few hundred questions per stratum, 'equivalent' is mostly unattainable: report CIs and say so.
- **Steadlith zero-token words.** Words made only of characters the bge tokenizer drops (U+200E, control characters) have 0 tokens, which Steadlith rejects; such versions are re-chunked with max(1, ntok(word)) (experiments/common.py; first seen on 4 WIKI documents, 90 versions). State it with the STL baseline.
- **E5c** runs one client thread with a 45-minute cap per invocation (the full 200 chunks x 3 reps would take ~4 h); report the completed calls/documents/repetitions printed in the E5 section.
- Counts are primary (hardware independent); times are measured under contention on a shared M3 Ultra; derived costs are labelled 'derived'.

## E0 / T1: EditStream-RAG statistics

| stream | docs | versions | edits | share_noop | share_ws | share_structure | share_single | share_multi | l_old_p50 | l_new_p50 | l_new_p95 | doc_tok_mean | edits_per_doc_day | time_span_days |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cpy | 300 | 26,117 | 25,817 | 0.000 | 0.018 | 0.000581 | 0.566 | 0.416 | 1.000 | 2.000 | 27.000 | 10449.060 | 0.008 | 7243.200 |
| k8s | 284 | 15,208 | 14,924 | 0.000 | 0.015 | 0.000536 | 0.592 | 0.392 | 1.000 | 2.000 | 39.000 | 5419.243 | 0.014 | 3923.000 |
| wiki | 1,000 | 83,949 | 82,949 | 0.000 | 0.002 | 0.005 | 0.763 | 0.231 | 1.000 | 1.000 | 14.000 | 2095.026 | 0.011 | 8574.800 |

Token deciles per document: cpy: [1271, 2208, 3341, 4217, 6044, 8259, 10949, 16615, 26897]; k8s: [997, 1490, 1935, 2391, 3041, 3926, 4823, 6923, 9998]; wiki: [582, 733, 884, 1074, 1302, 1602, 2132, 2918, 4463]

Files: `results/t1_streams.csv`, `results/t1_streams.meta.json`, `results/e0_filter_stats.json`, `results/e0_k8s_stats.json`, `results/e0_cpy_stats.json`, `results/data_versions.json`

## E1: re-embedding cost per edit (counts)

Rows: 8,372,430 (chunker x edit x T/interval). Run: time=2026-11-09 20:12:50; threads={"P3_THREADS": 8, "RAYON_NUM_THREADS": "8", "TOKENIZERS_PARALLELISM": null, "OMP_NUM_THREADS": "8"}.
Scope S1 (per-document multiset), per-revision sync. Mean per edit with document-clustered bootstrap 95% CI (1,000 resamples). `size` = achieved mean chunk tokens of the new versions.


### E1 at T=128 (size-matched; primary)

| stream | chunker | edits | docs | toks_frac | ci_lo | ci_hi | INS | INS_p95 | DEL | size |
|---|---|---|---|---|---|---|---|---|---|---|
| cpy | FIX | 25,817 | 300 | 0.524 | 0.515 | 0.534 | 56.541 | 198.000 | 56.045 | 126.451 |
| cpy | FIXO | 25,817 | 300 | 0.655 | 0.644 | 0.667 | 70.576 | 247.200 | 69.952 | 126.977 |
| cpy | REC | 25,817 | 300 | 0.089 | 0.080 | 0.099 | 4.485 | 15.000 | 4.006 | 131.390 |
| cpy | STRUCT-M | 25,817 | 300 | 0.078 | 0.070 | 0.087 | 3.901 | 13.000 | 3.426 | 128.481 |
| cpy | SentCDC | 25,817 | 300 | 0.080 | 0.072 | 0.090 | 3.540 | 11.000 | 3.071 | 133.523 |
| cpy | GCDC | 25,817 | 300 | 0.076 | 0.068 | 0.085 | 3.574 | 11.000 | 3.069 | 124.438 |
| cpy | BCDC | 25,817 | 300 | 0.084 | 0.076 | 0.093 | 3.657 | 11.000 | 3.203 | 136.521 |
| cpy | STL | 25,817 | 300 | 0.144 | 0.132 | 0.158 | 8.426 | 25.000 | 7.933 | 126.258 |
| cpy | STL+snap | 25,817 | 300 | 0.108 | 0.099 | 0.120 | 6.532 | 20.000 | 5.998 | 117.784 |
| cpy | KARA | 25,817 | 300 | 0.063 | 0.057 | 0.071 | 3.502 | 11.000 | 2.953 | 122.860 |
| cpy | DIFF[REC] | 25,817 | 300 | 0.069 | 0.062 | 0.077 | 3.532 | 11.000 | 3.021 | 126.876 |
| cpy | DIFF[STRUCT-M] | 25,817 | 300 | 0.068 | 0.061 | 0.076 | 3.432 | 11.000 | 2.933 | 125.741 |
| cpy | DOC | 25,817 | 300 | 0.982 | 0.980 | 0.984 | 105.804 | 286.000 | 105.324 | 131.390 |
| cpy | SENTWIN (T-indep.) | 25,817 | 300 | 0.026 | 0.024 | 0.029 | 6.939 | 26.000 | 4.188 | 24.506 |
| cpy | STRUCT-NM (T-indep.) | 25,817 | 300 | 0.037 | 0.034 | 0.042 | 4.493 | 16.000 | 2.995 | 48.022 |
| k8s | FIX | 14,924 | 284 | 0.573 | 0.564 | 0.582 | 23.255 | 79.000 | 22.815 | 124.681 |
| k8s | FIXO | 14,924 | 284 | 0.714 | 0.704 | 0.725 | 28.944 | 99.000 | 28.392 | 125.762 |
| k8s | REC | 14,924 | 284 | 0.165 | 0.153 | 0.177 | 4.640 | 14.000 | 4.211 | 128.153 |
| k8s | STRUCT-M | 14,924 | 284 | 0.144 | 0.134 | 0.155 | 3.547 | 12.000 | 3.184 | 131.098 |
| k8s | SentCDC | 14,924 | 284 | 0.167 | 0.156 | 0.179 | 3.490 | 12.000 | 3.163 | 135.630 |
| k8s | GCDC | 14,924 | 284 | 0.151 | 0.141 | 0.162 | 3.583 | 12.000 | 3.247 | 129.977 |
| k8s | BCDC | 14,924 | 284 | 0.167 | 0.156 | 0.179 | 4.127 | 14.000 | 3.724 | 127.706 |
| k8s | STL | 14,924 | 284 | 0.266 | 0.251 | 0.284 | 7.985 | 23.000 | 7.551 | 125.570 |
| k8s | STL+snap | 14,924 | 284 | 0.203 | 0.191 | 0.217 | 6.484 | 20.000 | 6.015 | 116.983 |
| k8s | KARA | 14,924 | 284 | 0.134 | 0.125 | 0.144 | 4.369 | 14.000 | 3.876 | 119.858 |
| k8s | DIFF[REC] | 14,924 | 284 | 0.143 | 0.133 | 0.154 | 4.093 | 13.000 | 3.643 | 124.768 |
| k8s | DIFF[STRUCT-M] | 14,924 | 284 | 0.144 | 0.134 | 0.155 | 3.481 | 12.000 | 3.118 | 131.796 |
| k8s | DOC | 14,924 | 284 | 0.985 | 0.983 | 0.987 | 39.905 | 111.000 | 39.476 | 128.153 |
| k8s | SENTWIN (T-indep.) | 14,924 | 284 | 0.064 | 0.059 | 0.070 | 8.500 | 37.000 | 6.500 | 21.423 |
| k8s | STRUCT-NM (T-indep.) | 14,924 | 284 | 0.098 | 0.091 | 0.107 | 4.804 | 18.000 | 4.174 | 60.426 |
| wiki | FIX | 82,949 | 1,000 | 0.576 | 0.568 | 0.583 | 7.503 | 26.000 | 7.351 | 117.059 |
| wiki | FIXO | 82,949 | 1,000 | 0.702 | 0.693 | 0.710 | 9.154 | 33.000 | 8.964 | 119.859 |
| wiki | REC | 82,949 | 1,000 | 0.305 | 0.294 | 0.316 | 1.992 | 5.000 | 1.845 | 130.837 |
| wiki | STRUCT-M | 82,949 | 1,000 | 0.274 | 0.263 | 0.285 | 1.606 | 4.000 | 1.458 | 122.506 |
| wiki | SentCDC | 82,949 | 1,000 | 0.279 | 0.268 | 0.290 | 1.698 | 4.000 | 1.555 | 119.190 |
| wiki | GCDC | 82,949 | 1,000 | 0.266 | 0.256 | 0.277 | 1.660 | 4.000 | 1.510 | 118.494 |
| wiki | BCDC | 82,949 | 1,000 | 0.297 | 0.287 | 0.308 | 1.997 | 5.000 | 1.853 | 123.653 |
| wiki | STL | 82,949 | 1,000 | 0.390 | 0.378 | 0.402 | 3.245 | 9.000 | 3.097 | 121.649 |
| wiki | STL+snap | 82,949 | 1,000 | 0.355 | 0.343 | 0.365 | 2.979 | 9.000 | 2.826 | 118.552 |
| wiki | KARA | 82,949 | 1,000 | 0.258 | 0.248 | 0.268 | 1.954 | 5.000 | 1.795 | 114.859 |
| wiki | DIFF[REC] | 82,949 | 1,000 | 0.291 | 0.280 | 0.303 | 1.865 | 5.000 | 1.718 | 127.702 |
| wiki | DIFF[STRUCT-M] | 82,949 | 1,000 | 0.308 | 0.296 | 0.320 | 1.819 | 5.000 | 1.692 | 133.215 |
| wiki | DOC | 82,949 | 1,000 | 0.998 | 0.998 | 0.999 | 13.497 | 40.000 | 13.350 | 130.837 |
| wiki | SENTWIN (T-indep.) | 82,949 | 1,000 | 0.093 | 0.089 | 0.099 | 3.914 | 13.000 | 2.830 | 19.543 |
| wiki | STRUCT-NM (T-indep.) | 82,949 | 1,000 | 0.198 | 0.190 | 0.206 | 1.796 | 5.000 | 1.601 | 90.766 |

### E1 at T=200 (calibrated, NOT size-matched for all chunkers)

| stream | chunker | edits | docs | toks_frac | ci_lo | ci_hi | INS | INS_p95 | DEL | size |
|---|---|---|---|---|---|---|---|---|---|---|
| cpy | FIX | 25,817 | 300 | 0.532 | 0.523 | 0.542 | 36.583 | 127.000 | 36.265 | 196.271 |
| cpy | FIXO | 25,817 | 300 | 0.663 | 0.652 | 0.676 | 45.499 | 158.000 | 45.101 | 197.530 |
| cpy | REC | 25,817 | 300 | 0.131 | 0.119 | 0.144 | 4.456 | 15.000 | 4.149 | 206.363 |
| cpy | STRUCT-M | 25,817 | 300 | 0.109 | 0.098 | 0.122 | 3.461 | 11.000 | 3.160 | 203.091 |
| cpy | SentCDC | 25,817 | 300 | 0.102 | 0.092 | 0.115 | 2.975 | 9.000 | 2.666 | 198.182 |
| cpy | GCDC | 25,817 | 300 | 0.100 | 0.090 | 0.112 | 3.150 | 10.000 | 2.842 | 198.435 |
| cpy | BCDC | 25,817 | 300 | 0.100 | 0.091 | 0.111 | 3.399 | 10.000 | 3.068 | 187.090 |
| cpy | STL | 25,817 | 300 | 0.165 | 0.151 | 0.181 | 6.217 | 18.000 | 5.901 | 198.015 |
| cpy | STL+snap | 25,817 | 300 | 0.146 | 0.134 | 0.161 | 5.696 | 18.000 | 5.364 | 190.796 |
| cpy | KARA | 25,817 | 300 | 0.083 | 0.075 | 0.093 | 3.009 | 9.000 | 2.643 | 186.817 |
| cpy | DIFF[REC] | 25,817 | 300 | 0.087 | 0.079 | 0.098 | 2.979 | 9.000 | 2.610 | 187.797 |
| cpy | DIFF[STRUCT-M] | 25,817 | 300 | 0.087 | 0.078 | 0.098 | 2.859 | 9.000 | 2.505 | 187.306 |
| cpy | DOC | 25,817 | 300 | 0.982 | 0.980 | 0.984 | 66.957 | 180.000 | 66.649 | 206.363 |
| cpy | SENTWIN (T-indep.) | 25,817 | 300 | 0.026 | 0.024 | 0.029 | 6.939 | 26.000 | 4.188 | 24.506 |
| cpy | STRUCT-NM (T-indep.) | 25,817 | 300 | 0.037 | 0.034 | 0.042 | 4.493 | 16.000 | 2.995 | 48.022 |
| k8s | FIX | 14,924 | 284 | 0.588 | 0.579 | 0.597 | 15.255 | 51.000 | 14.975 | 192.342 |
| k8s | FIXO | 14,924 | 284 | 0.728 | 0.718 | 0.740 | 18.813 | 63.000 | 18.462 | 194.704 |
| k8s | REC | 14,924 | 284 | 0.222 | 0.207 | 0.237 | 4.005 | 12.000 | 3.735 | 201.333 |
| k8s | STRUCT-M | 14,924 | 284 | 0.187 | 0.174 | 0.201 | 2.874 | 9.000 | 2.640 | 205.284 |
| k8s | SentCDC | 14,924 | 284 | 0.185 | 0.173 | 0.199 | 3.083 | 10.000 | 2.825 | 176.176 |
| k8s | GCDC | 14,924 | 284 | 0.191 | 0.178 | 0.205 | 2.909 | 10.000 | 2.686 | 205.278 |
| k8s | BCDC | 14,924 | 284 | 0.194 | 0.181 | 0.209 | 3.664 | 12.000 | 3.367 | 172.832 |
| k8s | STL | 14,924 | 284 | 0.294 | 0.277 | 0.313 | 5.620 | 16.000 | 5.347 | 195.427 |
| k8s | STL+snap | 14,924 | 284 | 0.257 | 0.242 | 0.273 | 5.074 | 15.000 | 4.785 | 188.142 |
| k8s | KARA | 14,924 | 284 | 0.170 | 0.159 | 0.183 | 3.523 | 10.000 | 3.197 | 185.477 |
| k8s | DIFF[REC] | 14,924 | 284 | 0.177 | 0.165 | 0.190 | 3.252 | 10.000 | 2.945 | 187.940 |
| k8s | DIFF[STRUCT-M] | 14,924 | 284 | 0.183 | 0.171 | 0.197 | 2.798 | 9.000 | 2.554 | 202.213 |
| k8s | DOC | 14,924 | 284 | 0.985 | 0.983 | 0.987 | 25.264 | 70.000 | 24.995 | 201.333 |
| k8s | SENTWIN (T-indep.) | 14,924 | 284 | 0.064 | 0.059 | 0.070 | 8.500 | 37.000 | 6.500 | 21.423 |
| k8s | STRUCT-NM (T-indep.) | 14,924 | 284 | 0.098 | 0.091 | 0.107 | 4.804 | 18.000 | 4.174 | 60.426 |
| wiki | FIX | 82,949 | 1,000 | 0.618 | 0.609 | 0.626 | 5.139 | 17.000 | 5.043 | 176.281 |
| wiki | FIXO | 82,949 | 1,000 | 0.738 | 0.729 | 0.747 | 6.163 | 21.000 | 6.044 | 181.379 |
| wiki | REC | 82,949 | 1,000 | 0.394 | 0.380 | 0.407 | 1.744 | 5.000 | 1.652 | 202.305 |
| wiki | STRUCT-M | 82,949 | 1,000 | 0.311 | 0.299 | 0.325 | 1.524 | 4.000 | 1.404 | 148.619 |
| wiki | SentCDC | 82,949 | 1,000 | 0.280 | 0.270 | 0.291 | 1.688 | 4.000 | 1.547 | 120.420 |
| wiki | GCDC | 82,949 | 1,000 | 0.306 | 0.294 | 0.320 | 1.480 | 4.000 | 1.361 | 149.802 |
| wiki | BCDC | 82,949 | 1,000 | 0.374 | 0.361 | 0.387 | 1.688 | 4.000 | 1.596 | 188.317 |
| wiki | STL | 82,949 | 1,000 | 0.439 | 0.425 | 0.452 | 2.345 | 6.000 | 2.253 | 187.599 |
| wiki | STL+snap | 82,949 | 1,000 | 0.424 | 0.411 | 0.437 | 2.269 | 6.000 | 2.176 | 185.884 |
| wiki | KARA | 82,949 | 1,000 | 0.334 | 0.321 | 0.346 | 1.664 | 4.000 | 1.565 | 178.056 |
| wiki | DIFF[REC] | 82,949 | 1,000 | 0.361 | 0.347 | 0.375 | 1.589 | 4.000 | 1.490 | 186.506 |
| wiki | DIFF[STRUCT-M] | 82,949 | 1,000 | 0.340 | 0.327 | 0.353 | 1.686 | 4.000 | 1.583 | 159.027 |
| wiki | DOC | 82,949 | 1,000 | 0.998 | 0.998 | 0.999 | 8.672 | 27.000 | 8.580 | 202.305 |
| wiki | SENTWIN (T-indep.) | 82,949 | 1,000 | 0.093 | 0.089 | 0.099 | 3.914 | 13.000 | 2.830 | 19.543 |
| wiki | STRUCT-NM (T-indep.) | 82,949 | 1,000 | 0.198 | 0.190 | 0.206 | 1.796 | 5.000 | 1.601 | 90.766 |

### E1 scopes S1/S2/S3 (T=200, per revision; S2 = global dedup per stream, S3 = cross-stream)

| stream | chunker | S1_INS | S2_INS | S3_INS | S1_TOKS | S2_TOKS |
|---|---|---|---|---|---|---|
| cpy | FIX | 36.583 | 36.544 | 32.322 | 7227.355 | 7220.268 |
| cpy | FIXO | 45.499 | 45.454 | 39.296 | 9029.302 | 9020.575 |
| cpy | REC | 4.456 | 4.444 | 3.782 | 943.700 | 941.483 |
| cpy | STRUCT-M | 3.461 | 3.448 | 3.196 | 755.176 | 752.847 |
| cpy | STRUCT-NM | 4.493 | 3.842 | 3.767 | 274.783 | 264.720 |
| cpy | SENTWIN | 6.939 | 5.631 | 5.518 | 186.386 | 164.119 |
| cpy | SentCDC | 2.975 | 2.954 | 2.858 | 702.204 | 699.931 |
| cpy | GCDC | 3.150 | 3.138 | 3.002 | 688.870 | 686.859 |
| cpy | BCDC | 3.399 | 3.389 | 3.255 | 699.255 | 697.556 |
| cpy | STL | 6.217 | 6.204 | 5.400 | 1240.760 | 1238.163 |
| cpy | STL+snap | 5.696 | 5.682 | 4.594 | 1093.451 | 1090.760 |
| cpy | KARA | 3.009 | 3.001 | 2.968 | 558.676 | 557.185 |
| cpy | DIFF[REC] | 2.979 | 2.969 | 2.930 | 583.331 | 581.438 |
| cpy | DIFF[STRUCT-M] | 2.859 | 2.846 | 2.781 | 578.848 | 576.973 |
| cpy | DOC | 66.957 | 66.689 | 66.224 | 13800.332 | 13749.329 |
| k8s | FIX | 15.255 | 15.204 | 14.372 | 2967.017 | 2959.349 |
| k8s | FIXO | 18.813 | 18.762 | 17.611 | 3694.461 | 3685.180 |
| k8s | REC | 4.005 | 3.879 | 3.556 | 845.792 | 821.475 |
| k8s | STRUCT-M | 2.874 | 2.795 | 2.653 | 732.155 | 710.040 |
| k8s | STRUCT-NM | 4.804 | 4.030 | 3.825 | 428.746 | 394.906 |
| k8s | SENTWIN | 8.500 | 5.958 | 5.562 | 316.405 | 271.940 |
| k8s | SentCDC | 3.083 | 2.952 | 2.792 | 714.991 | 691.405 |
| k8s | GCDC | 2.909 | 2.829 | 2.682 | 759.093 | 738.363 |
| k8s | BCDC | 3.664 | 3.564 | 3.364 | 719.482 | 699.723 |
| k8s | STL | 5.620 | 5.547 | 5.008 | 1124.688 | 1111.209 |
| k8s | STL+snap | 5.074 | 4.958 | 4.332 | 975.866 | 954.439 |
| k8s | KARA | 3.523 | 3.387 | 3.241 | 640.028 | 618.711 |
| k8s | DIFF[REC] | 3.252 | 3.138 | 3.042 | 656.922 | 634.517 |
| k8s | DIFF[STRUCT-M] | 2.798 | 2.721 | 2.605 | 692.860 | 671.102 |
| k8s | DOC | 25.264 | 25.111 | 24.937 | 5090.795 | 5072.254 |
| wiki | FIX | 5.139 | 5.134 | 4.643 | 940.972 | 940.690 |
| wiki | FIXO | 6.163 | 6.161 | 5.547 | 1161.468 | 1160.954 |
| wiki | REC | 1.744 | 1.725 | 1.595 | 364.944 | 364.468 |
| wiki | STRUCT-M | 1.524 | 1.493 | 1.409 | 313.043 | 312.730 |
| wiki | STRUCT-NM | 1.796 | 1.776 | 1.668 | 210.277 | 209.902 |
| wiki | SENTWIN | 3.914 | 3.776 | 3.431 | 93.668 | 92.691 |
| wiki | SentCDC | 1.688 | 1.647 | 1.523 | 280.024 | 279.655 |
| wiki | GCDC | 1.480 | 1.450 | 1.366 | 305.533 | 305.209 |
| wiki | BCDC | 1.688 | 1.687 | 1.596 | 357.841 | 357.681 |
| wiki | STL | 2.345 | 2.343 | 2.118 | 452.994 | 452.638 |
| wiki | STL+snap | 2.269 | 2.264 | 1.984 | 434.453 | 434.093 |
| wiki | KARA | 1.664 | 1.661 | 1.605 | 299.396 | 298.913 |
| wiki | DIFF[REC] | 1.589 | 1.574 | 1.511 | 320.164 | 319.723 |
| wiki | DIFF[STRUCT-M] | 1.686 | 1.639 | 1.494 | 326.414 | 326.043 |
| wiki | DOC | 8.672 | 8.656 | 8.518 | 1765.341 | 1764.071 |

### E1 sync interval (F6): total re-embedded tokens relative to per-revision sync (T=200, S1)

CPY 'release' = sync at release dates (main's file state at v3.* tag timestamps).

| stream | chunker | revision | daily | weekly | monthly | sync at release dates |
|---|---|---|---|---|---|---|
| cpy | FIX | 1.000 | 0.865 | 0.713 | 0.534 | 0.402 |
| cpy | FIXO | 1.000 | 0.865 | 0.714 | 0.534 | 0.401 |
| cpy | REC | 1.000 | 0.949 | 0.875 | 0.782 | 0.679 |
| cpy | STRUCT-M | 1.000 | 0.960 | 0.910 | 0.841 | 0.744 |
| cpy | STRUCT-NM | 1.000 | 0.976 | 0.943 | 0.910 | 0.856 |
| cpy | SENTWIN | 1.000 | 0.985 | 0.959 | 0.934 | 0.899 |
| cpy | SentCDC | 1.000 | 0.963 | 0.915 | 0.853 | 0.758 |
| cpy | GCDC | 1.000 | 0.964 | 0.917 | 0.855 | 0.763 |
| cpy | BCDC | 1.000 | 0.957 | 0.901 | 0.830 | 0.738 |
| cpy | STL | 1.000 | 0.940 | 0.865 | 0.770 | 0.658 |
| cpy | STL+snap | 1.000 | 0.945 | 0.871 | 0.775 | 0.669 |
| cpy | KARA | 1.000 | 0.968 | 0.923 | 0.869 | 0.787 |
| cpy | DIFF[REC] | 1.000 | 0.965 | 0.919 | 0.863 | 0.778 |
| cpy | DIFF[STRUCT-M] | 1.000 | 0.966 | 0.924 | 0.869 | 0.787 |
| cpy | DOC | 1.000 | 0.841 | 0.665 | 0.458 | 0.319 |
| k8s | FIX | 1.000 | 0.961 | 0.885 | 0.711 | n/a |
| k8s | FIXO | 1.000 | 0.961 | 0.884 | 0.710 | n/a |
| k8s | REC | 1.000 | 0.988 | 0.961 | 0.882 | n/a |
| k8s | STRUCT-M | 1.000 | 0.990 | 0.965 | 0.887 | n/a |
| k8s | STRUCT-NM | 1.000 | 0.992 | 0.972 | 0.893 | n/a |
| k8s | SENTWIN | 1.000 | 0.996 | 0.977 | 0.895 | n/a |
| k8s | SentCDC | 1.000 | 0.990 | 0.965 | 0.884 | n/a |
| k8s | GCDC | 1.000 | 0.990 | 0.965 | 0.885 | n/a |
| k8s | BCDC | 1.000 | 0.989 | 0.963 | 0.881 | n/a |
| k8s | STL | 1.000 | 0.984 | 0.949 | 0.855 | n/a |
| k8s | STL+snap | 1.000 | 0.985 | 0.954 | 0.860 | n/a |
| k8s | KARA | 1.000 | 0.990 | 0.968 | 0.892 | n/a |
| k8s | DIFF[REC] | 1.000 | 0.990 | 0.968 | 0.889 | n/a |
| k8s | DIFF[STRUCT-M] | 1.000 | 0.989 | 0.965 | 0.885 | n/a |
| k8s | DOC | 1.000 | 0.947 | 0.838 | 0.613 | n/a |
| wiki | FIX | 1.000 | 0.577 | 0.497 | 0.412 | n/a |
| wiki | FIXO | 1.000 | 0.576 | 0.496 | 0.411 | n/a |
| wiki | REC | 1.000 | 0.674 | 0.602 | 0.527 | n/a |
| wiki | STRUCT-M | 1.000 | 0.696 | 0.628 | 0.556 | n/a |
| wiki | STRUCT-NM | 1.000 | 0.741 | 0.678 | 0.611 | n/a |
| wiki | SENTWIN | 1.000 | 0.779 | 0.731 | 0.679 | n/a |
| wiki | SentCDC | 1.000 | 0.707 | 0.640 | 0.570 | n/a |
| wiki | GCDC | 1.000 | 0.698 | 0.631 | 0.559 | n/a |
| wiki | BCDC | 1.000 | 0.676 | 0.606 | 0.532 | n/a |
| wiki | STL | 1.000 | 0.655 | 0.582 | 0.506 | n/a |
| wiki | STL+snap | 1.000 | 0.660 | 0.587 | 0.511 | n/a |
| wiki | KARA | 1.000 | 0.706 | 0.638 | 0.568 | n/a |
| wiki | DIFF[REC] | 1.000 | 0.699 | 0.631 | 0.560 | n/a |
| wiki | DIFF[STRUCT-M] | 1.000 | 0.695 | 0.627 | 0.555 | n/a |
| wiki | DOC | 1.000 | 0.507 | 0.418 | 0.330 | n/a |

Steadlith long-word split fallbacks (max per document): cpy/STL=118, cpy/STL+snap=118, k8s/STL=588, k8s/STL+snap=588, wiki/STL=270, wiki/STL+snap=270

Files: `results/e1_edits.parquet`, `results/e1_summary.csv`, `results/e1_matched_size.csv`, `results/e1_edits.meta.json`, `results/calibration.json`, `results/runbooks/`

## E1-rep: replication of Bian et al. (T-rep)

Pairs: 120 adjacent WIKI revision pairs; metric = % of new-version embedding-input tokens recomputed under byte-identical reuse (Bian's workload fraction).

| chunker | recompute_pct | median_len | size_matched_300 |
|---|---|---|---|
| Bian-Fixed304 | 55.7 | 304.0 | True |
| Bian-Recursive400 | 31.2 | 343.0 | True |
| Bian-Recursive400-nomerge | 9.3 | 30.5 | False |
| LangChain-REC400 | 31.2 | 343.0 | True |
| Bian-STL176/352 | 32.8 | 281.0 | True |
| REC | 21.9 | 229.8 | False |
| SentCDC | 16.6 | 106.0 | False |
| STRUCT-M | 18.7 | 127.2 | False |
| REC@cal320 | 32.2 | 374.5 | False |
| STRUCT-M@cal320 | 18.7 | 127.2 | False |
| SentCDC@cal320 | 16.6 | 106.0 | False |
| Bian-Fixed@cal200 | 52.6 | 200.0 | False |
| Bian-STL@cal200 | 27.6 | 209.8 | False |

Bian et al. English Table I (their data, for reference, not reproduced here): {"fixed_recompute_pct": 57.4, "recursive_recompute_pct": 37.9, "cdc_recompute_pct": 42.8}

Files: `results/e1_rep.json`, `results/e1_rep.csv`, `results/bian_params.json`

## E1-fid: LangChain index() fidelity

| setting | edits | lc_added | S1_distinct | match_S1_distinct | match_S2 |
|---|---|---|---|---|---|
| wiki/source_only | 3,686 | 6,310 | 6,310 | 1.000 | 0.995 |
| wiki/source_only_batch10000 | 3,686 | 6,310 | 6,310 | 1.000 | 0.995 |
| wiki/start_index | 3,686 | 20,384 | 6,310 | 0.447 | 0.445 |
| k8s/source_only | 2,316 | 10,814 | 8,171 | 0.977 | 0.962 |
| k8s/source_only_batch10000 | 2,316 | 8,171 | 8,171 | 1.000 | 0.985 |
| k8s/start_index | 2,316 | 30,578 | 8,171 | 0.260 | 0.254 |

Files: `results/e1_fid.json`

## E2: Theorem 1, Monte Carlo and out-of-domain predictor

Theorem 1 (corrected bound) per SentCDC edit; denominators = covered_edits (guard-split edits excluded).

| stream | r | lam | edits | covered_edits | covered_share | guard_edits | viol_ins | viol_del | mean_ins | mean_bound | mc_edits | mc_pred | mc_meas |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wiki | 20 | 0.500 | 82,949 | 73,649 | 0.888 | 9,300 | 0 | 0 | 1.688 | 7.631 | 300 | 1.111 | 1.160 |
| wiki | 3 | 0.500 | 82,949 | 81,266 | 0.980 | 1,683 | 0 | 0 | 2.195 | 6.132 | 300 | 1.391 | 1.403 |
| k8s | 20 | 0.500 | 14,924 | 10,040 | 0.673 | 4,884 | 0 | 0 | 3.083 | 9.109 | 300 | 1.108 | 1.223 |
| k8s | 3 | 0.500 | 14,924 | 13,845 | 0.928 | 1,079 | 0 | 0 | 4.114 | 9.383 | 300 | 1.571 | 1.613 |
| cpy | 7 | 0.500 | 25,817 | 16,829 | 0.652 | 8,988 | 0 | 0 | 2.975 | 7.819 | 300 | 1.249 | 1.293 |
| cpy | 3 | 0.500 | 25,817 | 23,809 | 0.922 | 2,008 | 0 | 0 | 3.540 | 8.533 | 300 | 1.485 | 1.483 |
- k8s: MdAPE (all) 0.067, MdAPE non-trivial (SentCDC, REC, STRUCT-M) 0.106, Kendall tau 0.733 (95% CI 0.733, 0.867), items 6
- cpy: MdAPE (all) 0.103, MdAPE non-trivial (SentCDC, REC, STRUCT-M) 0.207, Kendall tau 1.000 (95% CI 1.000, 1.000), items 6

Out-of-domain INS predictor (calibrated on WIKI), per-chunker APE with document-clustered 95% CI:

| stream | chunker | APE | ci_lo | ci_hi |
|---|---|---|---|---|
| cpy | FIX | 0.068 | 0.038 | 0.095 |
| cpy | REC | 0.207 | 0.102 | 0.325 |
| cpy | STRUCT-M | 0.525 | 0.410 | 0.661 |
| cpy | SENTWIN | 0.026 | 0.005 | 0.044 |
| cpy | SentCDC | 0.079 | 0.025 | 0.140 |
| cpy | DOC | 0.127 | 0.116 | 0.138 |
| k8s | FIX | 0.045 | 0.014 | 0.072 |
| k8s | REC | 0.015 | 0.003 | 0.353 |
| k8s | STRUCT-M | 0.473 | 0.127 | 0.823 |
| k8s | SENTWIN | 0.030 | 0.001 | 0.124 |
| k8s | SentCDC | 0.106 | 0.003 | 0.299 |
| k8s | DOC | 0.090 | 0.063 | 0.119 |

Note: Per-chunker APE is the primary result; FIX/DOC/SENTWIN are closed forms (trivial), REC/STRUCT-M (greedy coupling) and SentCDC (Monte Carlo) are the non-trivial predictors and are summarised separately.

Files: `results/e2_thm1_wiki.json`, `results/e2_thm1_wiki_r3.json`, `results/e2_thm1_k8s.json`, `results/e2_thm1_k8s_r3.json`, `results/e2_thm1_cpy.json`, `results/e2_thm1_cpy_r3.json`, `results/e2_predict.json`, `results/e2_predict_edits.parquet`

## E3: drift of history-dependent chunkers

Drift vs fresh chunking at the largest t of each cohort (m = maintained, f = fresh); full curves in the CSVs.

| stream | cohort | chunker | t_max | docs | ks | divergence | out_of_range_m | out_of_range_f | mean_m | mean_f | renorm_toks_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cpy | >=100 | DIFF[REC] | 100 | 73 | 0.155 | 0.537 | 0.012 | 0.021 | 186.577 | 205.871 | 0.488 |
| cpy | >=100 | DIFF[STRUCT-M] | 100 | 73 | 0.117 | 0.397 | 0.026 | 0.052 | 188.455 | 204.276 | 0.365 |
| cpy | >=100 | KARA | 100 | 73 | 0.292 | 0.860 | 0.001 | 0.001 | 186.577 | 202.897 | 0.852 |
| cpy | >=200 | DIFF[REC] | 200 | 33 | 0.205 | 0.526 | 0.012 | 0.017 | 182.129 | 207.853 | 0.527 |
| cpy | >=200 | DIFF[STRUCT-M] | 200 | 33 | 0.149 | 0.422 | 0.020 | 0.051 | 177.219 | 198.291 | 0.421 |
| cpy | >=200 | KARA | 200 | 33 | 0.309 | 0.854 | 0.001 | 0.002 | 184.127 | 202.101 | 0.859 |
| cpy | all | DIFF[REC] | 299 | 11 | 0.224 | 0.526 | 0.010 | 0.016 | 178.782 | 206.099 | 0.523 |
| cpy | all | DIFF[STRUCT-M] | 299 | 11 | 0.176 | 0.458 | 0.014 | 0.055 | 171.062 | 195.906 | 0.451 |
| cpy | all | KARA | 299 | 11 | 0.287 | 0.819 | 0.002 | 0.001 | 181.454 | 198.804 | 0.828 |
| k8s | >=100 | DIFF[REC] | 100 | 29 | 0.135 | 0.404 | 0.019 | 0.032 | 186.252 | 201.747 | 0.420 |
| k8s | >=100 | DIFF[STRUCT-M] | 100 | 29 | 0.029 | 0.182 | 0.041 | 0.070 | 199.859 | 203.626 | 0.166 |
| k8s | >=100 | KARA | 100 | 29 | 0.269 | 0.874 | 0.002 | 0.007 | 188.746 | 201.382 | 0.880 |
| k8s | >=200 | DIFF[REC] | 200 | 5 | 0.194 | 0.524 | 0.011 | 0.028 | 183.377 | 203.566 | 0.535 |
| k8s | >=200 | DIFF[STRUCT-M] | 200 | 5 | 0.055 | 0.189 | 0.053 | 0.088 | 206.087 | 209.987 | 0.193 |
| k8s | >=200 | KARA | 200 | 5 | 0.319 | 0.888 | 0.000 | 0.009 | 184.906 | 200.500 | 0.906 |
| k8s | all | DIFF[REC] | 250 | 1 | 0.263 | 0.589 | 0.000 | 0.000 | 182.600 | 211.946 | 0.594 |
| k8s | all | DIFF[STRUCT-M] | 250 | 1 | 0.045 | 0.075 | 0.000 | 0.019 | 219.796 | 223.943 | 0.069 |
| k8s | all | KARA | 250 | 1 | 0.352 | 0.847 | 0.016 | 0.017 | 188.397 | 201.169 | 0.848 |
| wiki | >=100 | DIFF[REC] | 100 | 231 | 0.136 | 0.406 | 0.045 | 0.057 | 191.079 | 210.084 | 0.376 |
| wiki | >=100 | DIFF[STRUCT-M] | 100 | 231 | 0.117 | 0.335 | 0.196 | 0.312 | 175.133 | 158.880 | 0.294 |
| wiki | >=100 | KARA | 100 | 231 | 0.357 | 0.865 | 0.011 | 0.029 | 185.929 | 200.279 | 0.850 |
| wiki | >=200 | DIFF[REC] | 200 | 61 | 0.151 | 0.481 | 0.039 | 0.061 | 194.641 | 212.728 | 0.474 |
| wiki | >=200 | DIFF[STRUCT-M] | 200 | 61 | 0.108 | 0.324 | 0.232 | 0.339 | 176.430 | 159.757 | 0.307 |
| wiki | >=200 | KARA | 200 | 61 | 0.382 | 0.873 | 0.009 | 0.015 | 189.693 | 204.938 | 0.876 |
| wiki | all | DIFF[REC] | 299 | 11 | 0.104 | 0.344 | 0.103 | 0.132 | 179.833 | 193.084 | 0.253 |
| wiki | all | DIFF[STRUCT-M] | 299 | 11 | 0.094 | 0.263 | 0.213 | 0.290 | 186.046 | 169.065 | 0.124 |
| wiki | all | KARA | 299 | 11 | 0.287 | 0.782 | 0.051 | 0.069 | 185.293 | 194.116 | 0.749 |

Files: `results/e3_drift_cpy.csv`, `results/e3_drift_k8s.csv`, `results/e3_drift_wiki.csv`

Sentence-probe recall@1024, maintained vs fresh layout (WIKI-100, docs whose history reaches t):

| chunker | t | maintained | fresh | docs_at_t | gap |
|---|---|---|---|---|---|
| DIFF[REC] | 0 | 0.950 | 0.950 | 100.000 | 0.000 |
| DIFF[REC] | 50 | 0.907 | 0.890 | 74.000 | 0.017 |
| DIFF[REC] | 100 | 0.933 | 0.943 | 22.000 | -0.010 |
| DIFF[REC] | 200 | 0.947 | 0.937 | 4.000 | 0.010 |
| DIFF[STRUCT-M] | 0 | 0.917 | 0.917 | 100.000 | 0.000 |
| DIFF[STRUCT-M] | 50 | 0.900 | 0.883 | 74.000 | 0.017 |
| DIFF[STRUCT-M] | 100 | 0.870 | 0.867 | 22.000 | 0.003 |
| DIFF[STRUCT-M] | 200 | 0.920 | 0.897 | 4.000 | 0.023 |
| KARA | 0 | 0.957 | 0.957 | 100.000 | 0.000 |
| KARA | 50 | 0.917 | 0.917 | 74.000 | 0.000 |
| KARA | 100 | 0.970 | 0.937 | 22.000 | 0.033 |
| KARA | 200 | 0.947 | 0.940 | 4.000 | 0.007 |

Files: `results/e3_quality.json`, `results/e3_quality.csv`

E3q gap (maintained - fresh) with document-clustered 95% CIs and MDE (alpha 0.05 two-sided, power 0.8; mde_boot = 2.80 x clustered-bootstrap SE; `docs` = documents holding the probes at t, so t=200 rests on 4 documents and its CI is not reliable):

| chunker | t | probes | docs | gap | gap_ci_lo | gap_ci_hi | discordant | mde_boot |
|---|---|---|---|---|---|---|---|---|
| DIFF[REC] | 0 | 300 | 46 | 0.000 | 0.000 | 0.000 | 0 | 0.000 |
| DIFF[REC] | 50 | 300 | 58 | 0.017 | -0.007 | 0.045 | 13 | 0.038 |
| DIFF[REC] | 100 | 300 | 20 | -0.010 | -0.022 | 0.006 | 11 | 0.020 |
| DIFF[REC] | 200 | 300 | 4 | 0.010 | 0.000 | 0.018 | 11 | 0.017 |
| DIFF[STRUCT-M] | 0 | 300 | 46 | 0.000 | 0.000 | 0.000 | 0 | 0.000 |
| DIFF[STRUCT-M] | 50 | 300 | 58 | 0.017 | 0.000 | 0.034 | 11 | 0.025 |
| DIFF[STRUCT-M] | 100 | 300 | 20 | 0.003 | -0.008 | 0.017 | 19 | 0.017 |
| DIFF[STRUCT-M] | 200 | 300 | 4 | 0.023 | 0.008 | 0.033 | 9 | 0.019 |
| KARA | 0 | 300 | 46 | 0.000 | 0.000 | 0.000 | 0 | 0.000 |
| KARA | 50 | 300 | 58 | 0.000 | -0.028 | 0.030 | 22 | 0.040 |
| KARA | 100 | 300 | 20 | 0.033 | 0.014 | 0.056 | 20 | 0.030 |
| KARA | 200 | 300 | 4 | 0.007 | -0.005 | 0.042 | 26 | 0.034 |
| DIFF[REC] | pooled_t>0 | 900 | 61 | 0.006 | -0.007 | 0.020 | 35 | 0.020 |
| DIFF[STRUCT-M] | pooled_t>0 | 900 | 61 | 0.014 | 0.004 | 0.021 | 39 | 0.012 |
| KARA | pooled_t>0 | 900 | 61 | 0.013 | -0.0007 | 0.033 | 68 | 0.024 |

Files: `results/e3q_ci_mde.csv`

## E4: retrieval quality at nominal T=128


### E4 native background (primary)

Calibration T=128; background chunks 30000 (native); questions 900; model BAAI/bge-small-en-v1.5 on mps.

| stream | chunker | n | recall@1024 | recall@2048 | recall_lex@1024 | ndcg10 | used_tok@1024 | size |
|---|---|---|---|---|---|---|---|---|
| cpy | FIX | 200 | 0.400 | 0.475 | 0.555 | 0.305 | 1015.235 | 127.300 |
| cpy | REC | 200 | 0.420 | 0.505 | 0.530 | 0.335 | 1009.030 | 131.100 |
| cpy | STRUCT-M | 200 | 0.425 | 0.510 | 0.520 | 0.328 | 1009.020 | 130.700 |
| cpy | STRUCT-NM | 200 | 0.505 | 0.575 | 0.590 | 0.292 | 1011.275 | 44.300 |
| cpy | SentCDC | 200 | 0.365 | 0.425 | 0.505 | 0.263 | 1015.595 | 133.200 |
| cpy | STL | 200 | 0.395 | 0.455 | 0.515 | 0.285 | 997.635 | 126.800 |
| cpy | KARA | 200 | 0.410 | 0.510 | 0.525 | 0.327 | 1001.415 | 122.400 |
| k8s | FIX | 200 | 0.400 | 0.455 | 0.495 | 0.321 | 1008.285 | 126.500 |
| k8s | REC | 200 | 0.460 | 0.560 | 0.525 | 0.341 | 1010.080 | 131.200 |
| k8s | STRUCT-M | 200 | 0.445 | 0.505 | 0.520 | 0.345 | 1011.750 | 148.200 |
| k8s | STRUCT-NM | 200 | 0.515 | 0.600 | 0.575 | 0.320 | 981.735 | 65.300 |
| k8s | SentCDC | 200 | 0.395 | 0.435 | 0.455 | 0.295 | 1009.105 | 158.100 |
| k8s | STL | 200 | 0.410 | 0.470 | 0.485 | 0.317 | 1001.770 | 127.900 |
| k8s | KARA | 200 | 0.505 | 0.570 | 0.630 | 0.358 | 1005.680 | 118.900 |
| wiki | FIX | 500 | 0.592 | 0.642 | 0.632 | 0.460 | 1011.540 | 124.300 |
| wiki | REC | 500 | 0.616 | 0.690 | 0.670 | 0.508 | 1010.234 | 129.800 |
| wiki | STRUCT-M | 500 | 0.606 | 0.678 | 0.646 | 0.497 | 1016.862 | 125.800 |
| wiki | STRUCT-NM | 500 | 0.710 | 0.760 | 0.738 | 0.545 | 1016.278 | 94.600 |
| wiki | SentCDC | 500 | 0.618 | 0.670 | 0.674 | 0.512 | 1018.430 | 127.300 |
| wiki | STL | 500 | 0.560 | 0.616 | 0.622 | 0.437 | 1005.688 | 127.800 |
| wiki | KARA | 500 | 0.682 | 0.728 | 0.736 | 0.548 | 1005.162 | 120.500 |

### E4 no background (control)

Calibration T=128; background chunks 0 (none); questions 900; model BAAI/bge-small-en-v1.5 on mps.

| stream | chunker | n | recall@1024 | recall@2048 | recall_lex@1024 | ndcg10 | used_tok@1024 | size |
|---|---|---|---|---|---|---|---|---|
| cpy | FIX | 200 | 0.405 | 0.475 | 0.555 | 0.307 | 1014.965 | 127.300 |
| cpy | REC | 200 | 0.430 | 0.515 | 0.530 | 0.338 | 1008.430 | 131.100 |
| cpy | STRUCT-M | 200 | 0.425 | 0.510 | 0.515 | 0.332 | 1008.140 | 130.700 |
| cpy | STRUCT-NM | 200 | 0.510 | 0.580 | 0.595 | 0.293 | 1009.890 | 44.300 |
| cpy | SentCDC | 200 | 0.370 | 0.435 | 0.500 | 0.268 | 1015.300 | 133.200 |
| cpy | STL | 200 | 0.395 | 0.460 | 0.510 | 0.288 | 997.315 | 126.800 |
| cpy | KARA | 200 | 0.410 | 0.515 | 0.515 | 0.332 | 1000.495 | 122.400 |
| k8s | FIX | 200 | 0.400 | 0.465 | 0.495 | 0.324 | 1010.120 | 126.500 |
| k8s | REC | 200 | 0.470 | 0.560 | 0.535 | 0.346 | 1009.850 | 131.200 |
| k8s | STRUCT-M | 200 | 0.445 | 0.510 | 0.520 | 0.346 | 1011.720 | 148.200 |
| k8s | STRUCT-NM | 200 | 0.515 | 0.600 | 0.575 | 0.322 | 979.785 | 65.300 |
| k8s | SentCDC | 200 | 0.395 | 0.450 | 0.455 | 0.297 | 1017.170 | 158.100 |
| k8s | STL | 200 | 0.415 | 0.475 | 0.485 | 0.317 | 1000.005 | 127.900 |
| k8s | KARA | 200 | 0.505 | 0.580 | 0.630 | 0.358 | 1004.035 | 118.900 |
| wiki | FIX | 500 | 0.660 | 0.708 | 0.698 | 0.527 | 1005.978 | 124.300 |
| wiki | REC | 500 | 0.710 | 0.756 | 0.744 | 0.575 | 1011.594 | 129.800 |
| wiki | STRUCT-M | 500 | 0.672 | 0.736 | 0.706 | 0.557 | 1016.116 | 125.800 |
| wiki | STRUCT-NM | 500 | 0.746 | 0.792 | 0.770 | 0.592 | 1012.676 | 94.600 |
| wiki | SentCDC | 500 | 0.676 | 0.732 | 0.724 | 0.568 | 1017.732 | 127.300 |
| wiki | STL | 500 | 0.650 | 0.708 | 0.700 | 0.517 | 1000.900 | 127.800 |
| wiki | KARA | 500 | 0.744 | 0.800 | 0.774 | 0.611 | 999.634 | 120.500 |

### E4 stale-20 index (REC)

Calibration T=128; background chunks 30000 (native); questions 900; model BAAI/bge-small-en-v1.5 on mps.

| stream | chunker | n | recall@1024 | recall@2048 | recall_lex@1024 | ndcg10 | used_tok@1024 | size |
|---|---|---|---|---|---|---|---|---|
| cpy | REC-stale20 | 200 | 0.305 | 0.350 | 0.415 | 0.241 | 1008.660 | 130.700 |
| k8s | REC-stale20 | 200 | 0.230 | 0.275 | 0.360 | 0.155 | 1010.545 | 128.700 |
| wiki | REC-stale20 | 500 | 0.350 | 0.384 | 0.554 | 0.280 | 1010.272 | 129.100 |

### E4 history-dependent layouts: DIFF[REC], DIFF[STRUCT-M] maintained over the full history; KARA-fresh = KARA on the final version without history (optional review item 8; REC/STRUCT-M/KARA rows are the primary run's)

Calibration T=128; background chunks 30000 (native); questions 900; model BAAI/bge-small-en-v1.5 on mps.

| stream | chunker | n | recall@1024 | recall@2048 | recall_lex@1024 | ndcg10 | used_tok@1024 | size |
|---|---|---|---|---|---|---|---|---|
| cpy | REC | 200 | 0.420 | 0.505 | 0.530 | 0.335 | 1009.030 | 131.100 |
| cpy | STRUCT-M | 200 | 0.425 | 0.510 | 0.520 | 0.328 | 1009.020 | 130.700 |
| cpy | KARA | 200 | 0.410 | 0.510 | 0.525 | 0.327 | 1001.415 | 122.400 |
| cpy | DIFF[REC] | 200 | 0.435 | 0.515 | 0.550 | 0.337 | 1008.835 | 126.800 |
| cpy | DIFF[STRUCT-M] | 200 | 0.440 | 0.505 | 0.555 | 0.334 | 1009.140 | 127.500 |
| cpy | KARA-fresh | 200 | 0.435 | 0.520 | 0.560 | 0.334 | 998.885 | 130.200 |
| k8s | REC | 200 | 0.460 | 0.560 | 0.525 | 0.341 | 1010.080 | 131.200 |
| k8s | STRUCT-M | 200 | 0.445 | 0.505 | 0.520 | 0.345 | 1011.750 | 148.200 |
| k8s | KARA | 200 | 0.505 | 0.570 | 0.630 | 0.358 | 1005.680 | 118.900 |
| k8s | DIFF[REC] | 200 | 0.470 | 0.570 | 0.520 | 0.339 | 1010.180 | 127.700 |
| k8s | DIFF[STRUCT-M] | 200 | 0.455 | 0.540 | 0.515 | 0.351 | 1009.545 | 148.300 |
| k8s | KARA-fresh | 200 | 0.435 | 0.520 | 0.575 | 0.327 | 1003.200 | 124.100 |
| wiki | REC | 500 | 0.616 | 0.690 | 0.670 | 0.508 | 1010.234 | 129.800 |
| wiki | STRUCT-M | 500 | 0.606 | 0.678 | 0.646 | 0.497 | 1016.862 | 125.800 |
| wiki | KARA | 500 | 0.682 | 0.728 | 0.736 | 0.548 | 1005.162 | 120.500 |
| wiki | DIFF[REC] | 500 | 0.646 | 0.702 | 0.694 | 0.523 | 1010.966 | 129.500 |
| wiki | DIFF[STRUCT-M] | 500 | 0.604 | 0.682 | 0.644 | 0.508 | 1016.126 | 140.700 |
| wiki | KARA-fresh | 500 | 0.662 | 0.728 | 0.718 | 0.541 | 1003.854 | 127.200 |

Paired statistics vs REC (recall@1024; clustered bootstrap 95% CIs; McNemar Holm; Newcombe TOST 2 pp):

| stream | config | n | mean | ci_lo | ci_hi | diff | diff_ci_lo | diff_ci_hi | p_holm | discordant | tost90_lo | tost90_hi | n_required | tost_equivalent | mde_boot | mde_mcnemar |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | FIX | 900 | 0.507 | 0.473 | 0.541 | -0.031 | -0.059 | -0.003 | 0.094 | 158.000 | -0.054 | -0.008 | 2714.000 | False | 0.040 | 0.039 |
| all | REC | 900 | 0.538 | 0.507 | 0.571 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| all | STRUCT-M | 900 | 0.530 | 0.498 | 0.562 | -0.008 | -0.032 | 0.017 | 0.586 | 121.000 | -0.028 | 0.012 | 2079.000 | False | 0.035 | 0.034 |
| all | STRUCT-NM | 900 | 0.621 | 0.589 | 0.654 | 0.083 | 0.058 | 0.110 | 7.46e-11 | 127.000 | 0.063 | 0.103 | 2182.000 | False | 0.036 | 0.035 |
| all | SentCDC | 900 | 0.512 | 0.479 | 0.545 | -0.026 | -0.053 | 0.001 | 0.150 | 153.000 | -0.048 | -0.003 | 2628.000 | False | 0.039 | 0.039 |
| all | STL | 900 | 0.490 | 0.458 | 0.522 | -0.048 | -0.074 | -0.021 | 0.007 | 173.000 | -0.072 | -0.024 | 2972.000 | False | 0.039 | 0.041 |
| all | KARA | 900 | 0.582 | 0.548 | 0.616 | 0.044 | 0.018 | 0.070 | 0.007 | 152.000 | 0.022 | 0.067 | 2611.000 | False | 0.038 | 0.038 |
| cpy | FIX | 200 | 0.400 | 0.328 | 0.472 | -0.020 | -0.077 | 0.035 | 1.000 | 30.000 | -0.065 | 0.026 | 2319.000 | False | 0.081 | 0.077 |
| cpy | REC | 200 | 0.420 | 0.349 | 0.493 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| cpy | STRUCT-M | 200 | 0.425 | 0.352 | 0.497 | 0.005 | -0.036 | 0.046 | 1.000 | 17.000 | -0.030 | 0.040 | 1314.000 | False | 0.058 | 0.058 |
| cpy | STRUCT-NM | 200 | 0.505 | 0.437 | 0.576 | 0.085 | 0.035 | 0.139 | 0.020 | 31.000 | 0.039 | 0.130 | 2396.000 | False | 0.075 | 0.078 |
| cpy | SentCDC | 200 | 0.365 | 0.298 | 0.436 | -0.055 | -0.115 | 0.010 | 0.448 | 35.000 | -0.103 | -0.006 | 2705.000 | False | 0.087 | 0.083 |
| cpy | STL | 200 | 0.395 | 0.327 | 0.462 | -0.025 | -0.074 | 0.025 | 1.000 | 31.000 | -0.071 | 0.021 | 2396.000 | False | 0.071 | 0.078 |
| cpy | KARA | 200 | 0.410 | 0.342 | 0.480 | -0.010 | -0.068 | 0.047 | 1.000 | 34.000 | -0.058 | 0.038 | 2628.000 | False | 0.084 | 0.082 |
| k8s | FIX | 200 | 0.400 | 0.327 | 0.474 | -0.060 | -0.128 | 0.010 | 0.415 | 46.000 | -0.115 | -0.004 | 3556.000 | False | 0.097 | 0.095 |
| k8s | REC | 200 | 0.460 | 0.391 | 0.528 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| k8s | STRUCT-M | 200 | 0.445 | 0.374 | 0.510 | -0.015 | -0.069 | 0.041 | 0.728 | 33.000 | -0.063 | 0.033 | 2551.000 | False | 0.078 | 0.080 |
| k8s | STRUCT-NM | 200 | 0.515 | 0.446 | 0.585 | 0.055 | -0.000119 | 0.116 | 0.320 | 29.000 | 0.010 | 0.099 | 2242.000 | False | 0.084 | 0.075 |
| k8s | SentCDC | 200 | 0.395 | 0.327 | 0.462 | -0.065 | -0.123 | -0.005 | 0.320 | 39.000 | -0.116 | -0.013 | 3015.000 | False | 0.086 | 0.087 |
| k8s | STL | 200 | 0.410 | 0.340 | 0.477 | -0.050 | -0.115 | 0.015 | 0.462 | 40.000 | -0.102 | 0.002 | 3092.000 | False | 0.091 | 0.089 |
| k8s | KARA | 200 | 0.505 | 0.434 | 0.575 | 0.045 | -0.020 | 0.109 | 0.465 | 45.000 | -0.010 | 0.100 | 3478.000 | False | 0.093 | 0.094 |
| wiki | FIX | 500 | 0.592 | 0.547 | 0.636 | -0.024 | -0.059 | 0.010 | 0.673 | 82.000 | -0.054 | 0.006 | 2535.000 | False | 0.050 | 0.051 |
| wiki | REC | 500 | 0.616 | 0.572 | 0.658 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| wiki | STRUCT-M | 500 | 0.606 | 0.564 | 0.650 | -0.010 | -0.045 | 0.023 | 1.000 | 71.000 | -0.038 | 0.018 | 2195.000 | False | 0.048 | 0.047 |
| wiki | STRUCT-NM | 500 | 0.710 | 0.670 | 0.751 | 0.094 | 0.063 | 0.127 | 2.43e-08 | 67.000 | 0.068 | 0.120 | 2072.000 | False | 0.045 | 0.046 |
| wiki | SentCDC | 500 | 0.618 | 0.575 | 0.660 | 0.002 | -0.034 | 0.037 | 1.000 | 79.000 | -0.027 | 0.031 | 2443.000 | False | 0.050 | 0.050 |
| wiki | STL | 500 | 0.560 | 0.514 | 0.604 | -0.056 | -0.095 | -0.018 | 0.029 | 102.000 | -0.089 | -0.023 | 3154.000 | False | 0.055 | 0.057 |
| wiki | KARA | 500 | 0.682 | 0.640 | 0.724 | 0.066 | 0.032 | 0.096 | 0.000708 | 73.000 | 0.038 | 0.094 | 2257.000 | False | 0.046 | 0.048 |

History-dependent layouts, paired vs REC (recall@1024; same statistics):

| stream | config | n | mean | ci_lo | ci_hi | diff | diff_ci_lo | diff_ci_hi | p_holm | mde_boot | mde_mcnemar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | REC | 900 | 0.538 | 0.507 | 0.571 | n/a | n/a | n/a | n/a | n/a | n/a |
| all | STRUCT-M | 900 | 0.530 | 0.498 | 0.562 | -0.008 | -0.032 | 0.017 | 1.000 | 0.035 | 0.034 |
| all | KARA | 900 | 0.582 | 0.548 | 0.616 | 0.044 | 0.018 | 0.070 | 0.007 | 0.038 | 0.038 |
| all | DIFF[REC] | 900 | 0.560 | 0.528 | 0.593 | 0.022 | 0.007 | 0.037 | 0.042 | 0.022 | 0.023 |
| all | DIFF[STRUCT-M] | 900 | 0.534 | 0.501 | 0.566 | -0.003 | -0.028 | 0.022 | 1.000 | 0.035 | 0.035 |
| all | KARA-fresh | 900 | 0.561 | 0.526 | 0.592 | 0.023 | -0.003 | 0.049 | 0.289 | 0.037 | 0.037 |
| cpy | REC | 200 | 0.420 | 0.349 | 0.493 | n/a | n/a | n/a | n/a | n/a | n/a |
| cpy | STRUCT-M | 200 | 0.425 | 0.352 | 0.497 | 0.005 | -0.036 | 0.046 | 1.000 | 0.058 | 0.058 |
| cpy | KARA | 200 | 0.410 | 0.342 | 0.480 | -0.010 | -0.068 | 0.047 | 1.000 | 0.084 | 0.082 |
| cpy | DIFF[REC] | 200 | 0.435 | 0.365 | 0.505 | 0.015 | -0.020 | 0.051 | 1.000 | 0.050 | 0.051 |
| cpy | DIFF[STRUCT-M] | 200 | 0.440 | 0.372 | 0.512 | 0.020 | -0.020 | 0.062 | 1.000 | 0.059 | 0.059 |
| cpy | KARA-fresh | 200 | 0.435 | 0.362 | 0.505 | 0.015 | -0.042 | 0.073 | 1.000 | 0.081 | 0.075 |
| k8s | REC | 200 | 0.460 | 0.391 | 0.528 | n/a | n/a | n/a | n/a | n/a | n/a |
| k8s | STRUCT-M | 200 | 0.445 | 0.374 | 0.510 | -0.015 | -0.069 | 0.041 | 1.000 | 0.078 | 0.080 |
| k8s | KARA | 200 | 0.505 | 0.434 | 0.575 | 0.045 | -0.020 | 0.109 | 1.000 | 0.093 | 0.094 |
| k8s | DIFF[REC] | 200 | 0.470 | 0.403 | 0.538 | 0.010 | -0.020 | 0.041 | 1.000 | 0.044 | 0.049 |
| k8s | DIFF[STRUCT-M] | 200 | 0.455 | 0.383 | 0.522 | -0.005 | -0.064 | 0.051 | 1.000 | 0.083 | 0.085 |
| k8s | KARA-fresh | 200 | 0.435 | 0.362 | 0.510 | -0.025 | -0.086 | 0.031 | 1.000 | 0.083 | 0.083 |
| wiki | REC | 500 | 0.616 | 0.572 | 0.658 | n/a | n/a | n/a | n/a | n/a | n/a |
| wiki | STRUCT-M | 500 | 0.606 | 0.564 | 0.650 | -0.010 | -0.045 | 0.023 | 1.000 | 0.048 | 0.047 |
| wiki | KARA | 500 | 0.682 | 0.640 | 0.724 | 0.066 | 0.032 | 0.096 | 0.000708 | 0.046 | 0.048 |
| wiki | DIFF[REC] | 500 | 0.646 | 0.604 | 0.689 | 0.030 | 0.008 | 0.052 | 0.043 | 0.030 | 0.031 |
| wiki | DIFF[STRUCT-M] | 500 | 0.604 | 0.562 | 0.648 | -0.012 | -0.046 | 0.021 | 1.000 | 0.048 | 0.048 |
| wiki | KARA-fresh | 500 | 0.662 | 0.621 | 0.703 | 0.046 | 0.012 | 0.081 | 0.043 | 0.050 | 0.050 |

Freshness, stale-20 vs maintained REC (recall@1024; all questions; paired, clustered bootstrap 95% CI, McNemar):

| stream | config | n | mean | ci_lo | ci_hi | diff | diff_ci_lo | diff_ci_hi | mcnemar_p | discordant |
|---|---|---|---|---|---|---|---|---|---|---|
| cpy | REC | 200 | 0.420 | 0.349 | 0.493 | n/a | n/a | n/a | n/a | n/a |
| cpy | REC-stale20 | 200 | 0.305 | 0.235 | 0.370 | -0.115 | -0.175 | -0.061 | 0.000117 | 35.000 |
| k8s | REC | 200 | 0.460 | 0.391 | 0.528 | n/a | n/a | n/a | n/a | n/a |
| k8s | REC-stale20 | 200 | 0.230 | 0.172 | 0.291 | -0.230 | -0.302 | -0.163 | 7.67e-10 | 60.000 |
| wiki | REC | 500 | 0.616 | 0.572 | 0.658 | n/a | n/a | n/a | n/a | n/a |
| wiki | REC-stale20 | 500 | 0.350 | 0.308 | 0.394 | -0.266 | -0.308 | -0.224 | 2.7e-31 | 153.000 |
| all | REC | 900 | 0.538 | 0.507 | 0.571 | n/a | n/a | n/a | n/a | n/a |
| all | REC-stale20 | 900 | 0.313 | 0.282 | 0.345 | -0.224 | -0.256 | -0.195 | 7.85e-43 | 248.000 |

Files: `results/e4_stats_fresh.csv`, `results/e4_quality_fresh.csv`

Freshness, stale-20 vs maintained REC (recall@1024; edit-proximal questions only (evidence sentence unit absent from version n-20, i.e. from the stale index's version, by construction); paired, clustered bootstrap 95% CI, McNemar):

| stream | config | n | mean | ci_lo | ci_hi | diff | diff_ci_lo | diff_ci_hi | mcnemar_p | discordant |
|---|---|---|---|---|---|---|---|---|---|---|
| cpy | REC | 65 | 0.400 | 0.277 | 0.523 | n/a | n/a | n/a | n/a | n/a |
| cpy | REC-stale20 | 65 | 0.000 | 0.000 | 0.000 | -0.400 | -0.523 | -0.277 | 2.98e-08 | 26.000 |
| k8s | REC | 101 | 0.455 | 0.360 | 0.554 | n/a | n/a | n/a | n/a | n/a |
| k8s | REC-stale20 | 101 | 0.020 | 0.000 | 0.050 | -0.436 | -0.538 | -0.340 | 1.14e-13 | 44.000 |
| wiki | REC | 226 | 0.593 | 0.527 | 0.658 | n/a | n/a | n/a | n/a | n/a |
| wiki | REC-stale20 | 226 | 0.004 | 0.000 | 0.014 | -0.588 | -0.654 | -0.521 | 1.84e-40 | 133.000 |
| all | REC | 392 | 0.526 | 0.471 | 0.578 | n/a | n/a | n/a | n/a | n/a |
| all | REC-stale20 | 392 | 0.008 | 0.000 | 0.018 | -0.518 | -0.571 | -0.464 | 1.56e-61 | 203.000 |

Files: `results/e4_stats_fresh_prox.csv`, `results/e4_quality_fresh_prox.csv`

Files: `results/e4_quality.csv`, `results/e4_quality.meta.json`, `results/e4_quality_nobg.csv`, `results/e4_quality_stale.csv`, `results/e4_stats.csv`, `results/e4_quality_diff.csv`, `results/e4_stats_diff.csv`, `results/qa/`

## E5: contextual embeddings (LATE, STALE/WINDOW-D, CTX)

E5a: 300 WIKI edits, 3,283 unchanged chunks; cosine(old, new) of unchanged chunks by distance to the nearest changed chunk (LATE vs LATE-ctl):

| dist | n | cos_late_mean | cos_late_p05 | cos_ctl_min |
|---|---|---|---|---|
| 1.0000 | 506.0000 | 0.9962 | 0.9860 | 1.0000 |
| 2.0000 | 427.0000 | 0.9979 | 0.9941 | 1.0000 |
| 3.0000 | 339.0000 | 0.9990 | 0.9972 | 1.0000 |
| 4.0000 | 272.0000 | 0.9997 | 0.9988 | 1.0000 |
| 5.0000 | 235.0000 | 0.9998 | 0.9991 | 1.0000 |
| 6.0000 | 192.0000 | 0.9993 | 0.9993 | 1.0000 |
| 7.0000 | 156.0000 | 0.9999 | 0.9996 | 1.0000 |
| 8.0000 | 132.0000 | 0.9999 | 0.9998 | 1.0000 |
| 9.0000 | 117.0000 | 0.9991 | 0.9998 | 1.0000 |
| 10.0000 | 907.0000 | 1.0000 | 1.0000 | 1.0000 |

E5b: 266 docs, last 20 edits, 300 edit-proximal questions:

| policy | recall | forward_tokens | vectors |
|---|---|---|---|
| EXACT | 0.683 | 9,473,873 | 66,091 |
| STALE | 0.693 | 9,377,858 | 10,798 |
| WINDOW-0 | 0.663 | 3,810,879 | 10,798 |
| WINDOW-1 | 0.677 | 4,714,338 | 19,892 |
| WINDOW-2 | 0.677 | 5,462,693 | 27,207 |
| WINDOW-4 | 0.683 | 6,537,397 | 38,094 |

E5b paired vs EXACT (experiments/e5b_stats.py; document-clustered bootstrap 95% CI, McNemar Holm, Newcombe TOST 2 pp; forward_rel / vectors_rel = cost relative to EXACT):

| policy | n | docs | recall1024 | ci_lo | ci_hi | forward_rel | vectors_rel | diff | diff_ci_lo | diff_ci_hi | discordant | p_holm | tost90_lo | tost90_hi | n_required | tost_equivalent | mde_boot |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EXACT | 300 | 266 | 0.6833 | 0.6326 | 0.7367 | 1.0000 | 1.0000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| STALE | 300 | 266 | 0.6933 | 0.6429 | 0.7442 | 0.9899 | 0.1634 | 0.0100 | -0.0033 | 0.0240 | 5.0000 | 1.0000 | -0.0037 | 0.0238 | 773.0000 | False | 0.0203 |
| WINDOW-0 | 300 | 266 | 0.6633 | 0.6100 | 0.7162 | 0.4023 | 0.1634 | -0.0200 | -0.0490 | 0.0067 | 18.0000 | 1.0000 | -0.0440 | 0.0039 | 928.0000 | False | 0.0402 |
| WINDOW-1 | 300 | 266 | 0.6767 | 0.6262 | 0.7282 | 0.4976 | 0.3010 | -0.0067 | -0.0259 | 0.0131 | 8.0000 | 1.0000 | -0.0234 | 0.0100 | 773.0000 | False | 0.0264 |
| WINDOW-2 | 300 | 266 | 0.6767 | 0.6250 | 0.7300 | 0.5766 | 0.4117 | -0.0067 | -0.0203 | 0.0066 | 4.0000 | 1.0000 | -0.0193 | 0.0059 | 773.0000 | False | 0.0187 |
| WINDOW-4 | 300 | 266 | 0.6833 | 0.6316 | 0.7365 | 0.6900 | 0.5764 | 0.0000 | -0.0100 | 0.0099 | 2.0000 | 1.0000 | -0.0100 | 0.0100 | 773.0000 | False | 0.0131 |

E5c CTX per-call cost (one client thread, ABBA, measured under contention); completed: {"calls": 422, "docs": 21, "reps_started": 2, "full_reps": 1, "time_capped": true, "max_minutes": 45.0, "planned_docs": 21, "planned_reps": 3}

| variant | calls | sec_per_call | sec_per_call_median | prompt_tokens_mean | prompt_n_mean | s/call <=1k | s/call 1-2k | s/call 2-3k |
|---|---|---|---|---|---|---|---|---|
| no_prompt_cache | 211 | 12.973 | 12.398 | 1542.005 | 1542.005 | 9.658 | 14.135 | 18.435 |
| prompt_cache | 211 | 6.831 | 6.547 | 1542.005 | 224.777 | 6.873 | 6.827 | 6.747 |

Files: `results/e5a_late.json`, `results/e5a_late.parquet`, `results/e5b_policies.json`, `results/e5b_policies.csv`, `results/e5b_stats.csv`, `results/e5c_ctx_cost.json`, `results/e5c_ctx_calls.csv`

## E6: HNSW under churn (reduced) and rebuild time rho(N)

Final checkpoint, mean over HNSW seeds:

| chunker | policy | seeds | recall_exact | recall_fresh | recall_ef50 | recall_ef100 | recall_ef200 | tomb_share | tomb_share_churned | lat_ms | rebuilds | rebuild_s | edits | live |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DIFF[REC] | REBUILD(0.3) | 3 | 0.990 | 0.989 | 0.975 | 0.990 | 0.996 | 0.002 | 0.089 | 0.894 | 68.000 | 1088.424 | 7955.000 | 51156.000 |
| DIFF[REC] | REUSE | 3 | 0.991 | 0.989 | 0.974 | 0.991 | 0.996 | 3.91e-05 | 0.002 | 0.885 | 0.000 | 0.000 | 7955.000 | 51156.000 |
| DIFF[REC] | TOMB | 3 | 0.990 | 0.990 | 0.978 | 0.990 | 0.996 | 0.183 | 0.908 | 0.959 | 0.000 | 0.000 | 7955.000 | 51156.000 |
| REC | REBUILD(0.3) | 3 | 0.987 | 0.986 | 0.971 | 0.987 | 0.993 | 0.008 | 0.284 | 0.919 | 76.000 | 958.455 | 7955.000 | 51089.000 |
| REC | REUSE | 3 | 0.987 | 0.986 | 0.971 | 0.987 | 0.994 | 1.96e-05 | 0.000917 | 0.842 | 0.000 | 0.000 | 7955.000 | 51089.000 |
| REC | TOMB | 3 | 0.987 | 0.987 | 0.974 | 0.987 | 0.993 | 0.199 | 0.921 | 0.973 | 0.000 | 0.000 | 7955.000 | 51089.000 |
| SentCDC | REBUILD(0.3) | 3 | 0.989 | 0.990 | 0.972 | 0.989 | 0.995 | 0.006 | 0.165 | 0.984 | 48.000 | 595.435 | 7955.000 | 51684.000 |
| SentCDC | REUSE | 3 | 0.991 | 0.989 | 0.974 | 0.991 | 0.996 | 0.000 | 0.000 | 0.761 | 0.000 | 0.000 | 7955.000 | 51684.000 |
| SentCDC | TOMB | 3 | 0.989 | 0.989 | 0.970 | 0.989 | 0.995 | 0.189 | 0.877 | 0.965 | 0.000 | 0.000 | 7955.000 | 51684.000 |

Files: `results/e6_index.csv`, `results/e6_index.meta.json`

rho(N) (HNSW build seconds): real_vectors=571,994, rho_100000=20.197, rho_300000=64.125, rho_1000000=209.163
(N above the real-vector count uses perturbed copies of real vectors; labelled.)

Files: `results/e7_rho.json`

## E7: regimes, unit costs, timing, variance decomposition, judge-3, staleness

Unit costs (Theil-Sen over timed calls; seconds ~ alpha0 + alpha * tokens): alpha0=0.0751, alpha=2.218e-05, alpha_ci95=[2.112e-05, 2.318e-05], single_call_s=0.0669, beta=0.0003, gamma=2.98e-07

Measured maintenance time (WIKI-20 slices, rotated order, cold cache; measured under contention):

| chunker | regime | reps | edits | per_1k_edits_s | per_1k_sd | t_chunk | t_embed | t_index | calls | encoded |
|---|---|---|---|---|---|---|---|---|---|---|
| FIX | batched | 3 | 1556.000 | 17.828 | 1.432 | 5.339 | 18.664 | 2.803 | 86.000 | 3424.000 |
| FIX | online | 3 | 1556.000 | 97.378 | 4.029 | 5.646 | 125.252 | 19.654 | 1556.000 | 8674.000 |
| REC | batched | 3 | 1556.000 | 10.936 | 0.631 | 2.717 | 13.688 | 0.600 | 85.000 | 1461.000 |
| REC | online | 3 | 1556.000 | 74.778 | 6.903 | 3.263 | 110.928 | 2.142 | 1555.000 | 2690.000 |
| STRUCT-M | batched | 3 | 1556.000 | 14.394 | 1.760 | 2.844 | 18.828 | 0.571 | 87.000 | 1431.000 |
| STRUCT-M | online | 3 | 1556.000 | 74.491 | 3.777 | 3.279 | 110.796 | 1.663 | 1541.000 | 2330.000 |
| SentCDC | batched | 3 | 1556.000 | 13.936 | 1.756 | 1.577 | 18.871 | 0.766 | 87.000 | 1671.000 |
| SentCDC | online | 3 | 1556.000 | 71.118 | 5.574 | 1.977 | 106.080 | 2.057 | 1545.000 | 2647.000 |
| KARA | batched | 3 | 1556.000 | 12.989 | 1.109 | 7.281 | 12.326 | 0.591 | 87.000 | 1463.000 |
| KARA | online | 3 | 1556.000 | 75.323 | 8.033 | 7.646 | 107.710 | 1.827 | 1556.000 | 2476.000 |
| DOC | batched | 3 | 1556.000 | 22.761 | 3.042 | 2.657 | 28.310 | 4.435 | 88.000 | 4352.000 |
| DOC | online | 3 | 1556.000 | 142.543 | 10.233 | 3.342 | 169.329 | 49.100 | 1556.000 | 13753.000 |

E7c LLM extraction: chunks=200, wall_s=1288.039, calls=200, hits=0, sec_per_call_wall=6.440, mean_latency_s=50.676, prompt_tokens=56,077, completion_tokens=48,049, parallel=8

judge-3 valid: True; rho_N=64.125; c_q=0.0005 (95% CI [2.8245475937792796e-05, 0.0009075142043007108]); LLM s/call=6.440.

Variance decomposition of log derived cost (balanced designs; shares of SS):

| stream | design | chunker | scope | regime | interactions | cells | chunkers | interval |
|---|---|---|---|---|---|---|---|---|
| cpy | A_scope@revision/all_regimes | 0.070 | 8.66e-05 | 0.921 | 0.009 | 135 | 15 | n/a |
| cpy | A_scope@revision/without_llm | 0.630 | 0.00061 | 0.292 | 0.076 | 90 | 15 | n/a |
| cpy | B_interval@S1/all_regimes | 0.054 | n/a | 0.933 | 0.008 | 225 | 15 | 0.005 |
| cpy | B_interval@S1/without_llm | 0.591 | n/a | 0.251 | 0.072 | 150 | 15 | 0.086 |
| k8s | A_scope@revision/all_regimes | 0.030 | 0.000129 | 0.964 | 0.006 | 135 | 15 | n/a |
| k8s | A_scope@revision/without_llm | 0.369 | 0.001 | 0.555 | 0.075 | 90 | 15 | n/a |
| k8s | B_interval@S1/all_regimes | 0.025 | n/a | 0.968 | 0.006 | 180 | 15 | 0.001 |
| k8s | B_interval@S1/without_llm | 0.359 | n/a | 0.542 | 0.072 | 120 | 15 | 0.027 |
| wiki | A_scope@revision/all_regimes | 0.017 | 8.79e-05 | 0.977 | 0.006 | 135 | 15 | n/a |
| wiki | A_scope@revision/without_llm | 0.112 | 0.000556 | 0.840 | 0.048 | 90 | 15 | n/a |
| wiki | B_interval@S1/all_regimes | 0.011 | n/a | 0.975 | 0.005 | 180 | 15 | 0.009 |
| wiki | B_interval@S1/without_llm | 0.083 | n/a | 0.772 | 0.039 | 120 | 15 | 0.105 |

judge-3: cost-optimal index policy differs (chunker-aware vs index-only) in 109 of 195 defined cells (0 undefined).

Staleness simulation (derived, trace-driven; budget grid shared by all chunkers per stream):

| stream | chunker | budget_x_rec_load | budget_tokens_per_h | load_tokens_per_h | utilisation | staleness_mean_h | staleness_p95_h |
|---|---|---|---|---|---|---|---|
| cpy | FIX | 0.500 | 70.076 | 1073.356 | 15.317 | 1.04e+06 | 2.3e+06 |
| cpy | FIX | 1.100 | 154.167 | 1073.356 | 6.962 | 422828.380 | 955701.543 |
| cpy | FIX | 2.000 | 280.304 | 1073.356 | 3.829 | 190805.707 | 451839.763 |
| cpy | FIX | 10.000 | 1401.518 | 1073.356 | 0.766 | 4242.240 | 19931.781 |
| cpy | REC | 0.500 | 70.076 | 140.152 | 2.000 | 74808.901 | 161659.587 |
| cpy | REC | 1.100 | 154.167 | 140.152 | 0.909 | 3691.805 | 16508.711 |
| cpy | REC | 2.000 | 280.304 | 140.152 | 0.500 | 158.240 | 822.736 |
| cpy | REC | 10.000 | 1401.518 | 140.152 | 0.100 | 4.996 | 20.950 |
| cpy | STRUCT-M | 0.500 | 70.076 | 112.153 | 1.600 | 43524.197 | 95939.140 |
| cpy | STRUCT-M | 1.100 | 154.167 | 112.153 | 0.727 | 1366.044 | 5588.492 |
| cpy | STRUCT-M | 2.000 | 280.304 | 112.153 | 0.400 | 76.041 | 390.057 |
| cpy | STRUCT-M | 10.000 | 1401.518 | 112.153 | 0.080 | 4.263 | 18.725 |
| cpy | SentCDC | 0.500 | 70.076 | 104.286 | 1.488 | 36391.912 | 78525.500 |
| cpy | SentCDC | 1.100 | 154.167 | 104.286 | 0.676 | 891.282 | 3053.969 |
| cpy | SentCDC | 2.000 | 280.304 | 104.286 | 0.372 | 68.106 | 349.335 |
| cpy | SentCDC | 10.000 | 1401.518 | 104.286 | 0.074 | 4.213 | 18.198 |
| cpy | STL | 0.500 | 70.076 | 184.269 | 2.630 | 127375.835 | 264285.715 |
| cpy | STL | 1.100 | 154.167 | 184.269 | 1.195 | 9322.464 | 34460.486 |
| cpy | STL | 2.000 | 280.304 | 184.269 | 0.657 | 1003.690 | 4002.917 |
| cpy | STL | 10.000 | 1401.518 | 184.269 | 0.131 | 8.685 | 40.730 |
| cpy | KARA | 0.500 | 70.076 | 82.971 | 1.184 | 10491.606 | 28856.138 |
| cpy | KARA | 1.100 | 154.167 | 82.971 | 0.538 | 234.807 | 1052.973 |
| cpy | KARA | 2.000 | 280.304 | 82.971 | 0.296 | 41.399 | 219.335 |
| cpy | KARA | 10.000 | 1401.518 | 82.971 | 0.059 | 3.044 | 12.999 |
| cpy | DIFF[REC] | 0.500 | 70.076 | 86.632 | 1.236 | 14461.321 | 37339.495 |
| cpy | DIFF[REC] | 1.100 | 154.167 | 86.632 | 0.562 | 287.742 | 1161.952 |
| cpy | DIFF[REC] | 2.000 | 280.304 | 86.632 | 0.309 | 44.772 | 235.389 |
| cpy | DIFF[REC] | 10.000 | 1401.518 | 86.632 | 0.062 | 3.200 | 13.882 |
| cpy | DOC | 0.500 | 70.076 | 2049.529 | 29.247 | 2.09e+06 | 4.54e+06 |
| cpy | DOC | 1.100 | 154.167 | 2049.529 | 13.294 | 898145.875 | 1.98e+06 |
| cpy | DOC | 2.000 | 280.304 | 2049.529 | 7.312 | 452229.141 | 1.01e+06 |
| cpy | DOC | 10.000 | 1401.518 | 2049.529 | 1.462 | 17912.570 | 73460.538 |
| k8s | FIX | 0.500 | 67.033 | 470.299 | 7.016 | 234615.140 | 516806.128 |
| k8s | FIX | 1.100 | 147.472 | 470.299 | 3.189 | 86804.599 | 190590.290 |
| k8s | FIX | 2.000 | 268.131 | 470.299 | 1.754 | 33669.849 | 70554.216 |
| k8s | FIX | 10.000 | 1340.656 | 470.299 | 0.351 | 60.792 | 234.904 |
| k8s | REC | 0.500 | 67.033 | 134.066 | 2.000 | 40820.475 | 88042.383 |
| k8s | REC | 1.100 | 147.472 | 134.066 | 0.909 | 2569.388 | 5018.672 |
| k8s | REC | 2.000 | 268.131 | 134.066 | 0.500 | 150.261 | 583.528 |
| k8s | REC | 10.000 | 1340.656 | 134.066 | 0.100 | 10.596 | 59.262 |
| k8s | STRUCT-M | 0.500 | 67.033 | 116.053 | 1.731 | 30875.905 | 67240.618 |
| k8s | STRUCT-M | 1.100 | 147.472 | 116.053 | 0.787 | 875.448 | 2498.242 |
| k8s | STRUCT-M | 2.000 | 268.131 | 116.053 | 0.433 | 108.990 | 463.641 |
| k8s | STRUCT-M | 10.000 | 1340.656 | 116.053 | 0.087 | 8.869 | 51.590 |
| k8s | SentCDC | 0.500 | 67.033 | 113.332 | 1.691 | 29635.511 | 64043.457 |
| k8s | SentCDC | 1.100 | 147.472 | 113.332 | 0.769 | 733.524 | 2286.341 |
| k8s | SentCDC | 2.000 | 268.131 | 113.332 | 0.423 | 100.718 | 430.786 |
| k8s | SentCDC | 10.000 | 1340.656 | 113.332 | 0.085 | 8.391 | 47.769 |
| k8s | STL | 0.500 | 67.033 | 178.273 | 2.659 | 69192.655 | 147411.250 |
| k8s | STL | 1.100 | 147.472 | 178.273 | 1.209 | 14043.413 | 27147.110 |
| k8s | STL | 2.000 | 268.131 | 178.273 | 0.665 | 409.876 | 1576.670 |
| k8s | STL | 10.000 | 1340.656 | 178.273 | 0.133 | 14.942 | 79.988 |
| k8s | KARA | 0.500 | 67.033 | 101.450 | 1.513 | 23619.656 | 49592.163 |
| k8s | KARA | 1.100 | 147.472 | 101.450 | 0.688 | 479.145 | 1762.162 |
| k8s | KARA | 2.000 | 268.131 | 101.450 | 0.378 | 85.803 | 390.599 |
| k8s | KARA | 10.000 | 1340.656 | 101.450 | 0.076 | 7.665 | 43.987 |
| k8s | DIFF[REC] | 0.500 | 67.033 | 104.128 | 1.553 | 25072.338 | 52786.010 |
| k8s | DIFF[REC] | 1.100 | 147.472 | 104.128 | 0.706 | 524.357 | 1895.120 |
| k8s | DIFF[REC] | 2.000 | 268.131 | 104.128 | 0.388 | 89.181 | 396.250 |
| k8s | DIFF[REC] | 10.000 | 1340.656 | 104.128 | 0.078 | 7.856 | 44.846 |
| k8s | DOC | 0.500 | 67.033 | 806.936 | 12.038 | 425565.916 | 946934.377 |
| k8s | DOC | 1.100 | 147.472 | 806.936 | 5.472 | 173600.407 | 386320.583 |
| k8s | DOC | 2.000 | 268.131 | 806.936 | 3.009 | 79322.178 | 176238.230 |
| k8s | DOC | 10.000 | 1340.656 | 806.936 | 0.602 | 206.704 | 734.132 |
| wiki | FIX | 0.500 | 73.548 | 379.274 | 5.157 | 362541.868 | 804446.649 |
| wiki | FIX | 1.100 | 161.806 | 379.274 | 2.344 | 115536.905 | 272117.262 |
| wiki | FIX | 2.000 | 294.193 | 379.274 | 1.289 | 27875.788 | 77805.339 |
| wiki | FIX | 10.000 | 1470.965 | 379.274 | 0.258 | 13.859 | 68.811 |
| wiki | REC | 0.500 | 73.548 | 147.097 | 2.000 | 96380.815 | 206610.461 |
| wiki | REC | 1.100 | 161.806 | 147.097 | 0.909 | 3512.854 | 16445.475 |
| wiki | REC | 2.000 | 294.193 | 147.097 | 0.500 | 152.292 | 1335.574 |
| wiki | REC | 10.000 | 1470.965 | 147.097 | 0.100 | 1.165 | 3.413 |
| wiki | STRUCT-M | 0.500 | 73.548 | 126.177 | 1.716 | 70259.480 | 152878.571 |
| wiki | STRUCT-M | 1.100 | 161.806 | 126.177 | 0.780 | 1294.302 | 7205.548 |
| wiki | STRUCT-M | 2.000 | 294.193 | 126.177 | 0.429 | 82.328 | 533.102 |
| wiki | STRUCT-M | 10.000 | 1470.965 | 126.177 | 0.086 | 1.036 | 2.778 |
| wiki | SentCDC | 0.500 | 73.548 | 112.868 | 1.535 | 54064.981 | 119368.528 |
| wiki | SentCDC | 1.100 | 161.806 | 112.868 | 0.698 | 666.575 | 3895.888 |
| wiki | SentCDC | 2.000 | 294.193 | 112.868 | 0.384 | 54.114 | 286.117 |
| wiki | SentCDC | 10.000 | 1470.965 | 112.868 | 0.077 | 0.898 | 2.245 |
| wiki | STL | 0.500 | 73.548 | 182.586 | 2.483 | 138362.173 | 298144.967 |
| wiki | STL | 1.100 | 161.806 | 182.586 | 1.128 | 16394.454 | 44895.394 |
| wiki | STL | 2.000 | 294.193 | 182.586 | 0.621 | 461.486 | 2885.737 |
| wiki | STL | 10.000 | 1470.965 | 182.586 | 0.124 | 1.638 | 5.234 |
| wiki | KARA | 0.500 | 73.548 | 120.676 | 1.641 | 64237.940 | 138710.957 |
| wiki | KARA | 1.100 | 161.806 | 120.676 | 0.746 | 1019.051 | 6285.828 |
| wiki | KARA | 2.000 | 294.193 | 120.676 | 0.410 | 68.307 | 499.879 |
| wiki | KARA | 10.000 | 1470.965 | 120.676 | 0.082 | 0.930 | 2.343 |
| wiki | DIFF[REC] | 0.500 | 73.548 | 129.047 | 1.755 | 74947.784 | 160234.304 |
| wiki | DIFF[REC] | 1.100 | 161.806 | 129.047 | 0.798 | 1601.605 | 9259.821 |
| wiki | DIFF[REC] | 2.000 | 294.193 | 129.047 | 0.439 | 85.408 | 812.254 |
| wiki | DIFF[REC] | 10.000 | 1470.965 | 129.047 | 0.088 | 1.004 | 2.560 |
| wiki | DOC | 0.500 | 73.548 | 711.549 | 9.675 | 771086.148 | 1.67e+06 |
| wiki | DOC | 1.100 | 161.806 | 711.549 | 4.398 | 298317.551 | 660323.412 |
| wiki | DOC | 2.000 | 294.193 | 711.549 | 2.419 | 124671.555 | 287292.904 |
| wiki | DOC | 10.000 | 1470.965 | 711.549 | 0.484 | 194.197 | 1121.249 |

Files: `results/e7_unit_costs.json`, `results/e7_unit_calls.csv`, `results/e7_timing.csv`, `results/e7c_llm.json`, `results/e7_analysis.json`, `results/e7_derived_costs.csv`, `results/e7_judge3.csv`, `results/e7_staleness.csv`

## E8: sensitivity (counts)


`e8_sens_k8s.csv`:

| config | edits | ins | toks_frac | toks_frac_workload | mean_chunk_tok |
|---|---|---|---|---|---|
| BCDC avg=1800 | 14,924 | 3.246 | 0.307 | 0.307 | 334.004 |
| BCDC avg=450 | 14,924 | 3.987 | 0.174 | 0.174 | 138.867 |
| BCDC avg=900 | 14,924 | 3.183 | 0.238 | 0.238 | 246.367 |
| GCDC Tmin=100 p=0.15 | 14,924 | 3.224 | 0.166 | 0.166 | 160.815 |
| GCDC Tmin=100 p=0.3 | 14,924 | 3.607 | 0.151 | 0.151 | 136.343 |
| GCDC Tmin=150 p=0.15 | 14,924 | 3.052 | 0.175 | 0.175 | 183.617 |
| GCDC Tmin=150 p=0.3 | 14,924 | 3.317 | 0.165 | 0.165 | 164.491 |
| GCDC Tmin=25 p=0.15 | 14,924 | 3.698 | 0.148 | 0.148 | 112.019 |
| GCDC Tmin=25 p=0.3 | 14,924 | 4.488 | 0.124 | 0.124 | 81.288 |
| GCDC Tmin=50 p=0.15 | 14,924 | 3.499 | 0.153 | 0.153 | 130.832 |
| GCDC Tmin=50 p=0.3 | 14,924 | 4.101 | 0.134 | 0.134 | 101.988 |
| REC T=128 | 14,924 | 5.239 | 0.133 | 0.133 | 92.675 |
| REC T=200 | 14,924 | 4.388 | 0.186 | 0.186 | 152.960 |
| REC T=320 | 14,924 | 3.709 | 0.259 | 0.259 | 259.839 |
| STL min=100 max=200 | 14,924 | 6.302 | 0.284 | 0.284 | 172.801 |
| STL min=120 max=320 | 14,924 | 4.312 | 0.293 | 0.293 | 250.419 |
| STL min=176 max=352 | 14,924 | 4.287 | 0.327 | 0.327 | 288.734 |
| STL min=180 max=480 | 14,924 | 3.358 | 0.329 | 0.329 | 357.844 |
| STL min=60 max=120 | 14,924 | 8.887 | 0.250 | 0.250 | 106.783 |
| STRUCT-M T=128 | 14,924 | 4.292 | 0.116 | 0.116 | 90.596 |
| STRUCT-M T=200 | 14,924 | 3.558 | 0.144 | 0.144 | 135.507 |
| STRUCT-M T=320 | 14,924 | 3.029 | 0.176 | 0.176 | 191.331 |
| SentCDC r=1 lam=0.5 | 14,924 | 5.566 | 0.109 | 0.109 | 54.387 |
| SentCDC r=1 lam=1.0 | 14,924 | 5.642 | 0.109 | 0.109 | 53.609 |
| SentCDC r=2 lam=0.5 | 14,924 | 4.558 | 0.130 | 0.130 | 80.295 |
| SentCDC r=2 lam=1.0 | 14,924 | 4.615 | 0.129 | 0.129 | 78.811 |
| SentCDC r=3 lam=0.5 | 14,924 | 4.114 | 0.144 | 0.144 | 99.754 |
| SentCDC r=3 lam=1.0 | 14,924 | 4.150 | 0.142 | 0.142 | 98.087 |
| SentCDC r=4 lam=0.5 | 14,924 | 3.806 | 0.154 | 0.154 | 117.157 |
| SentCDC r=4 lam=1.0 | 14,924 | 3.823 | 0.153 | 0.153 | 115.827 |
| SentCDC r=5 lam=0.5 | 14,924 | 3.604 | 0.162 | 0.162 | 131.154 |
| SentCDC r=5 lam=1.0 | 14,924 | 3.616 | 0.161 | 0.161 | 130.066 |
| SentCDC r=6 lam=0.5 | 14,924 | 3.490 | 0.167 | 0.167 | 141.214 |
| SentCDC r=6 lam=1.0 | 14,924 | 3.495 | 0.167 | 0.167 | 140.186 |

`e8_sens_wiki.csv`:

| config | edits | ins | toks_frac | toks_frac_workload | mean_chunk_tok |
|---|---|---|---|---|---|
| BCDC avg=1800 | 82,949 | 1.642 | 0.487 | 0.487 | 296.170 |
| BCDC avg=450 | 82,949 | 2.112 | 0.270 | 0.270 | 105.906 |
| BCDC avg=900 | 82,949 | 1.688 | 0.373 | 0.373 | 197.254 |
| GCDC Tmin=100 p=0.15 | 82,949 | 1.706 | 0.255 | 0.255 | 116.652 |
| GCDC Tmin=100 p=0.3 | 82,949 | 1.866 | 0.237 | 0.237 | 102.273 |
| GCDC Tmin=150 p=0.15 | 82,949 | 1.624 | 0.270 | 0.270 | 128.256 |
| GCDC Tmin=150 p=0.3 | 82,949 | 1.723 | 0.257 | 0.257 | 118.099 |
| GCDC Tmin=25 p=0.15 | 82,949 | 1.937 | 0.226 | 0.226 | 88.653 |
| GCDC Tmin=25 p=0.3 | 82,949 | 2.314 | 0.191 | 0.191 | 64.806 |
| GCDC Tmin=50 p=0.15 | 82,949 | 1.838 | 0.236 | 0.236 | 99.856 |
| GCDC Tmin=50 p=0.3 | 82,949 | 2.109 | 0.208 | 0.208 | 79.618 |
| REC T=128 | 82,949 | 2.321 | 0.245 | 0.245 | 88.532 |
| REC T=200 | 82,949 | 1.946 | 0.320 | 0.320 | 142.029 |
| REC T=320 | 82,949 | 1.677 | 0.424 | 0.424 | 232.516 |
| STL min=100 max=200 | 82,949 | 2.694 | 0.413 | 0.413 | 162.123 |
| STL min=120 max=320 | 82,949 | 2.005 | 0.448 | 0.448 | 228.212 |
| STL min=176 max=352 | 82,949 | 1.949 | 0.483 | 0.483 | 265.867 |
| STL min=180 max=480 | 82,949 | 1.670 | 0.509 | 0.509 | 319.649 |
| STL min=60 max=120 | 82,949 | 3.812 | 0.371 | 0.371 | 103.212 |
| STRUCT-M T=128 | 82,949 | 2.097 | 0.201 | 0.201 | 74.683 |
| STRUCT-M T=200 | 82,949 | 1.792 | 0.240 | 0.240 | 102.460 |
| STRUCT-M T=320 | 82,949 | 1.587 | 0.278 | 0.278 | 130.774 |
| SentCDC r=1 lam=0.5 | 82,949 | 2.855 | 0.167 | 0.167 | 46.498 |
| SentCDC r=1 lam=1.0 | 82,949 | 2.904 | 0.164 | 0.164 | 45.377 |
| SentCDC r=2 lam=0.5 | 82,949 | 2.414 | 0.200 | 0.200 | 65.524 |
| SentCDC r=2 lam=1.0 | 82,949 | 2.454 | 0.197 | 0.197 | 64.036 |
| SentCDC r=3 lam=0.5 | 82,949 | 2.195 | 0.221 | 0.221 | 79.213 |
| SentCDC r=3 lam=1.0 | 82,949 | 2.235 | 0.217 | 0.217 | 77.293 |
| SentCDC r=4 lam=0.5 | 82,949 | 2.063 | 0.235 | 0.235 | 89.371 |
| SentCDC r=4 lam=1.0 | 82,949 | 2.100 | 0.232 | 0.232 | 87.354 |
| SentCDC r=5 lam=0.5 | 82,949 | 1.977 | 0.245 | 0.245 | 96.942 |
| SentCDC r=5 lam=1.0 | 82,949 | 2.010 | 0.242 | 0.242 | 94.912 |
| SentCDC r=6 lam=0.5 | 82,949 | 1.916 | 0.253 | 0.253 | 102.951 |
| SentCDC r=6 lam=1.0 | 82,949 | 1.943 | 0.249 | 0.249 | 100.939 |

Files: `results/e8_sens_k8s.csv`, `results/e8_sens_wiki.csv`

## E9 / T3: RAG answer quality

Answers by qwen3:8b (temperature 0) on the E4 contexts (budget 1,024 tokens); closed-book = no context.

| stream | config | n | em | f1 | contains |
|---|---|---|---|---|---|
| cpy | KARA | 69 | 0.130 | 0.253 | 0.174 |
| cpy | REC | 69 | 0.101 | 0.209 | 0.130 |
| cpy | REC-stale20 | 69 | 0.087 | 0.180 | 0.116 |
| cpy | SentCDC | 69 | 0.101 | 0.211 | 0.145 |
| cpy | closed-book | 69 | 0.014 | 0.067 | 0.014 |
| k8s | KARA | 62 | 0.306 | 0.439 | 0.355 |
| k8s | REC | 62 | 0.161 | 0.284 | 0.242 |
| k8s | REC-stale20 | 62 | 0.113 | 0.234 | 0.177 |
| k8s | SentCDC | 62 | 0.210 | 0.345 | 0.274 |
| k8s | closed-book | 62 | 0.016 | 0.092 | 0.032 |
| wiki | KARA | 169 | 0.290 | 0.427 | 0.337 |
| wiki | REC | 169 | 0.308 | 0.441 | 0.355 |
| wiki | REC-stale20 | 169 | 0.237 | 0.371 | 0.278 |
| wiki | SentCDC | 169 | 0.320 | 0.453 | 0.361 |
| wiki | closed-book | 169 | 0.053 | 0.147 | 0.071 |

Paired statistics vs REC (contains):

| stream | config | n | mean | ci_lo | ci_hi | diff | diff_ci_lo | diff_ci_hi | p_holm | tost90_lo | tost90_hi | n_required | tost_equivalent | mde_boot | mde_mcnemar |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cpy | KARA | 69 | 0.174 | 0.088 | 0.261 | 0.043 | -0.029 | 0.119 | 0.906 | -0.026 | 0.116 | 1569.000 | False | 0.105 | 0.107 |
| cpy | REC | 69 | 0.130 | 0.059 | 0.205 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| cpy | SentCDC | 69 | 0.145 | 0.070 | 0.232 | 0.014 | -0.059 | 0.088 | 1.000 | -0.056 | 0.087 | 1569.000 | False | 0.106 | 0.107 |
| cpy | closed-book | 69 | 0.014 | 0.000 | 0.044 | -0.116 | -0.186 | -0.045 | 0.023 | -0.196 | -0.049 | 1793.000 | False | 0.102 | 0.115 |
| k8s | KARA | 62 | 0.355 | 0.246 | 0.467 | 0.113 | 0.000 | 0.226 | 0.185 | 0.016 | 0.208 | 3241.000 | False | 0.160 | 0.163 |
| k8s | REC | 62 | 0.242 | 0.136 | 0.355 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| k8s | SentCDC | 62 | 0.274 | 0.167 | 0.385 | 0.032 | -0.066 | 0.129 | 0.754 | -0.056 | 0.120 | 2494.000 | False | 0.140 | 0.143 |
| k8s | closed-book | 62 | 0.032 | 0.000 | 0.082 | -0.210 | -0.317 | -0.115 | 0.000732 | -0.306 | -0.122 | 3241.000 | False | 0.148 | 0.163 |
| wiki | KARA | 169 | 0.337 | 0.266 | 0.407 | -0.018 | -0.074 | 0.036 | 1.000 | -0.063 | 0.028 | 1921.000 | False | 0.077 | 0.076 |
| wiki | REC | 169 | 0.355 | 0.279 | 0.426 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| wiki | SentCDC | 169 | 0.361 | 0.288 | 0.434 | 0.006 | -0.060 | 0.071 | 1.000 | -0.045 | 0.057 | 2470.000 | False | 0.090 | 0.086 |
| wiki | closed-book | 169 | 0.071 | 0.035 | 0.112 | -0.284 | -0.351 | -0.216 | 2.13e-14 | -0.343 | -0.226 | 4391.000 | False | 0.099 | 0.115 |
| all | KARA | 300 | 0.303 | 0.257 | 0.355 | 0.023 | -0.017 | 0.064 | 0.698 | -0.012 | 0.059 | 2113.000 | False | 0.059 | 0.060 |
| all | REC | 300 | 0.280 | 0.233 | 0.331 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| all | SentCDC | 300 | 0.293 | 0.244 | 0.347 | 0.013 | -0.030 | 0.057 | 0.698 | -0.023 | 0.050 | 2267.000 | False | 0.063 | 0.062 |
| all | closed-book | 300 | 0.050 | 0.027 | 0.076 | -0.230 | -0.278 | -0.185 | 1.02e-20 | -0.272 | -0.191 | 3556.000 | False | 0.067 | 0.078 |

T3 table: `results/t3_answers.csv` / `results/t3_answers.tex` (experiments/tables.py).

Freshness, stale-20 vs maintained REC (contains; all questions; paired, clustered bootstrap 95% CI, McNemar):

| stream | config | n | mean | ci_lo | ci_hi | diff | diff_ci_lo | diff_ci_hi | mcnemar_p | discordant |
|---|---|---|---|---|---|---|---|---|---|---|
| cpy | REC | 69 | 0.130 | 0.059 | 0.205 | n/a | n/a | n/a | n/a | n/a |
| cpy | REC-stale20 | 69 | 0.116 | 0.045 | 0.194 | -0.014 | -0.083 | 0.045 | 1.000 | 5.000 |
| k8s | REC | 62 | 0.242 | 0.136 | 0.355 | n/a | n/a | n/a | n/a | n/a |
| k8s | REC-stale20 | 62 | 0.177 | 0.095 | 0.274 | -0.065 | -0.148 | 0.017 | 0.289 | 8.000 |
| wiki | REC | 169 | 0.355 | 0.279 | 0.426 | n/a | n/a | n/a | n/a | n/a |
| wiki | REC-stale20 | 169 | 0.278 | 0.212 | 0.347 | -0.077 | -0.136 | -0.018 | 0.019 | 27.000 |
| all | REC | 300 | 0.280 | 0.233 | 0.331 | n/a | n/a | n/a | n/a | n/a |
| all | REC-stale20 | 300 | 0.220 | 0.173 | 0.268 | -0.060 | -0.100 | -0.020 | 0.006 | 40.000 |

Files: `results/e9_stats_fresh.csv`, `results/e9_answers_fresh.csv`

Freshness, stale-20 vs maintained REC (contains; edit-proximal questions only (evidence sentence unit absent from version n-20, i.e. from the stale index's version, by construction); paired, clustered bootstrap 95% CI, McNemar):

| stream | config | n | mean | ci_lo | ci_hi | diff | diff_ci_lo | diff_ci_hi | mcnemar_p | discordant |
|---|---|---|---|---|---|---|---|---|---|---|
| cpy | REC | 22 | 0.182 | 0.043 | 0.364 | n/a | n/a | n/a | n/a | n/a |
| cpy | REC-stale20 | 22 | 0.045 | 0.000 | 0.143 | -0.136 | -0.304 | 0.000 | 0.250 | 3.000 |
| k8s | REC | 31 | 0.226 | 0.097 | 0.375 | n/a | n/a | n/a | n/a | n/a |
| k8s | REC-stale20 | 31 | 0.097 | 0.000 | 0.200 | -0.129 | -0.258 | -0.032 | 0.125 | 4.000 |
| wiki | REC | 85 | 0.294 | 0.202 | 0.398 | n/a | n/a | n/a | n/a | n/a |
| wiki | REC-stale20 | 85 | 0.176 | 0.103 | 0.265 | -0.118 | -0.202 | -0.036 | 0.013 | 14.000 |
| all | REC | 138 | 0.261 | 0.185 | 0.331 | n/a | n/a | n/a | n/a | n/a |
| all | REC-stale20 | 138 | 0.138 | 0.085 | 0.197 | -0.123 | -0.186 | -0.059 | 0.000221 | 21.000 |

Files: `results/e9_stats_fresh_prox.csv`, `results/e9_answers_fresh_prox.csv`

Files: `results/e9_stats.csv`, `results/e9_answers_stale.csv`, `results/e9_stats.csv`, `results/t3_answers.csv`

## Tier B (run only after every Tier A result existed)

Driver: `experiments/tierB.sh` (hard stop for launches Nov 10 14:00). Items absent below were not run.

### REC-O (REC + 40-token overlap) counts vs REC and FIXO (S1, per revision)

| stream | T | chunker | edits | docs | toks_frac | ci_lo | ci_hi | INS | size |
|---|---|---|---|---|---|---|---|---|---|
| cpy | 128 | FIXO | 25,817 | 300 | 0.655 | 0.644 | 0.667 | 70.576 | 126.977 |
| cpy | 128 | REC | 25,817 | 300 | 0.089 | 0.080 | 0.099 | 4.485 | 131.390 |
| cpy | 200 | FIXO | 25,817 | 300 | 0.663 | 0.652 | 0.676 | 45.499 | 197.530 |
| cpy | 200 | REC | 25,817 | 300 | 0.131 | 0.119 | 0.144 | 4.456 | 206.363 |
| cpy | 200 | REC-O | 25,817 | 300 | 0.128 | 0.116 | 0.141 | 4.222 | 208.621 |
| cpy | 320 | FIXO | 25,817 | 300 | 0.677 | 0.665 | 0.690 | 28.852 | 313.732 |
| cpy | 320 | REC | 25,817 | 300 | 0.186 | 0.172 | 0.204 | 4.434 | 324.937 |
| k8s | 128 | FIXO | 14,924 | 284 | 0.714 | 0.704 | 0.725 | 28.944 | 125.762 |
| k8s | 128 | REC | 14,924 | 284 | 0.165 | 0.153 | 0.177 | 4.640 | 128.153 |
| k8s | 200 | FIXO | 14,924 | 284 | 0.728 | 0.718 | 0.740 | 18.813 | 194.704 |
| k8s | 200 | REC | 14,924 | 284 | 0.222 | 0.207 | 0.237 | 4.005 | 201.333 |
| k8s | 200 | REC-O | 14,924 | 284 | 0.225 | 0.210 | 0.241 | 4.005 | 203.770 |
| k8s | 320 | FIXO | 14,924 | 284 | 0.753 | 0.742 | 0.765 | 12.130 | 307.131 |
| k8s | 320 | REC | 14,924 | 284 | 0.295 | 0.278 | 0.313 | 3.446 | 317.520 |
| wiki | 128 | FIXO | 82,949 | 1,000 | 0.702 | 0.693 | 0.710 | 9.154 | 119.859 |
| wiki | 128 | REC | 82,949 | 1,000 | 0.305 | 0.294 | 0.316 | 1.992 | 130.837 |
| wiki | 200 | FIXO | 82,949 | 1,000 | 0.738 | 0.729 | 0.747 | 6.163 | 181.379 |
| wiki | 200 | REC | 82,949 | 1,000 | 0.394 | 0.380 | 0.407 | 1.744 | 202.305 |
| wiki | 200 | REC-O | 82,949 | 1,000 | 0.402 | 0.388 | 0.416 | 1.763 | 205.013 |
| wiki | 320 | FIXO | 82,949 | 1,000 | 0.789 | 0.778 | 0.798 | 4.167 | 277.229 |
| wiki | 320 | REC | 82,949 | 1,000 | 0.505 | 0.488 | 0.521 | 1.526 | 311.418 |

Files: `results/e1_edits_tierB.parquet`, `results/e1_summary_tierB.csv`

### E8 quality points (E4 protocol, native 30k background; recall@1024)


`_e8q_T200` (calibration T=200; SentCDC@lam=1 = calibrated r with lambda=1):

| stream | chunker | n | recall@1024 | ndcg10 | size |
|---|---|---|---|---|---|
| cpy | REC | 200 | 0.345 | 0.303 | 205.900 |
| cpy | STRUCT-M | 200 | 0.350 | 0.308 | 207.200 |
| cpy | SentCDC | 200 | 0.275 | 0.234 | 201.000 |
| k8s | REC | 200 | 0.350 | 0.290 | 206.400 |
| k8s | STRUCT-M | 200 | 0.350 | 0.302 | 228.800 |
| k8s | SentCDC | 200 | 0.370 | 0.291 | 200.300 |
| wiki | REC | 500 | 0.550 | 0.477 | 200.600 |
| wiki | STRUCT-M | 500 | 0.568 | 0.483 | 152.600 |
| wiki | SentCDC | 500 | 0.616 | 0.508 | 128.500 |

Paired vs REC (same run):

| stream | config | n | mean | diff | diff_ci_lo | diff_ci_hi | p_holm | mde_boot |
|---|---|---|---|---|---|---|---|---|
| cpy | REC | 200 | 0.345 | n/a | n/a | n/a | n/a | n/a |
| cpy | STRUCT-M | 200 | 0.350 | 0.005 | -0.046 | 0.058 | 1.000 | 0.075 |
| cpy | SentCDC | 200 | 0.275 | -0.070 | -0.127 | -0.010 | 0.058 | 0.084 |
| k8s | REC | 200 | 0.350 | n/a | n/a | n/a | n/a | n/a |
| k8s | STRUCT-M | 200 | 0.350 | 0.000 | -0.058 | 0.057 | 1.000 | 0.082 |
| k8s | SentCDC | 200 | 0.370 | 0.020 | -0.044 | 0.086 | 1.000 | 0.093 |
| wiki | REC | 500 | 0.550 | n/a | n/a | n/a | n/a | n/a |
| wiki | STRUCT-M | 500 | 0.568 | 0.018 | -0.016 | 0.053 | 0.368 | 0.050 |
| wiki | SentCDC | 500 | 0.616 | 0.066 | 0.030 | 0.101 | 0.000757 | 0.051 |
| all | REC | 900 | 0.460 | n/a | n/a | n/a | n/a | n/a |
| all | STRUCT-M | 900 | 0.471 | 0.011 | -0.015 | 0.037 | 0.456 | 0.037 |
| all | SentCDC | 900 | 0.486 | 0.026 | -0.001 | 0.053 | 0.173 | 0.039 |

`_e8q_T320` (calibration T=320; SentCDC@lam=1 = calibrated r with lambda=1):

| stream | chunker | n | recall@1024 | ndcg10 | size |
|---|---|---|---|---|---|
| cpy | REC | 200 | 0.240 | 0.253 | 327.700 |
| cpy | STRUCT-M | 200 | 0.240 | 0.252 | 331.300 |
| cpy | SentCDC | 200 | 0.270 | 0.230 | 221.900 |
| k8s | REC | 200 | 0.250 | 0.283 | 326.900 |
| k8s | STRUCT-M | 200 | 0.290 | 0.289 | 258.700 |
| k8s | SentCDC | 200 | 0.370 | 0.294 | 200.300 |
| wiki | REC | 500 | 0.452 | 0.437 | 321.800 |
| wiki | STRUCT-M | 500 | 0.568 | 0.483 | 152.600 |
| wiki | SentCDC | 500 | 0.616 | 0.508 | 128.500 |

Paired vs REC (same run):

| stream | config | n | mean | diff | diff_ci_lo | diff_ci_hi | p_holm | mde_boot |
|---|---|---|---|---|---|---|---|---|
| cpy | REC | 200 | 0.240 | n/a | n/a | n/a | n/a | n/a |
| cpy | STRUCT-M | 200 | 0.240 | 0.000 | -0.065 | 0.066 | 1.000 | 0.096 |
| cpy | SentCDC | 200 | 0.270 | 0.030 | -0.030 | 0.096 | 0.882 | 0.092 |
| k8s | REC | 200 | 0.250 | n/a | n/a | n/a | n/a | n/a |
| k8s | STRUCT-M | 200 | 0.290 | 0.040 | -0.020 | 0.110 | 0.280 | 0.093 |
| k8s | SentCDC | 200 | 0.370 | 0.120 | 0.054 | 0.192 | 0.001 | 0.097 |
| wiki | REC | 500 | 0.452 | n/a | n/a | n/a | n/a | n/a |
| wiki | STRUCT-M | 500 | 0.568 | 0.116 | 0.081 | 0.153 | 7.29e-10 | 0.051 |
| wiki | SentCDC | 500 | 0.616 | 0.164 | 0.131 | 0.200 | 2.18e-18 | 0.050 |
| all | REC | 900 | 0.360 | n/a | n/a | n/a | n/a | n/a |
| all | STRUCT-M | 900 | 0.433 | 0.073 | 0.044 | 0.102 | 9.69e-07 | 0.042 |
| all | SentCDC | 900 | 0.484 | 0.124 | 0.095 | 0.153 | 1.22e-16 | 0.041 |

`_e8q_lam1` (calibration T=128; SentCDC@lam=1 = calibrated r with lambda=1):

| stream | chunker | n | recall@1024 | ndcg10 | size |
|---|---|---|---|---|---|
| cpy | REC | 200 | 0.420 | 0.335 | 131.100 |
| cpy | SentCDC | 200 | 0.365 | 0.263 | 133.200 |
| cpy | SentCDC@lam=1 | 200 | 0.360 | 0.252 | 132.300 |
| k8s | REC | 200 | 0.460 | 0.341 | 131.200 |
| k8s | SentCDC | 200 | 0.395 | 0.295 | 158.100 |
| k8s | SentCDC@lam=1 | 200 | 0.390 | 0.291 | 156.400 |
| wiki | REC | 500 | 0.616 | 0.508 | 129.800 |
| wiki | SentCDC | 500 | 0.618 | 0.512 | 127.300 |
| wiki | SentCDC@lam=1 | 500 | 0.622 | 0.517 | 126.400 |

Paired vs REC (same run):

| stream | config | n | mean | diff | diff_ci_lo | diff_ci_hi | p_holm | mde_boot |
|---|---|---|---|---|---|---|---|---|
| cpy | REC | 200 | 0.420 | n/a | n/a | n/a | n/a | n/a |
| cpy | SentCDC | 200 | 0.365 | -0.055 | -0.115 | 0.010 | 0.146 | 0.087 |
| cpy | SentCDC@lam=1 | 200 | 0.360 | -0.060 | -0.119 | 0.000 | 0.146 | 0.089 |
| k8s | REC | 200 | 0.460 | n/a | n/a | n/a | n/a | n/a |
| k8s | SentCDC | 200 | 0.395 | -0.065 | -0.123 | -0.005 | 0.058 | 0.086 |
| k8s | SentCDC@lam=1 | 200 | 0.390 | -0.070 | -0.125 | -0.011 | 0.058 | 0.084 |
| wiki | REC | 500 | 0.616 | n/a | n/a | n/a | n/a | n/a |
| wiki | SentCDC | 500 | 0.618 | 0.002 | -0.034 | 0.037 | 1.000 | 0.050 |
| wiki | SentCDC@lam=1 | 500 | 0.622 | 0.006 | -0.031 | 0.041 | 1.000 | 0.052 |
| all | REC | 900 | 0.538 | n/a | n/a | n/a | n/a | n/a |
| all | SentCDC | 900 | 0.512 | -0.026 | -0.053 | 0.001 | 0.150 | 0.039 |
| all | SentCDC@lam=1 | 900 | 0.512 | -0.026 | -0.054 | 0.002 | 0.150 | 0.040 |

Files: `results/e4_quality_e8q_T200.csv`, `results/e4_quality_e8q_T320.csv`, `results/e4_quality_e8q_lam1.csv`, `results/e4_stats_e8q_T200.csv`, `results/e4_stats_e8q_T320.csv`, `results/e4_stats_e8q_lam1.csv`

### E6 REBUILD at tau 0.05 / 0.1 (seed 0)

| chunker | policy | recall_exact | recall_fresh | tomb_share_churned | lat_ms | rebuilds | rebuild_s | edits |
|---|---|---|---|---|---|---|---|---|
| DIFF[REC] | REBUILD(0.05) | 0.988 | 0.988 | 0.007 | 0.436 | 491.000 | 2635.646 | 7955.000 |
| DIFF[REC] | REBUILD(0.1) | 0.988 | 0.988 | 0.022 | 0.459 | 245.000 | 1267.307 | 7955.000 |
| REC | REBUILD(0.05) | 0.986 | 0.987 | 0.018 | 0.546 | 556.000 | 3548.930 | 7955.000 |
| REC | REBUILD(0.1) | 0.988 | 0.986 | 0.015 | 0.564 | 276.000 | 1694.883 | 7955.000 |
| SentCDC | REBUILD(0.05) | 0.992 | 0.990 | 0.009 | 0.705 | 352.000 | 2214.254 | 7955.000 |
| SentCDC | REBUILD(0.1) | 0.990 | 0.970 | 0.037 | 0.597 | 175.000 | 1149.096 | 7955.000 |

Files: `results/e6_index_tauB.csv`

### REC with 40-token overlap

Encoder: `BAAI/bge-small-en-v1.5` on cpu. Nominal T=128. Background chunks: 30000.

| stream | config | n | mean | diff | diff_ci_lo | diff_ci_hi | p_holm | tost_equivalent |
|---|---|---|---|---|---|---|---|---|
| cpy | REC | 200 | 0.420 | n/a | n/a | n/a | n/a | n/a |
| cpy | REC-O | 200 | 0.390 | -0.030 | -0.069 | 0.005 | 0.180 | False |
| k8s | REC | 200 | 0.460 | n/a | n/a | n/a | n/a | n/a |
| k8s | REC-O | 200 | 0.420 | -0.040 | -0.090 | 0.010 | 0.169 | False |
| wiki | REC | 500 | 0.616 | n/a | n/a | n/a | n/a | n/a |
| wiki | REC-O | 500 | 0.604 | -0.012 | -0.036 | 0.014 | 0.418 | False |
| all | REC | 900 | 0.538 | n/a | n/a | n/a | n/a | n/a |
| all | REC-O | 900 | 0.516 | -0.022 | -0.042 | -0.003 | 0.031 | False |

Files: `results/e4_stats_e4_reco.csv`, `results/e4_quality_e4_reco.meta.json`, `results/e4_quality_e4_reco.csv`

### Single-unit vectors with two neighboring units on each side

Encoder: `BAAI/bge-small-en-v1.5` on cpu. Nominal T=128. Background chunks: 30000.

| stream | config | n | mean | diff | diff_ci_lo | diff_ci_hi | p_holm | tost_equivalent |
|---|---|---|---|---|---|---|---|---|
| cpy | REC | 200 | 0.420 | n/a | n/a | n/a | n/a | n/a |
| cpy | SENTWIN | 200 | 0.670 | 0.250 | 0.180 | 0.320 | 8e-10 | False |
| k8s | REC | 200 | 0.460 | n/a | n/a | n/a | n/a | n/a |
| k8s | SENTWIN | 200 | 0.640 | 0.180 | 0.107 | 0.260 | 1.92e-05 | False |
| wiki | REC | 500 | 0.616 | n/a | n/a | n/a | n/a | n/a |
| wiki | SENTWIN | 500 | 0.818 | 0.202 | 0.156 | 0.246 | 1.13e-17 | False |
| all | REC | 900 | 0.538 | n/a | n/a | n/a | n/a | n/a |
| all | SENTWIN | 900 | 0.746 | 0.208 | 0.174 | 0.241 | 4.82e-30 | False |

Files: `results/e4_stats_e4_sentwin.csv`, `results/e4_quality_e4_sentwin.meta.json`, `results/e4_quality_e4_sentwin.csv`

## Review analyses (pre-draft review requests)

**Version accounting.** The extracted stream file holds `raw_rows` versions; the replay (common.load_stream) drops whitespace-only versions (revision-deleted text or blanking) and then consecutive text-identical versions:

| stream | raw_rows | dropped_empty | dropped_identical | replay_versions | docs | replay_edits |
|---|---|---|---|---|---|---|
| wiki | 83,992 | 36 | 7 | 83,949 | 1,000 | 82,949 |
| k8s | 15,208 | 0 | 0 | 15,208 | 284 | 14,924 |
| cpy | 26,117 | 0 | 0 | 26,117 | 300 | 25,817 |

Files: `results/review_versions.json`

**E1 at T=128 (per revision, S1) with document-clustered bootstrap 95% CIs** (2,000 resamples over documents); `dfrac_vs_REC` = paired difference in toks_frac vs REC on the same edits, clustered CI. Content-aligned = REC, STRUCT-M, SentCDC, GCDC, BCDC, KARA, DIFF[REC], DIFF[STRUCT-M]. Note the achieved sizes differ (`chunk_tok_mean`), so the raw T=128 differences mix placement and size (the interpolated comparison is in e1_matched_size.csv).

| stream | chunker | chunk_tok_mean | toks_frac_mean | toks_frac_ci_lo | toks_frac_ci_hi | dfrac_vs_REC | dfrac_ci_lo | dfrac_ci_hi |
|---|---|---|---|---|---|---|---|---|
| cpy | FIX | 126.4508 | 0.5238 | 0.5148 | 0.5337 | 0.4349 | 0.4239 | 0.4455 |
| cpy | FIXO | 126.9769 | 0.6551 | 0.6438 | 0.6674 | 0.5662 | 0.5539 | 0.5786 |
| cpy | REC | 131.3904 | 0.0889 | 0.0804 | 0.0989 | n/a | n/a | n/a |
| cpy | STRUCT-M | 128.4813 | 0.0778 | 0.0696 | 0.0875 | -0.0111 | -0.0135 | -0.0089 |
| cpy | SentCDC | 133.5225 | 0.0803 | 0.0723 | 0.0898 | -0.0086 | -0.0118 | -0.0056 |
| cpy | GCDC | 124.4377 | 0.0762 | 0.0685 | 0.0857 | -0.0126 | -0.0159 | -0.0096 |
| cpy | BCDC | 136.5210 | 0.0837 | 0.0756 | 0.0931 | -0.0051 | -0.0075 | -0.0028 |
| cpy | STL | 126.2582 | 0.1438 | 0.1320 | 0.1582 | 0.0549 | 0.0505 | 0.0601 |
| cpy | STL+snap | 117.7837 | 0.1083 | 0.0987 | 0.1196 | 0.0194 | 0.0171 | 0.0220 |
| cpy | KARA | 122.8604 | 0.0634 | 0.0572 | 0.0710 | -0.0255 | -0.0288 | -0.0227 |
| cpy | DIFF[REC] | 126.8756 | 0.0691 | 0.0624 | 0.0772 | -0.0198 | -0.0225 | -0.0174 |
| cpy | DIFF[STRUCT-M] | 125.7405 | 0.0678 | 0.0610 | 0.0760 | -0.0211 | -0.0239 | -0.0185 |
| cpy | DOC | 131.3904 | 0.9822 | 0.9801 | 0.9841 | 0.8933 | 0.8827 | 0.9024 |
| k8s | FIX | 124.6815 | 0.5725 | 0.5643 | 0.5814 | 0.4073 | 0.3959 | 0.4179 |
| k8s | FIXO | 125.7615 | 0.7144 | 0.7042 | 0.7253 | 0.5492 | 0.5367 | 0.5606 |
| k8s | REC | 128.1533 | 0.1652 | 0.1534 | 0.1776 | n/a | n/a | n/a |
| k8s | STRUCT-M | 131.0975 | 0.1442 | 0.1339 | 0.1549 | -0.0210 | -0.0247 | -0.0174 |
| k8s | SentCDC | 135.6298 | 0.1673 | 0.1557 | 0.1794 | 0.0021 | -0.0025 | 0.0070 |
| k8s | GCDC | 129.9769 | 0.1509 | 0.1402 | 0.1619 | -0.0144 | -0.0190 | -0.0097 |
| k8s | BCDC | 127.7057 | 0.1670 | 0.1557 | 0.1793 | 0.0018 | -0.0013 | 0.0052 |
| k8s | STL | 125.5702 | 0.2663 | 0.2500 | 0.2834 | 0.1011 | 0.0952 | 0.1077 |
| k8s | STL+snap | 116.9827 | 0.2030 | 0.1899 | 0.2165 | 0.0378 | 0.0342 | 0.0414 |
| k8s | KARA | 119.8579 | 0.1338 | 0.1244 | 0.1437 | -0.0314 | -0.0349 | -0.0278 |
| k8s | DIFF[REC] | 124.7684 | 0.1430 | 0.1329 | 0.1537 | -0.0222 | -0.0253 | -0.0193 |
| k8s | DIFF[STRUCT-M] | 131.7962 | 0.1443 | 0.1339 | 0.1549 | -0.0209 | -0.0240 | -0.0177 |
| k8s | DOC | 128.1533 | 0.9853 | 0.9830 | 0.9875 | 0.8201 | 0.8066 | 0.8324 |
| wiki | FIX | 117.0594 | 0.5760 | 0.5678 | 0.5838 | 0.2707 | 0.2613 | 0.2808 |
| wiki | FIXO | 119.8592 | 0.7021 | 0.6925 | 0.7109 | 0.3968 | 0.3859 | 0.4084 |
| wiki | REC | 130.8368 | 0.3053 | 0.2936 | 0.3168 | n/a | n/a | n/a |
| wiki | STRUCT-M | 122.5064 | 0.2740 | 0.2633 | 0.2852 | -0.0313 | -0.0368 | -0.0262 |
| wiki | SentCDC | 119.1898 | 0.2790 | 0.2684 | 0.2899 | -0.0263 | -0.0322 | -0.0205 |
| wiki | GCDC | 118.4938 | 0.2661 | 0.2556 | 0.2769 | -0.0392 | -0.0449 | -0.0339 |
| wiki | BCDC | 123.6530 | 0.2975 | 0.2864 | 0.3084 | -0.0078 | -0.0108 | -0.0047 |
| wiki | STL | 121.6486 | 0.3900 | 0.3775 | 0.4022 | 0.0847 | 0.0812 | 0.0883 |
| wiki | STL+snap | 118.5517 | 0.3547 | 0.3432 | 0.3659 | 0.0494 | 0.0467 | 0.0524 |
| wiki | KARA | 114.8591 | 0.2581 | 0.2480 | 0.2682 | -0.0472 | -0.0499 | -0.0446 |
| wiki | DIFF[REC] | 127.7024 | 0.2915 | 0.2799 | 0.3031 | -0.0138 | -0.0156 | -0.0121 |
| wiki | DIFF[STRUCT-M] | 133.2153 | 0.3081 | 0.2962 | 0.3204 | 0.0029 | -0.0012 | 0.0071 |
| wiki | DOC | 130.8368 | 0.9985 | 0.9981 | 0.9988 | 0.6932 | 0.6818 | 0.7050 |

Content-aligned cluster per stream: cpy: means 0.063-0.089, all 95% CIs overlap = False, max |paired diff vs REC| = 0.025; k8s: means 0.134-0.167, all 95% CIs overlap = False, max |paired diff vs REC| = 0.031; wiki: means 0.258-0.308, all 95% CIs overlap = False, max |paired diff vs REC| = 0.047. Reading: the per-chunker CIs do NOT all overlap and most paired differences vs REC exclude 0, so 'tie' must be stated as 'within a few points' (magnitudes above), not as statistical equivalence.

Files: `results/e1_ci_T128.csv`, `results/e1_ci_T128.json`

**Late-chunking window.** Share of documents whose text fits in one 8192-token jina-v2 window (8190 content tokens; jina tokenizer, no special tokens):

| stream | docs | final_fit_share | final_fit_docs | final_tok_p50 | final_tok_p90 | final_tok_max | version_fit_share | versions | docs_all_versions_fit |
|---|---|---|---|---|---|---|---|---|---|
| wiki | 1,000 | 0.977 | 977 | 1302.000 | 4463.600 | 45,784 | 0.977 | 83949.000 | 967.000 |
| k8s | 284 | 0.838 | 238 | 3041.000 | 9998.400 | 176,080 | n/a | n/a | n/a |
| cpy | 300 | 0.597 | 179 | 6044.500 | 26897.100 | 85,851 | n/a | n/a | n/a |

Files: `results/jina_window.json`

**Predictor ranking without the WIKI scale factor** (e2_predict_edits.parquet; WIKI is the calibration stream, so its scaled errors are in-domain). Per edit: Kendall tau-b of predicted_raw vs measured INS, unscaled APE of the mean and per-edit MdAPE (edits with measured > 0):

| stream | chunker | edits | measured_mean | predicted_raw_mean | ape_mean_unscaled | mdape_edit_unscaled | tau_edit |
|---|---|---|---|---|---|---|---|
| cpy | FIX | 1,500 | 34.654 | 37.535 | 0.083 | 0.069 | 0.790 |
| cpy | REC | 1,500 | 4.294 | 7.414 | 0.726 | 0.667 | 0.590 |
| cpy | STRUCT-M | 1,500 | 3.287 | 7.875 | 1.396 | 0.867 | 0.614 |
| cpy | SENTWIN | 1,500 | 7.060 | 7.412 | 0.050 | 0.000 | 0.972 |
| cpy | SentCDC | 1,500 | 2.835 | 3.483 | 0.229 | 0.198 | 0.667 |
| cpy | DOC | 1,500 | 66.208 | 54.416 | 0.178 | 0.175 | 0.971 |
| k8s | FIX | 1,500 | 15.637 | 17.351 | 0.110 | 0.083 | 0.731 |
| k8s | REC | 1,500 | 4.677 | 6.588 | 0.408 | 0.583 | 0.606 |
| k8s | STRUCT-M | 1,500 | 3.211 | 7.429 | 1.313 | 0.667 | 0.639 |
| k8s | SENTWIN | 1,500 | 8.408 | 9.339 | 0.111 | 0.000 | 0.968 |
| k8s | SentCDC | 1,500 | 3.421 | 3.485 | 0.018 | 0.089 | 0.687 |
| k8s | DOC | 1,500 | 25.181 | 20.024 | 0.205 | 0.206 | 0.951 |
| wiki | FIX | 1,500 | 5.055 | 5.874 | 0.162 | 0.136 | 0.697 |
| wiki | REC | 1,500 | 1.749 | 2.502 | 0.430 | 0.212 | 0.469 |
| wiki | STRUCT-M | 1,500 | 1.515 | 2.379 | 0.571 | 0.083 | 0.429 |
| wiki | SENTWIN | 1,500 | 3.457 | 3.726 | 0.078 | 0.000 | 0.880 |
| wiki | SentCDC | 1,500 | 1.681 | 1.914 | 0.139 | 0.045 | 0.536 |
| wiki | DOC | 1,500 | 8.750 | 6.384 | 0.270 | 0.257 | 0.926 |

Across the non-trivial predictors (SentCDC, REC, STRUCT-M, SENTWIN; 4 points per stream, so tau is coarse): cpy: tau raw 0.00, tau scaled 1.00, MdAPE scaled 0.143, unscaled 0.478; k8s: tau raw 0.33, tau scaled 0.33, MdAPE scaled 0.068, unscaled 0.260; wiki: tau raw 0.67, tau scaled 1.00, MdAPE scaled 0.000, unscaled 0.284.

Files: `results/e2_predict_rank.csv`, `results/e2_predict_rank.json`

**History dependence (experiments/kara_stability.py, T=128).** Key divergence 1 - |K(A) & K(B)| / |K(B)| per document, mean/median over documents: final layout after the per-revision replay vs the monthly replay (same final text, different histories), each vs a fresh chunking of the final text, and the divergence after ONE update from a fresh layout of version i to version i+1 vs the fresh layout of i+1 (<= 10 evenly spaced edits per document):

| stream | docs | chunker | measure | mean | median | n |
|---|---|---|---|---|---|---|
| wiki | 1,000 | DIFF[REC] | final_monthly_vs_fresh | 0.234 | 0.200 | 1,000 |
| wiki | 1,000 | DIFF[REC] | final_revision_vs_fresh | 0.328 | 0.300 | 1,000 |
| wiki | 1,000 | DIFF[REC] | final_revision_vs_monthly | 0.184 | 0.125 | 1,000 |
| wiki | 1,000 | DIFF[REC] | one_edit_vs_fresh | 0.051 | 0.040 | 1,000 |
| wiki | 1,000 | DIFF[STRUCT-M] | final_monthly_vs_fresh | 0.242 | 0.230 | 1,000 |
| wiki | 1,000 | DIFF[STRUCT-M] | final_revision_vs_fresh | 0.299 | 0.286 | 1,000 |
| wiki | 1,000 | DIFF[STRUCT-M] | final_revision_vs_monthly | 0.159 | 0.111 | 1,000 |
| wiki | 1,000 | DIFF[STRUCT-M] | one_edit_vs_fresh | 0.059 | 0.042 | 1,000 |
| wiki | 1,000 | KARA | final_monthly_vs_fresh | 0.773 | 0.842 | 1,000 |
| wiki | 1,000 | KARA | final_revision_vs_fresh | 0.788 | 0.855 | 1,000 |
| wiki | 1,000 | KARA | final_revision_vs_monthly | 0.461 | 0.500 | 1,000 |
| wiki | 1,000 | KARA | one_edit_vs_fresh | 0.192 | 0.182 | 1,000 |
| k8s | 284 | DIFF[REC] | final_monthly_vs_fresh | 0.319 | 0.288 | 284 |
| k8s | 284 | DIFF[REC] | final_revision_vs_fresh | 0.325 | 0.293 | 284 |
| k8s | 284 | DIFF[REC] | final_revision_vs_monthly | 0.039 | 0.000 | 284 |
| k8s | 284 | DIFF[REC] | one_edit_vs_fresh | 0.056 | 0.038 | 284 |
| k8s | 284 | DIFF[STRUCT-M] | final_monthly_vs_fresh | 0.197 | 0.182 | 284 |
| k8s | 284 | DIFF[STRUCT-M] | final_revision_vs_fresh | 0.204 | 0.185 | 284 |
| k8s | 284 | DIFF[STRUCT-M] | final_revision_vs_monthly | 0.017 | 0.000 | 284 |
| k8s | 284 | DIFF[STRUCT-M] | one_edit_vs_fresh | 0.032 | 0.022 | 284 |
| k8s | 284 | KARA | final_monthly_vs_fresh | 0.715 | 0.776 | 284 |
| k8s | 284 | KARA | final_revision_vs_fresh | 0.713 | 0.760 | 284 |
| k8s | 284 | KARA | final_revision_vs_monthly | 0.119 | 0.000 | 284 |
| k8s | 284 | KARA | one_edit_vs_fresh | 0.163 | 0.146 | 284 |
| cpy | 300 | DIFF[REC] | final_monthly_vs_fresh | 0.342 | 0.343 | 300 |
| cpy | 300 | DIFF[REC] | final_revision_vs_fresh | 0.366 | 0.364 | 300 |
| cpy | 300 | DIFF[REC] | final_revision_vs_monthly | 0.088 | 0.062 | 300 |
| cpy | 300 | DIFF[REC] | one_edit_vs_fresh | 0.046 | 0.034 | 300 |
| cpy | 300 | DIFF[STRUCT-M] | final_monthly_vs_fresh | 0.286 | 0.271 | 300 |
| cpy | 300 | DIFF[STRUCT-M] | final_revision_vs_fresh | 0.309 | 0.291 | 300 |
| cpy | 300 | DIFF[STRUCT-M] | final_revision_vs_monthly | 0.074 | 0.046 | 300 |
| cpy | 300 | DIFF[STRUCT-M] | one_edit_vs_fresh | 0.036 | 0.023 | 300 |
| cpy | 300 | KARA | final_monthly_vs_fresh | 0.742 | 0.790 | 300 |
| cpy | 300 | KARA | final_revision_vs_fresh | 0.743 | 0.795 | 300 |
| cpy | 300 | KARA | final_revision_vs_monthly | 0.239 | 0.191 | 300 |
| cpy | 300 | KARA | one_edit_vs_fresh | 0.132 | 0.116 | 300 |

Files: `results/kara_stability_wiki.json`, `results/kara_stability_k8s.json`, `results/kara_stability_cpy.json`

**Runbook vectors.** Not stored (disk); `experiments/runbook_vectors.py --stream S --chunker C` regenerates the bge-small vectors of a runbook deterministically from its point file (replays the chunker at the T=200 calibration, matches chunk keys, writes big-ann-benchmarks .fbin). Checked on 3 K8S documents for REC, KARA, DIFF[REC], SentCDC and 3 WIKI documents for KARA: 0 keys not reproduced (runner session 3).

## Figures


**ERROR** in this section: FileNotFoundError: [Errno 2] No such file or directory: '/Users/sonha/Documents/SonHa/ICDE27/P3_DeltaChunk/Artifact/P3_DeltaChunk/results/figures'
