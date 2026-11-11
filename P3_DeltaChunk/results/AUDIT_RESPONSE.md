# DeltaChunk experiment audit

Two independent read-only checks compared the experiment logs, draft, and result files on Nov 10, 2026. Tier A finished at 03:06 and the scheduled Tier B runs finished at 06:58 (+07).

| Check | Evidence and disposition |
|---|---|
| Common-size cost | `e1_matched_size.csv` supports ranges of 28.139–30.946%, 14.354–17.237%, and 6.799–9.209% at 133, 136, and 137 tokens. Seven methods bracket the WIKI point, eight bracket each Git point. The paper describes the observed spread and gives separate intervals at nominal T=128. |
| Theorem 1 denominator | The covered-only mean insertions and bounds are 1.45/7.48, 1.78/7.97, and 1.85/5.87. These were recomputed from the Parquet rows with `guard == False`. The manuscript now distinguishes these from the all-edit means. |
| QA reporting | The primary and stale answer files contained the same 300 closed-book rows. T3 and the summary now append only the stale REC arm. Verified closed-book counts are 69 CPY, 62 K8S, and 169 WIKI. Paired estimates already used the correct denominator. |
| Freshness | REC evidence recall is 484/900 maintained versus 282/900 stale. In the 392-question cohort whose evidence is absent in the old snapshot, it is 206/392 versus 3/392. E9 contains is 84/300 maintained versus 66/300 stale. The paper identifies the selected cohort and uses clustered intervals. |
| Late chunking | WINDOW-4 and EXACT each retrieved the evidence for 205/300 questions. WINDOW-4 used 69.0% of forward tokens and 57.6% of writes. The experiment did not establish equivalence under the prespecified rule. The CTX timing stopped after one full repetition and part of a second. |
| Index and timing | Maintained and independently rebuilt indexes both use exact neighbors as the reference. Micro-batching includes cancellation of superseded pending inserts. Derived regime costs are shown separately from measured timing. |
| Retrieval comparisons | The experiments calibrated nominal sizes per stream. Final-snapshot lengths and the text contributing to the fixed distractor-chunk budget still differ by method. The paper describes these as comparisons of realized configurations. |
| Reproduction | The Git SHAs, 476 release timestamps, Steadlith revision, realized QA sizes, and local server setup are recorded. The existing pinned clones passed `get_data.sh git`. A clean-machine run and public EAB package are pending. The release needs the required guard scripts and a decision on packaging the LLM response cache. |
