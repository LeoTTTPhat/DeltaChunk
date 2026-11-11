# Author audit sheet for the generated questions

`audit_sheet.csv` lists 100 retrieval questions in a stratified, salted-hash sample (17 edit-proximal and 17 other WIKI questions, 33 K8S and 33 CPY questions). Each row gives the three-unit passage shown to the question generator, the question and its reference answer.

Fill each `human_*` column with `yes` or `no`, judging from the passage only:

- `human_answerable`: the passage answers the question.
- `human_answer_correct`: the reference answer is a correct answer according to the passage.
- `human_self_contained`: the question can be understood without the passage.
- `human_unambiguous`: the question has one clear answer.

The sheet omits the second-model verdicts on purpose so that the audit stays blind. They are in `qcheck.csv` (same `qid`). No human audit has been performed at the time of writing.

Source text: Simple English Wikipedia (CC BY-SA 4.0), Kubernetes documentation (CC BY 4.0) and CPython documentation (PSF License); see `DATA_STATEMENT.md` in the release.
