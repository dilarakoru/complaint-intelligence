# Engineering decisions

## Problem and user journey
A reviewer needs to inspect a complaint outcome model, understand which inputs influence it, and distinguish a useful demo from trustworthy predictive evidence. The flow is: enter complaint details, inspect the classification and feature contributions, then optionally request a source-grounded excerpt.

## Decisions and trade-offs
| Decision | Reason | Trade-off |
|---|---|---|
| Deduplicate identities and normalized narratives before splitting | Repeated complaints contaminated the earlier holdout | Fewer usable examples; near-duplicates still need further work |
| Compare structured and TF-IDF logistic baselines | Establish a reproducible, inspectable reference before adding complexity | Does not capture all semantic relationships |
| Select using validation F1, report test once | Keep model selection separate from final evaluation | A 31-record test set has substantial uncertainty |
| Display exact linear contributions | Explain the deployed model faithfully | Contributions are log-odds, not causal effects or calibrated probabilities |
| Validate LLM excerpts against source text | Reject unsupported generated statements | Extractive fallback can be less informative |
| Keep optional LLM loading separate | The core product remains runnable without downloads | First LLM request can be slow on CPU |

## Evidence and next experiments
The local result is recorded in `evaluation-local.json`; the README explains the small sample and invalid earlier score. The public synthetic dataset proves reproducibility, not real-world quality. Next: acquire an authorized representative sample; use temporal or company-aware holdouts where appropriate; measure subgroup errors and uncertainty; evaluate calibration before exposing scores as probabilities. Add a small human-rated excerpt set before claiming summarization quality.

## Three-minute walkthrough
1. Train on the included synthetic dataset and start the API using the README.
2. Submit a complaint and inspect the output and feature contributions.
3. Explain why duplicate removal precedes splitting and why the earlier score is invalid.
4. Enable the optional model only when installed; explain the validated excerpt and fallback.
5. Run the tests and distinguish pipeline checks from evidence of generalization.
