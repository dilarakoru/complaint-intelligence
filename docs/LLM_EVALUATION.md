# Local LLM validation

Qwen2.5-0.5B-Instruct loaded from the existing local cache and generated output on CPU. No remote inference API was used.
A free-summary prompt produced unsupported facts on a synthetic fee-dispute example. The current application therefore validates an extractive output against the source and falls back deterministically when it does not match.
The regression test supplies an invented refund statement and verifies that it cannot be returned as the summary.
This is a small qualitative check, not a comprehensive hallucination benchmark. The local LLM output remains limited to source excerpts; the classifier is a separate, evaluated classical model.
