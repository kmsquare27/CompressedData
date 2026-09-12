# Validation record and remaining RunPod checks

This package was authored against your uploaded pipeline files and accepted-decision markdown. No real finalized dataset, model weights or CUDA GPU were available in the authoring environment.

## Checked locally

- All delivered Python source files parse using Python 3.11 grammar.
- `00_runpod_setup.sh` passes `bash -n` syntax validation.
- Twelve CPU unit/integration tests pass. They cover byte-preserving portable exports; CSV/hash violations collected before bundle creation; explicit short-population failure; deterministic complete page exposure; final partial accumulation-window normalization; distinct example/token objectives; assistant masking including merged boundary whitespace; vision-module exclusion from LoRA targets; path containment; JSON missing-number handling; exact paired saving calculations; PNG/PDF report generation; and exclusion of inconsistent counters from comparisons.
- Reporter integration fixtures are synthetic and temporary. No fixture measurements are included as thesis results.
- Every command-line script supports `--help` without loading model weights. GPU-only dependencies are imported after parsing in the training/doctor/preparation paths.

Run the same CPU checks after installing the RunPod environment:

```bash
python -m unittest discover -s tests -v
```

## Must still be checked on your RunPod

1. Installation and `pip check` in the documented Python 3.11 or 3.12/CUDA environment.
2. Real LoRA/BF16 backward with `01_check_environment.py`.
3. Download and immutable snapshot hash verification.
4. Actual Qwen processor tokenization/offsets/image grids over your real paired pages in preparation. The local masking tests establish the rules; they do not prove a downloaded tokenizer's behavior.
5. Full-model smoke for all three arms, including finite loss/gradients and a changed saved adapter.
6. The full pilot over every page, because worst-size smoke is not exhaustive memory validation.
7. A controlled interruption/resume trial if you intend to rely on resume for long runs. Adapter/optimizer/scheduler/RNG restoration is implemented, but was not exercised against a real Qwen model here. Keep the primary computational comparison as fresh uninterrupted runs.

The smoke verifier checks learned weights and saved-file integrity. It is not a model-quality evaluation. All memory, timing, energy and learning outcomes remain unmeasured until you execute the GPU workflow. No test-set evaluation or inference is performed by this package.

## Minimal 7B update

15 CPU tests passed after the update. New predecessor-integrity tests reject wrong arms, overlapping page IDs, changed model revisions/rank and changed adapter bytes. Python 3.11 syntax and bash syntax pass. No CUDA/model/processor training executed here; real BF16 LoRA smoke and continuation reload remain GPU validation steps.
