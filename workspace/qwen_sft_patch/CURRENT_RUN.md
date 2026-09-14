# Current two-arm SFT release

## Active decisions

- Independently initialize original and verified from the locked
  Qwen/Qwen2.5-VL-7B-Instruct base.
- Do not initialize from pilot adapters or train a naive arm.
- Source dataset: /workspace/html_compression_linux_v2.
- Stage-one page count comes from the audited export and preparation.
  The number 2250 in a configuration filename does not set that count.
- Later continue each corresponding stage-one adapter on disjoint new pages.
- Keep existing code, datasets, model files and pilot results preserved.

## Active layout

Scripts 00 through 18, sft_core.py and telemetry.py are at the project root.
There is no required compute_tools subdirectory in this release.

Deploy the complete finalized ZIP to a new directory, for example:
  /workspace/sft_ready_release/qwen_sft_patch

Reuse:
  /workspace/sft-venv
  /workspace/models/qwen7b
  /workspace/sft_artifacts/qwen7b_uploaded_model_lock.json

Do not rerun old overlay/repair installers on this finalized tree.

## Configurations

Stage one: configs/stage1_2250_12k.json
Stage two: configs/stage2_7500_12k.json

Both use a 12288 expanded sequence cap, subject to full-dataset preflight.
They retain three checkpoints per session and save at epoch ends as well as
every 50/100 updates respectively. Final adapters are saved separately.

## Before training

1. Verify the shared-volume mount, available storage and locked model.
2. Audit Step 08 records, referenced files, acceptance policy and excluded IDs.
3. Export exact audited IDs and preserve valid original fallbacks.
4. Prepare original/verified together with no silent truncation.
5. Inspect exclusions, actual processed lengths and image legibility.
6. Run the matched GPU smoke tests.
7. Check actual interruption recovery and parent-adapter reload on the GPU.

Strict pixel identity is the export default. The validated policy accepts
Step 08 status=ok without adding that stricter pixel rule. Choose based on
the saved Linux experiment, not a desired page count.

## Initialization

Fresh stage one:
  run_all_arms.py ... --arms original verified

No --resume or --init-from-runs argument belongs on the fresh stage-one command.

New-data continuation:
  run_all_arms.py ... --arms original verified --init-from-runs STAGE_ONE_RUN_ROOT

Parent smoke:
  use the same parent argument with --smoke and a separate diagnostic output.

Actual continuation uses the stage-one parent again, never the smoke output.
It starts a fresh optimizer and schedule.

Interruption recovery:
  14_train_sft.py ... --arm ARM --resume ACTUAL_CHECKPOINT --out NEW_SESSION

Use the same stage data/config/code for resume. A final-update checkpoint can
recover final saving without repeating training updates.

## Verification and reporting

Final adapter:
  python 16_verify_sft_checkpoint.py --run ARM_SESSION

Intermediate checkpoint files:
  python 16_verify_sft_checkpoint.py --checkpoint CHECKPOINT_DIRECTORY

Training comparison:
  python 15_report_sft_compute.py --runs RUN_ROOT --out NEW_REPORT --arms original verified

Inference comparison, when evaluation is ready:
  python 18_inference_profile.py --compare ORIGINAL_PROFILE VERIFIED_PROFILE --out NEW_REPORT

Inference generation still requires a completed adapter run; this patch does
not add base-model evaluation or reconstruction-quality scoring.

## Validation boundary

The finalizer executes focused CPU reporting and inference-contract tests in
a temporary copy before applying its changes. Synthetic fixture values are
not experimental results. Real dataset compatibility, CUDA execution,
processor behavior and checkpoint restoration remain pod checks.

Historical test records in other documents do not certify this release.
