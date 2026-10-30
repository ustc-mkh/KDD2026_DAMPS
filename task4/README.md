# Task 4: SMORE baseline and DAMPS

Run one seed (999) for each of five datasets and five variants, 25 distinct configurations. `plan.json` is fixed before training; no parameter search or selection on test data.

## Sources and integration

- Team base: ustc-mkh/KDD2026_DAMPS `cafcc930546799f5ca393ad515fbd5c7b6d85e6d`.
- Official SMORE: kennethorq/SMORE `5bccf2a4bbc733bf9e300aab68b3efd1bc004c75`.
- MMRec: enoche/MMRec `c68ee7c9b0de1ca7c5150258322edb4994be29d9`; its SMORE model is identical to the official file.
- Author archive: https://doi.org/10.5281/zenodo.18028194 (resolves to record 18028195).

`src/models/smore.py` preserves official graph construction, spectrum convolution, propagation, gates and losses. Changes: explicit graph cache directory, portable device loads, and optional calibration between projection and spectrum convolution. Raw mode does not instantiate a calibrator. The existing `models.damps.DAMPS` and MGCN files remain unchanged. The reusable `DAMPSProjected(projected_image_init, projected_text_init, variant)` adapter accepts `[items, embedding_dim]` matrices and returns two equally shaped calibrated matrices. It wraps the existing Task 1 shared DAMPS implementation (an unchanged snapshot in `common/damps_task1.py`) and uses its projection-level API with boolean APC/AVRF/IMCF switches and explicit initialization. Origin: the team COMP5331 project, commit `9395c29b4a1cb05996f55a143ddea7a1f71929d0`, `src/mmrecsys/nn/damps.py`, implementation version 3. No team files are modified. The original team KDD checkout has unrelated MGCN changes; they are left untouched.

## Published implementation discrepancies

The public DAMPS forward ignores its arguments' values and learns an additional embedding/projection pair. Directly calling it after SMORE projection would disconnect SMORE's projection from the loss. The adapter uses the supplied projected features instead, and initializes priors from those same projections. This is a declared integration change, not bit-for-bit reproduction of the author's complete MGCN architecture.

For reproducibility, the spectral operations follow released code: AVRF uses trainable logit-transformed, standardized ratios as direct masks (no sigmoid in forward), rather than the bounded raw ratio described in Eq. 16. Phase signs follow both released code and Eq. 8. The one-tap IMCF ratio `|x conj(y)|^2/(|x|^2 |y|^2+eps)` is algebraically almost one for nonzero spectra; tests record this degeneracy. These issues are not silently corrected or tuned to match reported metrics. Conclusions must distinguish the released-code adapter from a future paper-corrected implementation.

Ablations: `no_apc` bypasses both prior and residual rotations; `no_avrf` removes the variance branch and gives the coherence branch weight one; `no_imcf` removes coherence and gives variance weight one. Disabled parameters are not registered. Full model uses the author's softmax mixture. Ablation semantics are explicit because author ablation code is not provided in the base repository.

## Fixed protocol

`configs.json` uses the official recommended Baby/Sports/Clothing parameters. Electronics/MicroLens have no released dataset-specific SMORE recommendation; choose the first value of each official search list (3 UI layers, image/text k=10, regularization 1e-5, dropout=0, contrastive weight=.01). These two configurations are declared fallbacks, not claimed as the paper's undisclosed parameters. All variants share the same backbone settings, initial backbone RNG order, seed and dataset hashes.

Preserve downloaded `x_label` splits (0/1/2) and audit counts, feature sizes, finite values and SHA256 hashes. The author split code reserves one validation and one test item for users with fewer than ten interactions; otherwise floor(0.2*n) held-out items are divided between validation and test. Thus these are nominal per-user 80/10/10 splits with a low-count exception, not exact global 80/10/10. Measured train/valid/test counts are preserved in data_inventory.json. Full ranking masks training interactions as in MMRec. Metrics: Recall/NDCG at 10/20. Validation Recall@20 selects the checkpoint, with patience 20 and maximum 1000 epochs; test is evaluated once after restoring the selected checkpoint. `run_task4.py` uses shared trainer operations but bypasses the original grid-search quick_start, which picks a configuration using test scores and saves the final rather than best epoch state.

Environment: isolated Python 3.10.20 (available interpreter; differs from paper's 3.8.20), PyTorch 2.4.1+cu124, torchvision .19.1, NumPy 1.24.4 (paper 1.22.4), SciPy 1.10.1. Full package freeze and GPU metadata are saved with runtime audit. No existing environment is modified. GPU is A100 80GB rather than paper RTX 4090; timing is not directly comparable.

## Running

From repository root, with the isolated Python environment:

```sh
python task4/test_adapter.py
python src/run_task4.py --dataset baby --variant raw --gpu 0 --data-root /absolute/data --output /absolute/runs/baby__raw__seed999
python task4/dispatch.py --root /absolute/task-root --gpus 4,5
python task4/summarize.py --root /absolute/task-root
```

Queue root contains `data/{baby,sports,clothing,elec,microlens}` and `runs/`. Launch with nohup and redirected input/output. A lock prevents duplicate queues. Before each start: no compute processes on selected GPU, <=128 MiB allocation, <=5% utilization, CPU load <50% logical core count, available RAM >64 GiB, disk >100 GiB. Two GPUs maximum, four CPU threads per job, nice 10. Unavailable resources cause waiting; missing data remains pending. A failed job stops new dispatch and is recorded for review. Completed results are skipped on restart. Existing tasks are never terminated or reconfigured.

Outputs per run: `metadata.json`, data hash manifest, per-epoch validation/time/GPU memory JSONL, `progress.json`, validation-selected `best.pt`, `result.json`, console log. Summary reads completed result files only and produces CSV plus an ablation plot. Checkpoints/data/runtime output stay outside Git. Do not publish preliminary results as completed experiments. PR target is `ustc-mkh/KDD2026_DAMPS`, after experiments and analysis are complete.
