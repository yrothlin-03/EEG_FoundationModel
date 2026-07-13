for more informations about the data and the method that are used -> https://github.com/yrothlin-03/EEG_preprocessing_TELECOM_PARIS.git


1. **Latent space quality assessment**
   Define and implement metrics/benchmarks to evaluate the quality of the learned latent space (e.g., codebook utilization, reconstruction fidelity, probing against known physiological measures, cross-subject/session generalization).

2. **Pretraining objective comparison**
   Compare different self-supervised pretraining strategies — reconstruction-based (SSL), contrastive, masked modeling, jepa — to identify which objective(s) yield the most informative and generalizable representations.

3. **Benchmarking against existing foundation models**
   Evaluate the proposed model against publicly released EEG foundation models (using their official pretrained checkpoints) on shared downstream tasks, to establish a fair external point of comparison.

4. **Stochastic modeling for autoregressive generation**
   Investigate stochastic latent/state modeling approaches (e.g., Deep Kalman Filters, stochastic Mamba, Neural SDEs) for autoregressive generation, applied to the imagined speech decoding task.



## 29/06/2026 - 05/07/2026

- Added new datasets and pretraining loaders to `EEG_preprocessing_TELECOM_PARIS`.
- Implemented the new model architecture and the pretraining pipeline; ran a first set of pretraining configs.

## 06/07/2026 - 12/07/2026

- Added a supervised contrastive pretraining pipeline (did not yield usable results) and a downstream pipeline.
- Ran ablation studies on the channel adaptor (spatial information), a codebook analysis, and an EEGNet benchmark.
- Selected 14 papers to read and collected the code of 6 baseline models (CBraMod, EEGMamba, EEGPT, LaBraM, LUNA, REVE).
- Started investigating Neural SDEs.
