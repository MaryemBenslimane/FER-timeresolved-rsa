# Time-resolved, time-locked EEG↔model RSA

Time-resolved Representational Similarity Analysis between human EEG responses to
dynamic facial expressions and vision / vision-language models, on a **matched 16-point
time axis** with **no ROI averaging** and a **strictly time-locked** estimator.

18 stimuli (6 actors × 3 emotions: *fear / happy / neutral*), 26 subjects, 6 bilateral
EEG ROIs. Both the EEG and every model are reduced to one 18×18 representational
dissimilarity matrix (RDM) per timepoint; model frame *k* is compared with EEG window
*k* (never searched over time), and the best model layer is chosen by
leave-one-subject-out so the estimate is not circular.

## Why this repo exists (design decisions, all enforced in `fer_tr/core.py`)

- **Matched 16-point time axis.** EEG is re-windowed into **16 non-overlapping ~62.5 ms
  windows** tiling the full 1 s epoch; every model is resampled to **16 frames** on the
  *same* frame grid, so the diagonal (window *k* ↔ frame *k*) is well defined.
- **Time-locked, not peak-searched.** The estimator correlates matched timepoints only.
  It does **not** take an argmax over (window × layer × time) — that unconstrained search
  answers a weaker question and is prone to selection bias.
- **No ROI averaging.** The 6 ROIs (Ant-L/R, Post-L/R, Temp-L/R) are analysed separately.
- **Leave-one-subject-out layer selection.** Only the layer is selected, from the other
  25 subjects; the held-out subject is scored there (no double-dipping).
- **Canonical stimulus order.** Emotion-major, numeric actors
  (`fear=0–5, happy=6–11, neutral=12–17`; actors `01,02,04,07,08,11`), matching the EEG.

## Layout

```
config.py                 paths ($FER_ROOT), constants, model registry, canonical order
fer_tr/core.py            the single source of truth for the estimator + statistics
                          (upper-tri, rank-normalise, time-locked RSA, per-timepoint
                          variant, noise ceiling, bootstrap, sign-flip perm, BH-FDR)
analysis/
  rsa_per_roi.py          time-locked RSA per model × ROI  [--per-emotion]
  noise_ceiling.py        upper/lower EEG noise ceiling (Nili et al. 2014)
figures/
  plot_timecourses.py     per-model RSA time courses  [--selection fixed|pertimepoint] [--emotion]
  plot_emotion_heatmaps.py  model × ROI heatmap, one per emotion
  plot_rsa_vs_ceiling.py    trained vs untrained overlaid on the ceiling band, per ROI
eeg/
  compute_windowed_eeg_rdms_16win.py    EEG RDMs, 16 non-overlapping windows
models/                   Stage-A RDM extraction (each family self-contained)
  build_stimuli_h5_16frames.py          16-frame stimulus slides from the videos
  image/    extract_image_models_16frames.py (+ FaceNet.py)   ResNet18 / VGGFace
  utils/    load_data_FER.py                                  (Stimuliloader for image/)
  video_cnn/ extract_timewise_activations.py, compute_rdms_from_activations.py,
             models.py                                        CNN2D / CNN3D (CAER)
  dino/     extract_dino_per_frame.py (+ rdms_compute.py)     DINOv2-temporal
  vlm/      vlm_extract_video_mode.py, stimuli_metadata_canon.csv   Qwen2.5-VL
docs/REPRODUCE.md         step-by-step
```

## Install & data

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export FER_ROOT=/path/to/data          # default: /home/maryem/scratch/FER
```

Expected under `$FER_ROOT` (see `config.py`):

| Path | Contents |
|------|----------|
| `eeg_data/Unmasked_avg_windowed_rdms_16win_full/` | per-subject EEG RDMs `(6, 16, 18, 18)` + `roi_order.npy` |
| `analysis/model_rdms_canonical/` | per-model RDMs `(n_layer, 16, 18, 18)`, canonical order |

## Run (analysis + figures only need the RDM `.npy` files)

```bash
python analysis/rsa_per_roi.py                       # main table (all stimuli)
python analysis/rsa_per_roi.py --per-emotion         # + fear/happy/neutral
python analysis/noise_ceiling.py                     # EEG noise ceiling
python figures/plot_timecourses.py                   # time courses (max 1 layer)
python figures/plot_timecourses.py --selection pertimepoint
python figures/plot_emotion_heatmaps.py
python figures/plot_rsa_vs_ceiling.py                # RSA vs ceiling, trained vs untrained
```

## A necessary caveat on the results

The upper noise ceiling of these 16-window EEG RDMs is ≈ **0.19**, and the **lower
bound is ≈ 0** (see `analysis/noise_ceiling.py`). Note that `1/√26 ≈ 0.196`: an upper
ceiling at that value with a near-zero lower bound indicates the per-subject RDMs carry
little cross-subject-reliable structure at this window resolution. The time-locked model
RSAs sit near the noise floor accordingly. `plot_rsa_vs_ceiling.py` makes this explicit,
and it should be read before drawing conclusions about any individual model.

## Notes

- **Stage A (extraction).** Each model family under `models/` is vendored with the
  local modules it imports (`FaceNet.py`, `models.py`, `rdms_compute.py`,
  `utils/load_data_FER.py`), so the extractors run standalone given the raw stimuli,
  checkpoints, and (for DINOv2/VLM) the torch.hub / HF weights. The analysis/figure
  stage needs only the resulting RDM `.npy` files.
- All randomness uses a fixed seed (`config.SEED`).

## License

MIT — see [LICENSE](LICENSE) (fill in year/authors before release).
