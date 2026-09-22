# Emotion-Trained Vision Models Do Not Necessarily Learn EEG-Aligned Facial Dynamics

Code and manuscript source for the submission. The repository contains the
paper and all scripts used to extract model representations, compute
representational dissimilarity matrices (RDMs), and run the time-locked
EEG–model RSA analyses and statistics reported in the paper.

**No data are included**: no EEG recordings or EEG RDMs, stimulus videos or frames,
model checkpoints or derived results. See [Expected inputs](#expected-inputs).

## Layout

```
paper/                  LaTeX source (main.tex, bibliography, style, figures, tables)
config.py               Paths (from FER_ROOT), stimulus/emotion indices and seeds used by fer_tr/
eeg/                    compute_windowed_eeg_rdms_16win.py: per-ROI EEG RDMs, 16 windows (18 stimuli)
models/                 Stimulus preparation, activation extraction and model RDMs
  extract_model_rdms.py   Image CNNs (ResNet18/50, FaceNet, VGG16/-BN): activations, predictions, RDMs
  image/extract_image_models_16frames.py
                          FaceNet "Complete VGGFace2" and its AffectNet fine-tune (Table 1)
  extract_qwen_rdms.py    Qwen3-VL video model -> per-(layer, frame) RDMs
  m3dfel_extract_rdms.py  M3DFEL video model, pretrained and random init
  image/ video_cnn/ vlm/ dino/   Model-specific extraction code
  utils/                  Weight loading and dataset helpers
  third_party/M3DFEL/     Upstream M3DFEL code (CVPR 2023, TFace), see its README
  train_vggface2_rgb.py   Face-identity pretraining
  run_*.sh                SLURM drivers
vggface_train/          FaceNet VGGFace2 training, including training-set-size subsets and repeats
affectnet_3class_train/ AffectNet fine-tuning of the face-pretrained models
fer_tr/                 Shared RSA / statistics code for the time-resolved pipeline
analysis/
  rsa_full_grid.py        18-stimulus RSA grid: subject x ROI x layer x emotion x EEG window
  rsa_full_grid_54.py     Same for the 54-stimulus occlusion study
  cluster_permutation_rsa_54.py, submit_cluster_permutation.sh
                          Layer x time cluster permutation tests
  contrasts/              Paired training-regime contrasts (frozen plan in PLAN.md, hashed in PLAN.sha256)
  seed_variability/       Multi-seed random-initialization baselines
  layer_matched/          Table 1 with layer-count-matched FaceNet variants
  partial_rsa.py, compute_*_rdm.py, build_control_*.py
                          Low-level controls (pixel, motion energy, optical flow, identity)
  noise_ceiling.py        Per-window EEG noise ceiling (the paper's summary is written by build_control_rsa.py)
  probe/                  Cross-validated emotion probes vs. EEG alignment
  build_*/export_*/plot_*.py
                          Tables and figures in the paper
requirements.txt
LICENSE
```

## Environment

Python 3.12 with the packages in `requirements.txt`. All model extractions were
run on a CUDA GPU. **Random-initialization baselines must be run on a GPU**:
weights are randomized after the model is moved to the device, so the same seed
gives different weights on CPU.

```bash
python -m venv $HOME/envs/fer && source $HOME/envs/fer/bin/activate
pip install -r requirements.txt
```

The shell drivers are SLURM scripts. Before submitting them, set `#SBATCH --account`
(`YOUR_ALLOCATION`) and the environment path. Scripts resolve paths relative to the
repository root. Run them from there, or set `FER_ROOT` / `WEIGHTS_PATH` where the
script supports it.

## Expected inputs

Place these at the repository root. None of them is distributed:

| Path | Content |
|---|---|
| `stim_frames/`, `stimulus_order.json` | 18-stimulus video frames and their order |
| `stim_frames_54/` | 54-stimulus (occlusion study) frames and `stimulus_manifest.json` |
| `eeg_data/Unmasked_avg_windowed_rdms_16win_full/` | 18-stimulus per-participant, per-ROI EEG RDMs, `(6, 16, 18, 18)` |
| `eeg_data/Unmasked_avg_wholescalp_16win/` | 18-stimulus whole-scalp EEG RDMs, `(16, 18, 18)` |
| `eeg_rdms_complete/` | 54-stimulus per-ROI (`perroi/`) and whole-scalp (`wholescalp/`) EEG RDMs |
| `net_weights/` | Pretrained and fine-tuned checkpoints |

Public datasets used for training: VGGFace2, AffectNet, DFEW, and ImageNet / Kinetics
pretrained weights through `torchvision`, `timm`, `facenet-pytorch` and `transformers`.

## Pipeline

0. **EEG RDMs**: `eeg/compute_windowed_eeg_rdms_16win.py` builds the 18-stimulus per-ROI RDMs
   from per-participant condition-averaged epochs (`*_avg.pkl`: ROI -> conditions x channels x time).
   The whole-scalp and 54-stimulus EEG RDMs are provided as inputs; no script for them is included.
1. **Stimuli**: `models/prepare_stimuli_54.py`, `models/build_stimuli_h5_16frames.py`.
2. **Model RDMs**: `models/extract_model_rdms.py`, `models/extract_qwen_rdms.py`,
   `models/m3dfel_extract_rdms.py` (drivers: `models/run_*.sh`).
   The two FaceNet rows "Complete VGGFace2" and "AffectNet FT" use a separate extractor,
   `models/image/extract_image_models_16frames.py`. They are facenet-pytorch's InceptionResnetV1
   (public VGGFace2 weights, no 512-d embedding layer, 1792 -> 3 head) with the stimulus
   normalization of `utils.load_data_FER.Stimuliloader`, which `extract_model_rdms.py` does not
   build. Run from `models/` (driver: `analysis/layer_matched/run_layer_matched.sh`):
   ```bash
   python image/extract_image_models_16frames.py --model vggface --layers weights \
       --hdf5-dir ../stim_frames --n-slides 16 \
       --out-dir ../analysis/layer_matched/rdms --out-name vggface_affectnet_16f_pretrained_weights
   python image/extract_image_models_16frames.py --model vggface --layers weights \
       --finetuned ../net_weights/best_inceptionresnetv1_fer.pt \
       --hdf5-dir ../stim_frames --n-slides 16 \
       --out-dir ../analysis/layer_matched/rdms --out-name vggface_affectnet_16f_finetuned_weights
   ```
   `--model vggface` is FaceNet, not VGG16. `--layers weights` hooks Conv/Linear only (133 layers,
   matching the other FaceNet variants); the random 3-unit `logits` head is dropped at the RSA step.
3. **RSA grids**: `analysis/rsa_full_grid.py` (18 stimuli), `analysis/rsa_full_grid_54.py` (54 stimuli).
4. **Statistics**: cluster permutation (`analysis/cluster_permutation_rsa_54.py`),
   paired contrasts (`analysis/contrasts/run_contrasts.py`), seed baselines
   (`analysis/seed_variability/`), layer-matched Table 1 (`analysis/layer_matched/`).
5. **Controls and probes**: `analysis/partial_rsa.py`, `analysis/probe/`.
6. **Tables and figures**: `analysis/export_*.py`, `analysis/plot_*.py`.

## Building the paper

```bash
cd paper && latexmk -pdf main.tex
```
