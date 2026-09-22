# Emotion-Trained Vision Models Do Not Necessarily Learn EEG-Aligned Facial Dynamics

Code for extracting model representations of dynamic facial-expression videos,
computing representational dissimilarity matrices (RDMs), and running the
time-locked EEG–model RSA analyses, cluster permutation tests and low-level
controls.

**No data are included**: no EEG recordings or EEG RDMs, stimulus videos or frames,
model checkpoints or derived results. See [Expected inputs](#expected-inputs).

## Layout

```
config.py               Paths (from FER_ROOT), stimulus/emotion indices and seeds used by fer_tr/
eeg/
  compute_windowed_eeg_rdms_16win.py   Per-ROI EEG RDMs, 16 windows (18 stimuli)
models/
  prepare_stimuli_54.py, build_stimuli_h5_16frames.py
                          Stimulus frames (16 per video) for the 18- and 54-stimulus studies
  extract_model_rdms.py   Image CNNs (ResNet18/50, FaceNet, VGG16): activations, predictions, RDMs,
                          pretrained or random init (--random --seed N)
  image/
    ResNet.py, VGG16.py, FaceNet.py
                          Network definitions (ResNet18/50, VGG16 / VGG-Face, InceptionResnetV1)
    extract_image_models_16frames.py
                          FaceNet "Complete VGGFace2" and its AffectNet fine-tune
  m3dfel_extract_rdms.py  M3DFEL video model, pretrained and random init
  extract_qwen_rdms.py    Qwen3-VL video model -> per-(layer, frame) RDMs
  vlm/                    Qwen video-mode extraction, activations -> RDMs, SLURM drivers (run_qwen3vl*.sh)
  build_emotion_rdms.py, build_emotion_rdms_54.py
                          Split model RDMs into per-emotion RDMs
  train_vggface2_rgb.py, check_vggface2_layout.py
                          Face-identity pretraining on RGB VGGFace2, and a dataset check
  utils/                  Weight loading and stimulus/dataset loaders
vggface_train/          VGGFace2 identity training of ResNet18/50 and FaceNet (train.py),
                        and the training-set-size subset indexes
affectnet_3class_train/ train_affectnet.py: 3-class AffectNet (fear/happy/neutral) fine-tuning
                        of FaceNet, ResNet18, ResNet50 and VGG-Face
M3DFEL/                 M3DFEL code (CVPR 2023, Tencent TFace) with our changes for 3-class DFEW
                        (happy/fear/neutral) training and local R3D-18 weights; see its README
fer_tr/                 Shared time-locked RSA code (core.py, timelocked_rsa_per_roi.py)
analysis/
  rsa_full_grid.py        18-stimulus RSA grid: subject x ROI x layer x emotion x EEG window
  rsa_full_grid_54.py     Same for the 54-stimulus occlusion study
  rsa_per_roi.py          Per-ROI time-locked RSA through fer_tr/
  cluster_permutation_rsa_54.py
                          Layer x time cluster permutation tests with Holm correction across grids
  compute_*_rdm.py, build_control_rdms_54.py
                          Low-level control RDMs (pixel difference, motion energy, optical flow, identity)
  build_control_rsa.py, build_control_rsa_54.py, partial_rsa.py
                          Control RSA, noise-ceiling summary, partial RSA
  noise_ceiling.py        Per-window EEG noise ceiling
  _stim_loader_v2.py      Stimulus video loader for the motion controls
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

Scripts resolve paths relative to the repository root. Run them from there, or set
`FER_ROOT` (data root) and `WEIGHTS_PATH` (checkpoint folder, default `./net_weights`).
The `models/vlm/run_qwen3vl*.sh` drivers are SLURM scripts: set `#SBATCH --account`
(`YOUR_ALLOCATION`) and the environment path before submitting.

## Expected inputs

Place these at the repository root (or under `FER_ROOT`). None of them is distributed:

| Path | Content |
|---|---|
| `stim_frames/`, `stimulus_order.json` | 18-stimulus video frames (`Slide1..16.h5`) and their order |
| `stim_frames_54/` | 54-stimulus (occlusion study) frames and `stimulus_manifest.json` |
| `eeg_data/Unmasked_avg_windowed_rdms_16win_full/` | 18-stimulus per-participant, per-ROI EEG RDMs, `(6, 16, 18, 18)` |
| `eeg_data/Unmasked_avg_wholescalp_16win/` | 18-stimulus whole-scalp EEG RDMs, `(16, 18, 18)` |
| `eeg_rdms_complete/` | 54-stimulus per-ROI (`perroi/`) and whole-scalp (`wholescalp/`) EEG RDMs |
| `net_weights/` | Pretrained and fine-tuned checkpoints |
| `M3DFEL/pretrained/r3d_18-b3b3357e.pth` | Kinetics-400 R3D-18 weights (torchvision) for the M3DFEL backbone; set `M3DFEL_DIR` to use another M3DFEL copy |

Public datasets used for training: VGGFace2, AffectNet, DFEW, and ImageNet / Kinetics
pretrained weights through `torchvision`, `timm`, `facenet-pytorch` and `transformers`.

## Pipeline

0. **EEG RDMs**: `eeg/compute_windowed_eeg_rdms_16win.py` builds the 18-stimulus per-ROI RDMs
   from per-participant condition-averaged epochs (`*_avg.pkl`: ROI -> conditions x channels x time).
   The whole-scalp and 54-stimulus EEG RDMs are inputs; no script for them is included.
1. **Stimuli**: `models/prepare_stimuli_54.py`, `models/build_stimuli_h5_16frames.py`.
2. **Training** (optional; checkpoints go to `net_weights/`):
   - VGGFace2 identity pretraining: `vggface_train/build_index.py`, then `vggface_train/train.py --arch {resnet18,resnet50,facenet}`.
   - AffectNet fine-tuning: `affectnet_3class_train/train_affectnet.py --arch {facenet,resnet18,resnet50,vgg16_vggface}`.
3. **Model RDMs**: `models/extract_model_rdms.py` (image CNNs; `--random --seed N` for random init),
   `models/m3dfel_extract_rdms.py`, `models/extract_qwen_rdms.py` / `models/vlm/`.
   The FaceNet "Complete VGGFace2" and "AffectNet FT" variants use
   `models/image/extract_image_models_16frames.py`: they are facenet-pytorch's InceptionResnetV1
   (public VGGFace2 weights, no 512-d embedding layer, 1792 -> 3 head) with the stimulus
   normalization of `utils.load_data_FER.Stimuliloader`, which `extract_model_rdms.py` does not
   build. Run from `models/`:
   ```bash
   python image/extract_image_models_16frames.py --model vggface --layers weights \
       --hdf5-dir ../stim_frames --n-slides 16 --out-name vggface_affectnet_16f_pretrained_weights
   python image/extract_image_models_16frames.py --model vggface --layers weights \
       --finetuned ../net_weights/best_inceptionresnetv1_fer.pt \
       --hdf5-dir ../stim_frames --n-slides 16 --out-name vggface_affectnet_16f_finetuned_weights
   ```
   `--model vggface` is FaceNet, not VGG16. `--layers weights` hooks Conv/Linear only (133 layers,
   matching the other FaceNet variants); the random 3-unit `logits` head is dropped at the RSA step.
   Then split into per-emotion RDMs with `models/build_emotion_rdms.py` / `build_emotion_rdms_54.py`.
4. **RSA grids**: `analysis/rsa_full_grid.py` (18 stimuli), `analysis/rsa_full_grid_54.py` (54 stimuli).
5. **Statistics**: `analysis/cluster_permutation_rsa_54.py` (sign-flip cluster permutation over
   layer x time, Holm across the 7 scopes x 4 emotion conditions of each model).
6. **Controls**: `analysis/compute_*_rdm.py`, `analysis/build_control_rdms_54.py`,
   `analysis/build_control_rsa.py` (also writes the noise-ceiling summary), `analysis/build_control_rsa_54.py`,
   `analysis/partial_rsa.py`; per-window noise ceiling with `analysis/noise_ceiling.py`.
