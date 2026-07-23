# Reproduction

```bash
export FER_ROOT=/path/to/data
pip install -r requirements.txt
```

Four stages. **Stages B–D need only the RDM `.npy` files** — if you already have
`eeg_data/Unmasked_avg_windowed_rdms_16win_full/` and the canonical model RDMs, skip
straight to B.

## Stage A — build the matched RDMs (GPU for the models)

**EEG — 16 non-overlapping windows tiling the full epoch**
```bash
python eeg/compute_windowed_eeg_rdms_16win.py \
    --input-dir  $FER_ROOT/eeg_data/Unmasked_avg \
    --output-dir $FER_ROOT/eeg_data/Unmasked_avg_windowed_rdms_16win_full \
    --metric correlation --n-windows 16
# -> <subject>_windowed_rdms.npy  (6 ROI, 16, 18, 18)  + roi_order.npy, window_bounds.npy
```

**Stimulus frames — 16 per video, on the grid every other model uses**
```bash
python models/build_stimuli_h5_16frames.py \
    --video-dir $FER_ROOT/Stims_Videos/Stims_Videos --out-dir $FER_ROOT/hdf5_16frames
```

**Image models — ResNet18 / VGGFace (fine-tuned, pretrained, and random)**
```bash
python models/image/extract_image_models_16frames.py --model resnet18 --finetuned <ckpt> \
    --hdf5-dir $FER_ROOT/hdf5_16frames --n-slides 16 --out-name resnet18_affectnet_16f
python models/image/extract_image_models_16frames.py --model resnet18 \
    --hdf5-dir $FER_ROOT/hdf5_16frames --n-slides 16 --out-name resnet18_affectnet_16f_pretrained
python models/image/extract_image_models_16frames.py --model resnet18 --random \
    --hdf5-dir $FER_ROOT/hdf5_16frames --n-slides 16 --out-name resnet18_affectnet_16f_random
# (repeat with --model vggface -> vggface_affectnet_16f[_pretrained|_random])
```

**CNN2D / CNN3D-CAER — trained and untrained**
```bash
# activations from the fixed stimulus videos (add --untrained for the random-init control)
python models/video_cnn/extract_timewise_activations.py --input $FER_ROOT/Stims_Videos/Stims_Videos \
    --checkpoint_2d <cnn2d.pt> --checkpoint_3d <cnn3d.pt> --out_dir <act_dir>/trained --clip_len 16
python models/video_cnn/extract_timewise_activations.py --input $FER_ROOT/Stims_Videos/Stims_Videos \
    --untrained --out_dir <act_dir>/untrained --clip_len 16
# activations -> RDMs (16 timepoints), then install as cnn{2,3}d_caer[_random]_canon.npy
python models/video_cnn/compute_rdms_from_activations.py --root <act_dir>/trained \
    --models cnn2d cnn3d --timepoints 16 --metric correlation --out_dir <rdm_dir>
```

**DINOv2-temporal — per-frame RDMs from the trained checkpoint**
```bash
python models/dino/extract_dino_per_frame.py --checkpoint $FER_ROOT/DINOv2/best_model.pt \
    --out $FER_ROOT/analysis/model_rdms_canonical/dinov2_temporal_canon.npy
```

**Qwen2.5-VL — native video mode** (16 canonical frames duplicated → 16 temporal groups)
```bash
python models/vlm/vlm_extract_video_mode.py \
    --metadata_csv models/vlm/stimuli_metadata_canon.csv \
    --output_dir $FER_ROOT/VLM/out_qwen_video --model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --nframes 16 --frame-list
# then build + orient the RDM (parent project's compute_rdms_from_vlm_activations.py)
```

Place every model RDM in `$FER_ROOT/analysis/model_rdms_canonical/` under the tags in
`config.MODELS`, oriented `(n_layer, 16, 18, 18)` in the canonical stimulus order.

## Stage B — RSA tables
```bash
python analysis/rsa_per_roi.py                 # -> rsa_per_roi.csv            (all stimuli)
python analysis/rsa_per_roi.py --per-emotion   # -> rsa_per_roi_per_emotion.csv (+ BH-FDR)
```
Each cell: subject-mean time-locked RSA, 95% bootstrap CI, sign-flip permutation p
(per-emotion also q_fdr / sig_fdr).

## Stage C — noise ceiling
```bash
python analysis/noise_ceiling.py    # -> noise_ceiling_16win.{csv,json,npz}
```

## Stage D — figures
```bash
python figures/plot_timecourses.py                            # rsa_timecourses.{pdf,png}
python figures/plot_timecourses.py --selection pertimepoint   # rsa_timecourses_pertimepoint.*
python figures/plot_timecourses.py --emotion fear             # rsa_timecourses_fear.*
python figures/plot_emotion_heatmaps.py                       # rsa_heatmap_<emotion>.*
python figures/plot_rsa_vs_ceiling.py                         # rsa_vs_ceiling_<model>.*
```

## Estimator, in one place

`fer_tr/core.py`:
- `timelocked_rsa` — one LOSO-selected layer, fixed across the epoch ("max 1 layer").
- `timecourse` / `timecourse_pertimepoint` — the same, traced over time; the latter
  re-selects the layer at every timepoint ("max over all layers") and is **not** on a
  comparable scale to the fixed-layer version.
- `noise_ceiling` — Nili et al. (2014) upper/lower bounds.
- `bootstrap_ci`, `signflip_p`, `bh_fdr` — inference. Seed = `config.SEED`.
