# UCF50 Video Action Classification

This project classifies short video clips into the 50 action classes of
[UCF50](https://www.crcv.ucf.edu/data/UCF50.php) (Basketball, Biking, PlayingGuitar, ...).
Two models are supported, and both start from **ImageNet (object recognition) weights only**.
No video pre-training (Kinetics etc.) is used.

| `--model_type` | Architecture | Idea |
|---|---|---|
| `lrcn` (default) | **LRCN**: ImageNet CNN frame encoder (ResNet-50 / **ConvNeXt-Tiny**) → bidirectional LSTM → temporal mean-pool → linear | A 2D CNN encodes each frame and an RNN models the order of the frames |
| `3dcnn` | **R3D-18** with weights *inflated* from ImageNet ResNet-18 (I3D trick) | 3D convolutions learn motion directly |

The codebase was inherited from an HMDB51 project in a broken state. Everything that was
found and fixed is listed in [What was wrong with the original code](#what-was-wrong-with-the-original-code)
below, and in more detail in [`report.md`](report.md).

---

## Results

Test numbers are on **4 held-out UCF50 groups (1081 clips)**. No actor/scene in the test set
appears in training, and exact duplicate videos were removed. Checkpoints and settings are
chosen on validation only; the test split is never used for any decision.

| Run (`runs/<name>`) | Model | Trained on | Best val acc | **Test acc** (1 view) | Test acc (6-view TTA) | W&B |
|---|---|---|---|---|---|---|
| `lrcn-r50-bilstm` | LRCN, ResNet-50 + BiLSTM | train | 93.32% | 90.66% | 91.21% | [fkpls76b](https://wandb.ai/euskon-johns-hopkins-university/ucf50-video-classification/runs/fkpls76b) |
| `r3d18-inflated` | R3D-18, ImageNet-inflated | train | 80.07% | 73.91% | 74.93% | [51suf9ug](https://wandb.ai/euskon-johns-hopkins-university/ucf50-video-classification/runs/51suf9ug) |
| `lrcn-convnext-t-bilstm` | LRCN, ConvNeXt-Tiny + BiLSTM | train | 95.92% | 92.23% | 92.41% | [k4aqmsko](https://wandb.ai/euskon-johns-hopkins-university/ucf50-video-classification/runs/k4aqmsko) |
| **`lrcn-convnext-t-final`** | **LRCN, ConvNeXt-Tiny + BiLSTM** | **train + val** | – | **93.80%** ✅ | 93.71% | [d0nd2zhb](https://wandb.ai/euskon-johns-hopkins-university/ucf50-video-classification/runs/d0nd2zhb) |

**Final model: `lrcn-convnext-t-final`, 93.80% test accuracy.** All design choices
(backbone, LR recipe, 20-epoch schedule) were made on the validation split with the
`lrcn-convnext-t-bilstm` run. The final model then reuses that exact recipe, trained on
train+val (more groups, so more actor and scene diversity) with the fixed schedule, and is
evaluated on the test split once. Ensembles and TTA were also compared *on validation*
(`--eval_split val`) and did not help, so the final number is a single model with a single
center view.

W&B project: <https://wandb.ai/euskon-johns-hopkins-university/ucf50-video-classification>.
Each run logs `train/loss`, `train/acc`, `val/loss`, `val/acc` and `lr` per epoch, plus
`test/loss`, `test/acc` (and `test_tta/*`) and a test confusion matrix at the end. The
train+val final run has no separate val curve, since val is part of its training data. Its
validation curve is the `lrcn-convnext-t-bilstm` run, which uses the identical recipe.

> Train/val losses include label smoothing (0.1), so they flatten out around 0.7 rather than
> 0. The test loss is the plain negative log-likelihood.

---

## Quick start

```bash
# 1. Environment (Python 3.12, PyTorch + CUDA 12.6 wheels, wandb, pylint, pytest)
conda env create -f environment.yaml
conda activate assignment4
# WSL/Linux only: avoid a libstdc++ clash between torch and scipy
conda env config vars set -n assignment4 LD_LIBRARY_PATH=$CONDA_PREFIX/lib

# 2. Data: unpack UCF50 so that data/UCF50/<ClassName>/v_<Class>_gXX_cYY.avi exists
#    (https://www.crcv.ucf.edu/data/UCF50.rar)

# 3. One-time preprocessing: dedupe, extract 32 frames/video, build the group split (~2-3 min)
python src/prepare_data.py

# 4. Log in to Weights & Biases once (or pass --wandb_mode disabled / offline)
wandb login

# 5. Train
#    a) pick the recipe on validation (best-val checkpoint + early stopping, ≈45 min)
bash train.sh --run_name lrcn-convnext-t-bilstm
#    b) final model: same recipe on train+val, fixed schedule, last epoch kept  (≈55 min)
bash train.sh --train_on_val --run_name lrcn-convnext-t-final
#    other configurations from the results table
bash train.sh --cnn_backbone resnet50 --run_name lrcn-r50-bilstm
bash train.sh --model_type 3dcnn --learning_rate 1e-3 --backbone_lr_mult 0.3 \
              --n_epochs 25 --patience 0 --run_name r3d18-inflated

# 6. Evaluate on the test groups (single view + 6-view TTA are both reported)
bash test.sh runs/lrcn-convnext-t-final/best.pt
bash test.sh runs/a/best.pt runs/b/best.pt                 # equal-weight ensemble
bash test.sh runs/a/best.pt runs/b/best.pt --eval_split val  # compare choices on val, not test

# 7. Quality checks
pytest            # unit tests (CPU-only, ~10 s)
pylint src tests  # lint (10.00/10)
```

`train.sh` and `test.sh` pass any extra flags through to `src/run.py`, for example
`bash train.sh --n_epochs 5 --batch_size 4 --wandb_mode disabled`.

---

## How it works
- (B, T, C, H, W) = Batch size, Time steps, Channels, Height, and Width
```
data/UCF50/*.avi
   │  prepare_data.py: MD5 dedupe → 32 uniformly spaced frames/video → group split
   ▼
data/UCF50_frames/<Class>/<video>/frame_0000..0031.jpg  +  splits.json
   │  UCF50Dataset: pick fr_per_vid (16) frames, one per temporal segment
   │                (random within segment for train, segment center for eval)
   │  clip-consistent augmentation (same crop/flip/jitter for every frame)
   ▼
(B, T, C, H, W) ──► LRCN          or   (B, C, T, H, W) ──► R3D-18
   │  AdamW (backbone LR ×0.1), warm-up + cosine LR, label smoothing 0.1,
   │  fp16 autocast, grad clipping, early stopping on val accuracy
   ▼
runs/<run_name>/best.pt  ──►  test-set evaluation  ──►  W&B + test_report.txt
```

### Why splitting by *group* matters (the data leak)
Each UCF50 class has 25 groups. All clips in a group are cut from the **same long video**
(same person, background, camera). A random per-clip split puts sibling clips of the same
group in both train and test, so the model can score highly just by recognising the scene.
We hold out whole groups instead (default: test = 4 groups, val = 3 groups, train = the rest).
The split is the same for every class, and it is saved in `splits.json` so it is reproducible.

---

## Project structure

```
assignment4/
├── README.md            ← this guide
├── report.md            ← bug-hunt / change log with evidence
├── train.sh / test.sh   ← one-line training / evaluation wrappers around src/run.py
├── environment.yaml     ← conda env (pip installs CUDA 12.6 torch wheels)
├── requirements.txt     ← plain pip alternative
├── pytest.ini           ← pytest config (tests/ dir, src/ on the import path)
├── .pylintrc            ← pylint config (100-char lines, ML short names)
├── src/
│   ├── prepare_data.py  ← CLI: dedupe → extract frames → group split → splits.json
│   ├── video_datasets.py← UCF50Dataset, group_split, duplicate finder, frame sampling, collates
│   ├── utils.py         ← frame extraction (get_frames/store_frames), transforms, DataLoaders, seeding
│   ├── models.py        ← LRCN, R3D18, ImageNet→3D kernel inflation, build_model factory
│   ├── train.py         ← optimizer/scheduler builders, run_epoch, train loop (+ checkpointing)
│   ├── evaluate.py      ← multi-view (TTA) prediction, metrics, report, confusion matrix
│   └── run.py           ← CLI entry point: --mode train | eval (+ ensembles), W&B logging
├── tests/               ← pytest unit tests for the key pieces
├── data/                ← (git-ignored) UCF50 videos and extracted frames
└── runs/                ← (git-ignored) one folder per run: best.pt, history.json,
                           test[_tta]_report.txt, test[_tta]_confusion.npy
```

### Module details

- **`prepare_data.py`** is resumable: frames that already exist are skipped, so re-running with
  a different `--seed` / `--n_test_groups` / `--n_val_groups` only rewrites `splits.json`.
  `splits.json` also records the held-out groups, the removed duplicates and any videos that
  could not be decoded.
- **`video_datasets.UCF50Dataset(samples, fr_per_vid, transforms, train)`** returns
  `(clip[T,C,H,W], label)`. Frames are loaded in sorted (temporal) order. Every clip has
  exactly `fr_per_vid` frames, so no padding is needed.
- **`utils.compose_data_transforms`** uses `torchvision.transforms.v2` on the whole clip, so the
  random parameters are drawn once per clip.
- **`models.build_model(cfg)`** builds either model from a config dict. The same dict is saved
  in every checkpoint, which lets `--mode eval` rebuild the model from `--ckpt` alone.
- **`train.train(...)`** saves `best.pt` whenever validation accuracy improves and stops
  early after `--patience` epochs without improvement. With `--train_on_val` it trains on
  train+val for the full schedule and keeps the last epoch. It never touches the test split.
- **`run.py`** evaluates the checkpoint on the test split once, at the end of training,
  both single-view and with 6-view TTA (3 temporal offsets × mirror). In eval mode it
  accepts several checkpoints and also scores their probability-averaged ensemble.
  `--eval_split val` does the same on validation, so those choices never use the test set.

---

## Tuning: where to change things

All options are `src/run.py` flags (`python src/run.py --help`). Pass them to `train.sh`.

| What | Flag(s) | Default | Where it lives |
|---|---|---|---|
| Model family | `--model_type lrcn\|3dcnn` | `lrcn` | `models.build_model` |
| LRCN backbone | `--cnn_backbone resnet18…152, convnext_tiny, convnext_small, efficientnet_v2_s` | `convnext_tiny` | `models.BACKBONES` |
| ImageNet init | `--pretrained / --no-pretrained` | on | `models.py` |
| LSTM | `--rnn_hidden_size`, `--rnn_n_layers`, `--bidirectional`, `--temporal_pool mean\|last` | 512, 1, on, mean | `models.LRCN` |
| Frames per clip | `--fr_per_vid` | 16 | `video_datasets.sample_frame_indices` |
| Input size | `--img_size` | 224 (lrcn) / 160 (3dcnn) | `utils.DEFAULT_IMG_SIZE` |
| Augmentation | edit `compose_data_transforms` | crop 0.35-1, flip, jitter 0.2 | `utils.py` |
| Batch size | `--batch_size` | 8 | (16×224² clips: ConvNeXt-T ≈ 8.4 GB, ResNet-50 ≈ 9.3 GB VRAM) |
| LR / schedule | `--learning_rate`, `--backbone_lr_mult`, `--warmup_epochs`, `--n_epochs` | 3e-4, 0.1, 1, 20 | `train.build_optimizer/build_scheduler` |
| Regularisation | `--dropout`, `--weight_decay`, `--label_smoothing` | 0.5, 0.01, 0.1 | `run.py` |
| Early stopping | `--patience` (0 = off) | 6 | `train.train` |
| Final fit on train+val | `--train_on_val` | off | `run.run_train`, `train.train` |
| Test-time augmentation | `--test_clips`, `--test_hflip / --no-test_hflip` | 3, on | `run.make_test_loaders` |
| Eval split / ensembles | `--eval_split val\|test`, several `--ckpt` paths | test | `run.run_eval` |
| Split | `prepare_data.py --n_test_groups --n_val_groups --seed` | 4, 3, 0 | `video_datasets.group_split` |
| Frames stored per video | `prepare_data.py --n_frames` | 32 | (re-extract after changing) |
| Logging | `--wandb_project`, `--wandb_mode online\|offline\|disabled`, `--run_name` | online | `run.init_wandb` |
| DataLoader workers | `--num_workers` (prepare: `--workers`) | 6 | keep ≤ 6 on a 16 GB WSL machine |

---

## What was wrong with the original code

Short version (full details, evidence and reasoning are in [`report.md`](report.md)):

1. **Data leak (split):** clips were split randomly per clip, so sibling clips from the same
   UCF50 group (same actor/scene) were in both train and test. → Split by group.
2. **Data leak (duplicates):** 14 videos are byte-identical copies of other videos, 9 of
   them across different groups. → MD5 dedupe in `prepare_data.py`.
3. **The really significant bug: the LSTM mixed videos together.** `nn.LSTM` was built
   without `batch_first=True` but fed `(batch, 1, features)`, so it treated the *batch* as the
   time axis and each video's prediction depended on the other videos in the batch. It also
   never modelled the frames of one video as a sequence. → `batch_first=True`, whole
   `(B, T, F)` sequence in one call. A regression test is in `tests/test_models.py`.
4. **Frames loaded in the wrong order / only the start of the video:** `glob` results were not
   sorted (`frame10` sorted before `frame2`), and `[:16]` kept only the first frames.
   → Zero-padded names, sorted loading, segment sampling over the whole clip.
5. **Augmentation applied independently per frame**, so frames of one clip had different
   flips/shifts. → Clip-consistent `transforms.v2`.
6. **No UCF50 support at all:** there was no preprocessing script (the README pointed to a
   non-existent `preprocess.py`), the code was HMDB51-specific (`--n_classes 51`, paths), and
   `get_frames` returned `n+1` frames using `int16` indices.
7. **Crashes / dead code:** `run_training.py` imported a non-existent `compose_dataloaders`;
   `ReduceLROnPlateau(verbose=...)` and `resnetXX(pretrained=...)` were removed/deprecated in
   current PyTorch; `test.sh` pointed at another user's checkpoint and had no `--frame_dir`;
   `--pretrained False` parsed as the truthy string `"False"`; `resnet152` silently built a
   `resnet34`; `model_type=3dcnn` had collate/normalisation code but no model.
8. **Misc:** `test.py` shadowed Python's stdlib `test` package (renamed to `evaluate.py`);
   the W&B API key file `.keys` was not git-ignored.

### Model-level improvements (and why)

1. **Stronger ImageNet backbone + bidirectional LSTM with temporal mean-pooling.** ResNet-34
   (73.3% ImageNet top-1) → ResNet-50 V2 weights (80.9%) → **ConvNeXt-Tiny (82.5%)**. Better
   per-frame features were the biggest single gain (+1.6 pt test from R50 to ConvNeXt-T).
   Averaging the BiLSTM outputs over all time steps uses every frame, not only the last
   hidden state.
2. **Transfer-learning recipe:** discriminative learning rates (backbone ×0.1 so the
   pretrained features are not destroyed), AdamW, linear warm-up + cosine decay, label
   smoothing 0.1, dropout 0.5, gradient clipping, and early stopping on validation accuracy.
3. **Better data pipeline / regularisation:** TSN-style segment sampling over the whole
   video (random offset per segment during training, which acts as temporal augmentation),
   clip-consistent RandomResizedCrop + flip + colour jitter.
4. **3D CNN via ImageNet weight inflation (I3D):** R3D-18 with every 2D kernel repeated over
   time and divided by its temporal size, so the network starts with ImageNet features but
   can learn spatio-temporal filters. *Outcome:* without video pre-training, 4.8k clips are
   not enough to learn good 3D filters (73.9% test), and adding it to an ensemble lowered
   accuracy, so it is not part of the final model.
5. **Final fit on train+val** with the val-chosen recipe and a fixed schedule: more
   training groups (actor/scene diversity) for the cross-group test (+1.6 pt test).
6. **Engineering:** mixed precision (fp16 autocast + GradScaler) and channels-last memory
   format cut activation memory enough for 8 clips × 16 frames × 224² to fit on a 10 GB GPU.
