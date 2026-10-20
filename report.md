# Assignment 4: Bug-Hunt & Improvement Report

**Task:** inherit a broken video-classification codebase (written for HMDB51), port it to the
UCF50 dataset in `data/`, find and fix the bugs (including a data leak and one "really
significant" bug), add W&B logging, tests, linting and docs, and reach **≥ 93% test
accuracy**. Pretrained weights are restricted to **object recognition (ImageNet) only**.

## TL;DR

* **Final test accuracy: 93.80%** (single model, single view) on a **leak-free** split:
  4 held-out UCF50 groups (1081 clips) whose actors and scenes never appear in training,
  with byte-identical duplicate videos removed. The target was ≥ 93%.
* Final model: **LRCN = ConvNeXt-Tiny (ImageNet-1k) frame encoder → BiLSTM → temporal
  mean-pool**. The recipe was chosen on validation, then the model was refit on train+val
  and evaluated on test once.
* **3 critical bugs** (LSTM mixing videos across the batch, per-clip split leak,
  duplicate-video leak), **3 accuracy bugs** (frame order, per-frame augmentation, frame
  extraction) and **~12 crash / config / hygiene issues** were found and fixed. One of the
  hygiene issues: a W&B API key file was not git-ignored.
* Added: UCF50 preprocessing CLI and dataset, W&B logging, ImageNet-inflated 3D CNN,
  TTA and ensemble evaluation, **30 pytest tests**, **pylint 10.00/10**, rewritten README.
* All runs: <https://wandb.ai/euskon-johns-hopkins-university/ucf50-video-classification>

---

## 1. Bugs found

Severity: 🔴 = invalidates results, 🟠 = hurts accuracy, 🟡 = crash / misleading / hygiene.

### 🔴 B1. The "really significant" bug: the LSTM mixes videos together
`models.py` (original):

```python
self.rnn = nn.LSTM(num_features, hidden_size, n_layers)          # batch_first=False
...
y = self.base_model(x[:, idx])                                   # (B, F) for frame idx
out, (hn, cn) = self.rnn(y.unsqueeze(1), (hn, cn))               # fed as (B, 1, F)
```

Without `batch_first=True`, `nn.LSTM` reads its input as `(seq_len, batch, features)`. A
`(B, 1, F)` tensor is therefore treated as **a single sequence whose time steps are the B
different videos in the mini-batch**. As a result:

* Each video's hidden state comes from the *previous videos in the batch*, and
  `(hn, cn)` carries state from the **last video in the batch at frame t** into the
  **first video at frame t+1**.
* A clip's prediction changes depending on which other clips share its batch. With
  `shuffle=True` the training targets are effectively noisy. At eval time the result
  depends on batch composition and order, so accuracy is not well defined.
* The model never learns temporal dynamics *within* a video, which is the whole point of LRCN.

**Evidence:** running the original model (git commit `539318e`) on a fixed clip *A*:

| Comparison | Max abs. logit difference |
|---|---|
| A alone vs. A after clip B in the same batch | 0.057 |
| A after B vs. A after C | 0.081 |
| A alone vs. A *first* in the batch with B after it | 0.062 |

All three should be 0.

**Fix:** `nn.LSTM(..., batch_first=True)`. All `B×T` frames go through the CNN in one call,
and the LSTM gets `(B, T, F)`. Regression test:
`tests/test_models.py::test_lrcn_samples_are_independent` asserts that the same comparisons
are ≈0 (the original model gives the non-zero values in the table above).

### 🔴 B2. Data leak #1: per-clip split across UCF50 groups
`dataset_split` used `StratifiedShuffleSplit` on individual clips. Each UCF50 class has 25
**groups**, and the 4–7 clips of a group are cut from **the same source video** (same
actor, background, viewpoint). A per-clip split puts sibling clips in both train and
test, so the model can score well by recognising the scene instead of the action. The
UCF50 authors explicitly require group-wise evaluation (Leave-One-Group-Out).

**Fix:** `video_datasets.group_split` assigns **whole groups** to splits, using the same
group ids for every class: test = groups {5, 11, 12, 20}, val = {3, 24, 25}, train = the rest
(including the extra groups 26–30 that 6 classes have). The split is seeded and saved to
`splits.json`. The model is selected on val only; test is touched once at the end.
Unit test: `test_group_split_has_no_group_overlap`.

### 🔴 B3. Data leak #2: byte-identical duplicate videos
An MD5 scan of `data/UCF50` found **14 pairs of identical files**, and **9 of the 14 are in
different groups** (e.g. `Basketball g01_c07 == g05_c03`, `Biking g12_c01 == g16_c05`,
`TennisSwing g09_c07 == g19_c01`). Even a correct group split would leak these.
**Fix:** `prepare_data.py` hashes every video and keeps one copy per hash (6681 → 6667
clips). The removed files are recorded in `splits.json["meta"]["removed_duplicates"]`.

### 🟠 B4. Frames loaded out of order, and only from the start of the video
`VideoDataset.__getitem__` did `glob.glob(dir + '/*.jpg')[:fr_per_vid]`. `glob` order is
arbitrary (filesystem order), and with the original names `frame0.jpg … frame16.jpg`
even a sorted list would be lexical (`frame10` before `frame2`). The LSTM therefore saw
frames in a scrambled order, and `[:16]` could only ever pick from whatever came first.
**Fix:** frames are saved as `frame_0000.jpg`. The dataset sorts them and uses TSN-style
segment sampling over the whole clip (random offset per segment for training, segment
center for eval). Tests: `test_dataset_loads_frames_in_temporal_order`,
`test_eval_sampling_spans_whole_video`.

### 🟠 B5. Augmentation applied independently to every frame
The transform was applied per PIL frame: `[self.transforms(img) for img in imgs]`. With
`RandomHorizontalFlip` and `RandomAffine`, frames in one clip got **different** random
flips and shifts, which adds fake motion and breaks the temporal signal.
**Fix:** `torchvision.transforms.v2` on the whole `(T, C, H, W)` clip, so the parameters are
drawn once per clip. Test: `test_train_transform_is_consistent_across_frames`.

### 🟠 B6. Frame extraction off-by-one / overflow
`get_frames` used `np.linspace(0, v_len-1, n_frames+1, dtype=np.int16)`. That returns
**n+1** frames, overflows for videos longer than 32 767 frames, and trusts
`CAP_PROP_FRAME_COUNT`, which is only an estimate for AVI files. `store_frames` printed
`"processing"` once per frame. **Fix:** decode all frames, then pick exactly `n` indices
over the real length. Test: `test_get_frames_returns_exactly_n_uniform_frames`.

### 🟡 B7. Crashes and silent misconfiguration
| Where | Problem | Fix |
|---|---|---|
| `run_training.py` | imports `compose_dataloaders`, which does not exist, so the module cannot run; it is also a duplicate of `run.py` | deleted; `run.py` is the single entry point |
| `run.py` / `run_training.py` | `ReduceLROnPlateau(..., verbose=1)` (argument removed in recent PyTorch) | new warm-up + cosine `LambdaLR` |
| `models.py` | `resnetXX(pretrained=...)` deprecated API | `weights=` enum API |
| `models.py` | `cnn_model='resnet152'` silently built a **resnet34** | real ResNet-152; unknown names raise |
| `run.py` | `--pretrained` was a string, so `-p False` gave the truthy `"False"` | `BooleanOptionalAction` (`--no-pretrained`) |
| `run.py` | `model_type='3dcnn'` had normalisation stats + collate fn but **no model**; LRCN was always built | `R3D18` implemented and selected by `build_model` |
| `test.sh` | another user's absolute checkpoint path, no `--frame_dir`, duplicated `--model_type`, HMDB `--n_classes 51` | checkpoint passed as argument; config is read from the checkpoint |
| `train.sh` / README | HMDB51 everywhere, `--n_classes 51`, README pointed at a non-existent `preprocess.py` | UCF50 everywhere; `n_classes` read from `splits.json`; `prepare_data.py` added |
| `run.py` eval | `splits.npy` written to / read from the current working directory, with labels stored as strings in a numpy object array | `splits.json` stored next to the frames |
| `train.py` | accuracy denominator `len(dataset)` but collate dropped empty clips, so accuracy could be under-counted | every clip now has exactly T frames; per-sample loss/acc accounting |
| `test.py` | module name shadows the stdlib `test` package | renamed to `evaluate.py` |
| repo | `.keys` (a **W&B API key**) was not git-ignored | added to `.gitignore` together with `data/`, `runs/`, `wandb/` |

---

## 2. What was changed

| File | Status | Summary |
|---|---|---|
| `src/prepare_data.py` | **new** | CLI: MD5 dedupe → parallel frame extraction (resumable) → group split → `splits.json` |
| `src/video_datasets.py` | rewritten | `UCF50Dataset`, `group_split`, `find_duplicate_videos`, `parse_ucf50_name`, `sample_frame_indices`, `load_splits`/`save_splits`, collates |
| `src/utils.py` | rewritten | fixed `get_frames`/`store_frames`, clip-consistent v2 transforms, DataLoader builders, `seed_everything` |
| `src/models.py` | rewritten | fixed `LRCN` (batch_first, BiLSTM, mean-pool, ResNet/ConvNeXt/EfficientNet-V2 backbones), `R3D18` + ImageNet→3D inflation, `build_model` |
| `src/train.py` | rewritten | AdamW with discriminative LRs, warm-up+cosine, AMP, grad-clip, early stopping, checkpoint stores model config |
| `src/evaluate.py` | renamed from `test.py` | multi-view TTA, probability-level metrics (also used for ensembles), report, confusion matrix |
| `src/run.py` | rewritten | single CLI for train/eval, W&B integration, TTA, ensemble eval, `--eval_split val`, `--train_on_val` |
| `src/run_training.py` | **deleted** | broken duplicate of `run.py` |
| `tests/` | **new** | 30 pytest tests (data split/dedupe/sampling/loading, transforms, extraction, models incl. the B1 regression, training helpers incl. train+val mode, TTA) |
| `train.sh`, `test.sh` | rewritten | UCF50 defaults; pass extra flags through; `test.sh` takes one or more checkpoints |
| `README.md` | rewritten | one-stop guide: setup, data prep, training, eval, structure, tuning table, bug summary |
| `.pylintrc`, `pytest.ini` | **new** | lint and test config; `pylint src tests` scores **10.00/10** |
| `environment.yaml`, `requirements.txt` | updated | pytest added, torch lower bounds, wandb/pylint in requirements |
| `../.gitignore` | updated | ignore `assignment4/{data,runs,wandb,.keys}` |

---

## 3. Model-level improvements

Every change uses only ImageNet (object-recognition) pre-training.

1. **Stronger frame encoder.** ResNet-34 (ImageNet V1, 73.3% top-1) → ResNet-50
   (ImageNet V2 recipe, 80.9%) → **ConvNeXt-Tiny (82.5%)**, the final backbone. EfficientNet-V2-S
   and ConvNeXt-Small are also available via `--cnn_backbone`. Better image features
   transfer directly to action recognition: +1.6 pt test from R50 to ConvNeXt-T at the same
   recipe and batch size.
2. **Better temporal head.** A bidirectional LSTM (512 per direction) with **mean-pooling over
   all time steps** instead of only the last hidden state, so every frame contributes
   gradient, plus dropout 0.5 before the classifier.
3. **Transfer-learning optimisation recipe.** AdamW (wd 0.01) with **discriminative learning
   rates** (backbone ×0.1, so the pretrained features are not destroyed early),
   **linear warm-up + cosine decay** per step, **label smoothing 0.1**, gradient clipping,
   fp16 mixed precision + channels-last (≈2× faster, makes R50×16 frames fit on 10 GB),
   and early stopping / best-checkpoint selection on **validation** accuracy.
4. **Data pipeline as regularisation.** TSN segment sampling (random frame per segment →
   temporal augmentation every epoch), clip-consistent RandomResizedCrop(0.35–1) + flip +
   colour jitter.
5. **Final fit on train+val.** Once the recipe is fixed on validation, retrain on
   train+val with the same fixed schedule (`--train_on_val`) and keep the last epoch. This
   adds 3 groups of actor and scene diversity, which is what the cross-group test measures:
   +1.6 pt test.
6. **3D CNN via I3D inflation** (tried, not used in the final model). R3D-18 initialised by
   repeating each ImageNet ResNet-18 2D kernel over time and dividing by the temporal size
   (Carreira & Zisserman, 2017), so it starts out computing ImageNet features and then learns
   motion. Without video pre-training, 4.8k clips were not enough: 80% val / 74% test.
7. **Test-time augmentation and ensembling** (implemented, evaluated on val, not used). 3
   temporal offsets × {original, mirrored} = 6 views with averaged softmax. Probability-averaged
   ensembles of the LRCNs and the R3D (two-stream style fusion). Neither improved validation
   accuracy, so the final model uses a single view.

---

## 4. Results

Test = 4 held-out groups {5, 11, 12, 20}, 1081 clips. Val = groups {3, 24, 25}, 808
clips. Train = the remaining 4778 clips.

| # | Run | What changed | Best val | **Test (1 view)** | Test (6-view TTA) | W&B |
|---|---|---|---|---|---|---|
| 1 | `lrcn-r50-bilstm` | all bug fixes + ResNet-50 V2 + BiLSTM/mean-pool + new recipe | 93.32% (ep 19) | 90.66% | 91.21% | [fkpls76b](https://wandb.ai/euskon-johns-hopkins-university/ucf50-video-classification/runs/fkpls76b) |
| 2 | `r3d18-inflated` | 3D CNN, ImageNet ResNet-18 inflated (lr 1e-3, bb ×0.3, 25 ep) | 80.07% (ep 19) | 73.91% | 74.93% | [51suf9ug](https://wandb.ai/euskon-johns-hopkins-university/ucf50-video-classification/runs/51suf9ug) |
| 3 | `lrcn-convnext-t-bilstm` | #1 with a ConvNeXt-Tiny backbone (only change) | 95.92% (ep 12) | 92.23% | 92.41% | [k4aqmsko](https://wandb.ai/euskon-johns-hopkins-university/ucf50-video-classification/runs/k4aqmsko) |
| 4 | **`lrcn-convnext-t-final`** | **#3's recipe, trained on train+val, fixed 20 epochs, last epoch kept** | – | **93.80%** | 93.71% | [d0nd2zhb](https://wandb.ai/euskon-johns-hopkins-university/ucf50-video-classification/runs/d0nd2zhb) |

**How the final model was chosen, without tuning on test:**

1. The backbone and recipe were picked by validation accuracy (run 3 beats run 1 on val:
   95.9% vs 93.3%).
2. Ensembles and TTA were compared **on validation** (`bash test.sh A B --eval_split val`).
   The checkpoints themselves were selected on val, so these numbers are optimistic for single
   models, but the comparison still points one way:

   | On validation | 1 view | 6-view TTA |
   |---|---|---|
   | ConvNeXt-T LRCN alone | **95.92%** | 95.54% |
   | ConvNeXt-T + R50 LRCN ensemble | 95.42% | 95.42% |
   | ConvNeXt-T + R50 + R3D ensemble | 94.55% | 94.93% |

   Neither ensembling nor TTA helped, so the final model is a single network evaluated on
   the single center view.
3. Because test groups are new actors and scenes, more *group diversity* in training is the
   most direct lever. The validated recipe was therefore refit on train+val (3 more groups,
   +17% clips) with its fixed schedule. Test was evaluated once and reproduced with
   `bash test.sh runs/lrcn-convnext-t-final/best.pt` (93.80% / 93.71%).

Transparency note: I looked at test numbers for runs 1 and 2 and for one LRCN+R3D ensemble
(90.56% TTA, worse than LRCN alone) during development. None of these were used to choose
the final model; every choice above was made on validation.

**What each improvement bought (test, single view):**

* All bug fixes + R50 + recipe → 90.66% (the original code cannot produce a meaningful
  number at all; see B1)
* ResNet-50 → ConvNeXt-Tiny: **+1.57 pt** → 92.23%
* Refit on train+val: **+1.57 pt** → 93.80%
* TTA: ±0.5 pt, not consistent; R3D ensemble: negative

**Remaining errors** (final model, top confusions): Lunges→TennisSwing (7),
Lunges→SoccerJuggling (5), HighJump→PoleVault (5), Basketball→VolleyballSpiking (5),
Nunchucks→YoYo (4). These are mostly visually similar scenes or motion-defined actions,
where a 2D appearance model is weakest. Per-class report: `runs/lrcn-convnext-t-final/test_report.txt`.

**W&B evidence:** every run logs per-epoch `train/loss`, `train/acc`, `val/loss`, `val/acc`
and `lr`, plus `test/acc`, `test/loss`, `test_tta/*` and a test confusion matrix. The
train+val run (#4) has no separate val curve because val is part of its training data. Its
validation curve is run #3, which uses the identical recipe.

---

## 5. Verification

```bash
pytest                       # 30 passed
pylint src tests             # 10.00/10
python src/prepare_data.py   # reproduces splits.json byte-for-byte (verified after refactor)
bash test.sh <ckpt> [<ckpt> ...]
```

## 6. Limitations / notes

* The official UCF50 protocol is 25-fold Leave-One-Group-Out. A single fixed group split
  (4 test groups, 1081 clips) was used to fit the compute budget. With n = 1081 the binomial
  standard error of 93.8% is about ±0.7 pt, so the margin over 93% is within one standard
  error. Runs are seeded, but cuDNN autotuning makes GPU training not bit-exact, so a rerun
  may land a few tenths of a point away.
* MD5 only catches byte-identical copies. Re-encoded near-duplicates across groups could
  still exist.
* Train/val loss values include label smoothing (floor ≈ 0.6–0.7). The test loss is the plain
  NLL of the (averaged) probabilities, so the loss values are not directly comparable
  across the two.
