# Data and splits

Manifests contain no media. Their order, labels, source groups, and eligibility are shared across checkpoints. Paths are relative to `--data-root`. Download each dataset under its own terms; model weights and dataset licenses remain with their original providers.

| Manifest | Items | Groups / targets |
| --- | ---: | --- |
| `esc50.json` | 2,000 | 1,524 recording groups; 50 classes |
| `hmdb51.json` | 6,371 | clip groups; 51 classes; 8 frames |
| `objectnet.json` | 969 | 103 eligible classes |
| `country211.json` | 673 | 104 eligible classes |
| `ave_train.json` | 3,731 | 3,694 source videos; 28 classes |
| `ave_test.json` | 402 | 402 source videos; 28 classes |
| `clevr.json` | 15,000 | unique scenes; 24 binary color–shape labels |
| `clevr_fresh.json` | 2,000 | test-only scenes, 250 per object count |
| `cater.json` | 5,493 | complete encoded clips; ordering tasks use 1,613 unique sources |

## Audio and natural images

Obtain [ESC-50](https://github.com/karolpiczak/ESC-50) at revision `33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6`; put its `audio/` directory at `data/esc50/audio/`. Recording groups come from the released source-file IDs, not the official ESC fold number.

The paper's Country211 and ObjectNet sets are the eligible image-classification subsets released with [MMEB-V2](https://huggingface.co/datasets/TIGER-Lab/MMEB-V2), not the complete upstream datasets. Download the pinned image archive:

```bash
hf download TIGER-Lab/MMEB-V2 image-tasks/mmeb_v1.tar.gz --repo-type dataset \
  --revision e7bbfeb69a70dfe32ff36da3d6d8dbe31fc36af1 --local-dir data/downloads
```

Extract the archive and place its `MMEB/Country211/` and `MMEB/ObjectNet/` directories under `data/mmeb/`. The manifests name the exact images. No extra crops, augmentations, or background changes are applied.

## HMDB51

Obtain the original [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) videos. Put them at `data/hmdb51/videos/<class>/<filename>.avi`. The manifest uses the union of official split membership, with clip IDs as groups, matching the paper's recoverability analysis. These grouped CV scores are not official challenge split scores.

Extraction decodes with PyAV, samples rounded endpoint-inclusive uniform indices, and retains min(8, source frame count) RGB frames without resizing. The same decoded frames feed every checkpoint. No processor resampling is allowed.

## AVE

Use the released [AVE dataset](https://github.com/YapengTian/AVE-ECCV18). The paper's test inputs use the distributed `AVE.tar` containing `AVE/AVE_Dataset/` with its original frames and audio. Training uses its original ten-second MP4 clips and official annotations. The supplied training manifest has already excluded sources occurring in the official test set, as well as zero-duration events. It retains multiple training annotations from the same source as rows, with a shared source group.

```bash
uv run python scripts/prepare_media.py ave-test \
  --manifest splits/ave_test.json --source data/downloads/AVE.tar --data-root data
uv run python scripts/prepare_media.py ave-train \
  --manifest splits/ave_train.json --source data/ave/videos --data-root data
```

Training frames use FFmpeg at 10 fps, JPEG quality 2; audio is mono 16 kHz PCM. Both train and test retain 16 event-aligned frames and the released event audio interval. A source that ends slightly before an annotated audio endpoint is cropped at EOF, with the shortfall printed; no silence is appended. Test media are copied directly from the archive. The zero-based source indices in the manifests account for the archive's one-based filenames.

Extract audio and video from each manifest separately. The saved filenames use `ave_train_...` and `ave_test_...`, so training and test caches cannot collide. The protocol rejects any train/test source overlap.

## CLEVR

Download [CLEVR v1.0](https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip). Place the validation images at `data/clevr/images/` and scene annotations at `data/clevr/scenes/CLEVR_val_scenes.json`. Targets indicate presence anywhere in the scene of each color–shape combination. Text inputs serialize every object's attributes followed by exact scene relations; the extraction script builds this text from the original scene graphs.

To render the fresh-scene control, obtain the public [CLEVR scene generator](https://github.com/facebookresearch/clevr-dataset-gen) at commit `f0ce2c81750bfae09b5bf94d009f42e055f2cb3a`, including its Blender assets. Use Blender 2.78c:

```bash
blender --background --python scripts/render_clevr.py -- \
  --generator third_party/clevr-dataset-gen --schedule splits/clevr_render_schedule.json \
  --output data/clevr_fresh --start 0 --count 2000
```

The wrapper fixes each scene's Python RNG seed and object count without changing the upstream placement or rendering algorithm. Images use 320×240 pixels, 512 Cycles samples, CPU rendering, and a 200-pixel visibility minimum. Render disjoint index ranges into the same output directory if needed; existing scene/image files are never overwritten. The expected filenames are `CLEVR_e3b_000000.png` through `CLEVR_e3b_001999.png`.

`clevr_fresh_supports.json` stores only original-validation training indices. Each target-count domain has 45 fixed 5,000-scene supports for each training-count domain. The fits are separate from the full-data nested recognition analysis. Fresh scenes are always test-only. Paired-scene intervals stratify by exact object count and average AUROC over the individual fits; they do not pool their score scales.

## CATER

Obtain the released [CATER](https://github.com/rohitgirdhar/CATER) max2action videos. The source filenames are `CATER_new_*.avi`. Prepare exactly the manifest's 32 frames, with FFmpeg JPEG quality 3 and no resize:

```bash
uv run python scripts/prepare_media.py cater \
  --manifest splits/cater.json --source data/cater/videos --data-root data
uv run --extra encode python scripts/extract_embeddings.py \
  --model qwen2b --checkpoint models/qwen2b --manifest splits/cater.json \
  --data-root data --cache-dir embeddings --reverse
uv run python scripts/run_recap.py temporal --manifest splits/cater.json \
  --bank embeddings/cater_qwen2b_video_d2048.npz --width 2048 \
  --output outputs/temporal/qwen2b/d2048
```

`order_tasks` fixes six action-pair targets from strictly non-overlapping event intervals; ambiguous orders are excluded. Reversal changes frame order and flips the target. Both views share the same source group. The main temporal table uses Gemma and Qwen 2B at their full and minimum widths.

## Cache format

Each `.npz` bank stores `x` (N×D), `ids` (N strings), `model` (checkpoint key), `modality`, and `prompts` (4×K×D) where native text readers are evaluated. AVE and CATER omit unused prompt queries. A temporal bank also stores `reverse` (N×D). No pickle is used. Labels and groups live in the manifest, so independently encoded banks cannot silently change them. An accompanying JSON file records the checkpoint revision, software versions, modality, width, and inference settings. Truncation always renormalizes in FP32.

The supplied manifests and supported checkpoint widths are the paper's input contract. Missing media or incompatible banks raise errors; they are not silently substituted or filtered.
