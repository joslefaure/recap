# Reproduce the paper

[← Overview](../README.md) · [Interactive examples](../demo/README.md) · [Data and splits](../splits/README.md)

Run all commands from the repository root. This guide preserves the checkpoint settings, folds, and candidate sets used for the paper.

## Setup

Use Python 3.10–3.12. The CPU environment is enough to run readers and rebuild the numerical paper figures.

```bash
uv sync
uv run python -m unittest discover -s tests -v
uv run python scripts/make_figures.py --output outputs/paper
```

With pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m unittest discover -s tests -v
```

For feature extraction, install the separate model dependencies:

```bash
uv sync --extra encode
# or: pip install -e '.[encode]'
```

The extraction stack pins PyTorch 2.8.0, Transformers 5.19.0, and Sentence Transformers 6.1.0. Choose a PyTorch wheel compatible with your CUDA driver. The CPU reader stack pins NumPy 2.2.6 and scikit-learn 1.7.2. Model code is loaded from local, revision-pinned checkpoints; Jina and Omni Mini use their released custom Python code. Start with one sample in a separate manifest before a full extraction. Default extraction batch size is one.

## Quickstart

The bundled `paper_results/results.json` is a small numerical snapshot of the completed evaluations. It contains measured scores, not embeddings or media. The plotting command above regenerates five quantitative figures and nine LaTeX tables under `outputs/paper/`. It requires no GPU or model download. Rebuilding figures is separate from rerunning the experiments.

To extract a checkpoint, download the revision in `configs/models.json`. For example, using the Hugging Face CLI:

```bash
hf download Qwen/Qwen3-VL-Embedding-2B \
  --revision 9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda \
  --local-dir models/qwen2b

uv run --extra encode python scripts/extract_embeddings.py \
  --model qwen2b --checkpoint models/qwen2b \
  --manifest splits/objectnet.json --data-root data --cache-dir embeddings

uv run python scripts/run_recap.py recoverability \
  --manifest splits/objectnet.json \
  --bank embeddings/objectnet_qwen2b_image_d2048.npz \
  --output outputs/recoverability/objectnet/qwen2b
```

Obtain the benchmark media as described in [data and split guide](../splits/README.md). The included manifests fix item order, source groups, class order, and targets. Do not replace them with random subsets when comparing checkpoints.

## Checkpoints

| Key | Checkpoint | Supported widths |
| --- | --- | --- |
| `gemma2` | `google/embeddinggemma-2` | 768, 512, 256, 128 |
| `qwen2b` | `Qwen/Qwen3-VL-Embedding-2B` | 2048, 1024, 512, 256, 128, 64 |
| `qwen8b` | `Qwen/Qwen3-VL-Embedding-8B` | 4096, 2048, 1024, 512, 256, 128, 64 |
| `jina` | `jinaai/jina-embeddings-v5-omni-small` | 1024, 768, 512, 256, 128, 64, 32 |
| `ovis` | `ATH-MaaS/Ovis-Omni-Embedding-3B` | 2048 |
| `omni09` | `MBZUAI/Omni-Embed-Mini-0.9B` | 1024, 512, 256, 128 |
| `omni23` | `MBZUAI/Omni-Embed-Mini-2.3B` | 2048, 1024, 512, 256, 128 |

Qwen has no native audio pathway. Jina consistently uses the retrieval adapter. Ovis uses the last non-padding Thinker state, without an MRL projection. Prefix truncation is followed by FP32 normalization; Gemma 64D is not supported.

Ovis needs an empty `--runtime-dir models/ovis-runtime`. Its embedding snapshot lacks a generation-only speaker map; the wrapper supplies an empty map in a separate runtime directory and reads only the Thinker output. Weight files are linked without modification.

Omni Mini also needs an empty runtime directory so its component paths resolve locally:

```bash
# 0.9B: --vision-model points to Qwen/Qwen3.5-0.8B, revision below.
# 2.3B: --image-processor points to the image/video processor files from its backbone.
uv run --extra encode python scripts/extract_embeddings.py \
  --model omni09 --checkpoint models/omni09 --runtime-dir models/omni09-runtime \
  --vision-model models/qwen35-08b \
  --manifest splits/objectnet.json --data-root data --cache-dir embeddings
```

For 2.3B, use `--model omni23 --checkpoint models/omni23 --runtime-dir models/omni23-runtime --image-processor models/qwen2b` with the same manifest and cache arguments. Its processor comes from the pinned Qwen 2B checkpoint above.

The 0.9B vision component is `Qwen/Qwen3.5-0.8B` at `2fc06364715b967f1860aea9cf38778875588b17`. Component downloads are part of the released checkpoint's implementation, not additional evaluated models. Keep runtime directories out of the release: they contain local paths and links.

## Running the protocols

All commands below read cached embeddings. Outputs go to new directories; existing results are never overwritten. Each run writes `results.json` and held-out predictions in `predictions.npz`. Scores in JSON are fractions; plots and most tables convert them to percentages.

### Axis 1: Compression ranking stability

 Full and minimum supported widths use identical source-grouped folds. Prompt selection uses only inner validation data. Ridge is refit with fixed alpha 10 at each width. The output reports both reader gaps, their difference, and whether their signs reverse.

```bash
uv run python scripts/run_recap.py compression \
  --manifest splits/objectnet.json --bank embeddings/objectnet_jina_image_d1024.npz \
  --output outputs/compression/objectnet/jina
```

The paper's endpoint panel uses Jina, Omni Mini 0.9B/2.3B, and Qwen 8B. Ovis has no compressed endpoint.

### Axis 2: Cross-modal AVE transfer

 Extract audio and video separately for `splits/ave_train.json` and `splits/ave_test.json`, using `--modality audio` or `--modality video`. Supply audio then video in both bank lists:

```bash
uv run python scripts/run_recap.py transfer \
  --manifest splits/ave_train.json \
  --bank embeddings/ave_train_jina_audio_d1024.npz embeddings/ave_train_jina_video_d1024.npz \
  --test-manifest splits/ave_test.json \
  --test-bank embeddings/ave_test_jina_audio_d1024.npz embeddings/ave_test_jina_video_d1024.npz \
  --output outputs/transfer/jina
```

Readers fit the 3,731 source-disjoint training rows and score all 402 official test events. Logistic C is fixed at 1. Matrix rows are training modalities and columns are test modalities. Portability penalties compare vertically within a target column. The paper panel uses Jina, Ovis, and the two Omni Mini checkpoints.

### Axis 3: Nested label recoverability

 Use `recoverability` with each applicable checkpoint on ESC-50, HMDB51, ObjectNet, and Country211 at full width. Five outer folds and five inner folds use seeds 20261011, 20261012, and 20261013. Ridge alpha is selected from 0.1, 1, 10, 100, 1000; logistic C from 0.01, 0.1, 1, 10, 100. Selection averages the inner-fold metric and breaks ties by listed order. Natural tasks use balanced accuracy over classes present in the evaluated partition; final out-of-fold scores cover the complete vocabulary.

For CLEVR, extract image and scene-text banks with `--modality image` and `--modality text`, then run `recoverability --width D` at Gemma 768/128 and Qwen 2B 2048/64. The 24 binary color–shape targets use macro AUROC. Folds are stratified by object count. The reported native baseline is the fixed “present” template, and the fixed-Ridge scores also report AUROC within exact object counts. A stratum is scored only if both binary classes are present; the output records the number of scored strata.

The fresh-scene control refits the exact 5,000-scene supports from the original validation set and evaluates only the 2,000 fresh scenes:

```bash
uv run python scripts/run_recap.py fresh \
  --manifest splits/clevr.json --bank embeddings/clevr_qwen2b_image_d2048.npz \
  --test-manifest splits/clevr_fresh.json --test-bank embeddings/clevr-fresh_qwen2b_image_d2048.npz \
  --supports splits/clevr_fresh_supports.json --width 2048 \
  --output outputs/fresh/qwen2b/d2048
```

Use `temporal` on `splits/cater.json` after extraction with `--reverse`. It fits paired forward/reversed views with alpha 10. The inner threshold uses three source-disjoint folds; both views of every source remain together. Joint correctness requires both directions to be correct. The reported score weights six ordering tasks and three seeds equally.

### Joint selection under resource constraints

 Pass all nonempty candidate banks listed for the task in `configs/selection.json`:

```bash
uv run python scripts/run_recap.py selection \
  --manifest splits/esc50.json \
  --bank embeddings/esc-50_gemma2_audio_d768.npz embeddings/esc-50_jina_audio_d1024.npz \
  --output outputs/selection/esc50
```

The paper's recorded candidate set is narrower than the full supported-width registry: Gemma 128D and Jina 128/512D for ESC-50; add Qwen 2B/8B at 64D for HMDB51. The other checkpoints had no eligible candidate in that recorded search. `configs/selection.json` preserves this set so reproduction does not expand the search after seeing the results. Evaluating every supported width would be a new selection experiment.

Both strategies receive the same 1/2/4/8 source groups per class. Inner validation uses only that support and min(5, g) folds; g=1 selects native prompts using the support itself and fits no supervised reader. Byte caps are 512 and 2,048 for FP32 vectors. Reader weights are shared storage outside the per-vector cap. Ties prefer narrower width, then reader, checkpoint, and template in the recorded order. ESC-50 fixes 40 eligible classes and HMDB51 fixes all 51. Each point averages 45 distinct outer-fold/support-draw evaluations; support-draw seeds are 20261021–20261023.

## Reproducing tables and figures

The default figure command reads the measured snapshot. To build from a complete set of new runs instead:

```bash
uv run python scripts/collect_results.py --runs outputs --output outputs/recomputed.json
uv run python scripts/make_figures.py --results outputs/recomputed.json --output outputs/recomputed-paper
```

The collector requires complete main-paper runs and rejects duplicate cells. Store full-width natural runs, the four ObjectNet compression runs, the four AVE runs, and both selection runs under `outputs/`. Add CLEVR image/text endpoint and CATER endpoint runs for their appendix tables. The collector never fills a missing measurement with a published value. Additional historical calibration controls are present in the bundled snapshot; rerunning the core protocols leaves uncomputed controls as `--`.

The quantitative figures are `width_changes_reader`, `reader_transfer`, `target_fixed_transfer`, `audio_readers`, and `budget_selection`. Tables retain the manuscript filenames. Add `--media-root data` to rebuild the teaser and qualitative image panel once the selected media in `configs/qualitative.json` are available. Full benchmark media are not distributed here; the demo includes a few attributed presentation samples. The qualitative items are fixed examples, not estimates of how often recovery occurs.

The tests exercise the CPU protocols and input validation. They do not download or run the seven encoders. Exact scores also depend on the pinned checkpoint, preprocessing, software stack, and decoded inputs. Bootstrap intervals resample paired source groups conditional on the saved fits; they do not measure encoder-training or probe-refit uncertainty. These diagnostics measure accessible label structure under specified readers, not information-theoretic capacity.
