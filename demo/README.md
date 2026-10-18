# Listen. Switch the reader. Keep the input fixed.

[← Repository overview](../README.md) · [Reproduce the paper](../docs/reproduce.md) · [Example predictions](examples.json)

A small, offline companion to RECAP. Start the server from the repository root:

```bash
python -m http.server 8000 --bind 127.0.0.1
```

Open **[localhost:8000/demo/](http://localhost:8000/demo/)**. You can also open [index.html](index.html) directly in a browser. GitHub displays HTML source rather than running the page, so use the local server for the interactive version.

| Try this | What to notice |
| --- | --- |
| **Listen before revealing the label.** | Four checkpoints make different prompt errors on one ESC-50 sound. Nested logistic readers recover the dataset label from the same vectors. |
| **Keep the target fixed; switch the training modality.** | The AVE horse video stays a horse video when the audio-trained reader predicts speech. On the guitar recording, the video-trained reader predicts banjo. |
| **Compress Jina from 1,024D to 32D.** | The ranking between selected prompts and fixed Ridge reverses on ObjectNet. These are full-manifest scores, not the score of one example. |
| **Look at both image outcomes.** | A learned reader rescues the cellphone and regresses on the bath towel. Better mean accuracy does not mean every prediction improves. |

Everything shown is a saved held-out result. The page has no inference server, API calls, external fonts, trackers, or JavaScript dependencies. Media do not autoplay. The 16-frame horse preview plays at 5 fps; it is not native-rate playback or synchronized with another sound.

The examples are outcome-selected. [examples.json](examples.json) records the IDs, predictions, and selection rules; [MEDIA.md](MEDIA.md) records sources and display transformations. Population-level evidence is kept separate from the examples.

To rebuild the overview, waveform, presentation media, and browser data from the prepared benchmark files:

```bash
uv run python scripts/build_demo.py --data-root data
```

This uses the existing evaluation outputs and local media. It neither loads a checkpoint nor fits a reader. Rebuilding the video preview also requires FFmpeg. Omit `--data-root` to regenerate only the overview and numerical browser data.
