# Listen. Switch the reader. Keep the input fixed.

[← Repository overview](../README.md) · [Reproduce the paper](../docs/reproduce.md) · [Example predictions](examples.json)

**[▶ Open the interactive demo](https://anonymous.4open.science/w/recap-975C/demo/)** — audio, video, images, and reader switches work directly in your browser.

| Try this | What to notice |
| --- | --- |
| **Listen before revealing the label.** | Four checkpoints make different prompt errors on one ESC-50 sound. Nested logistic readers recover the dataset label from the same vectors. |
| **Keep the target fixed; switch the training modality.** | The AVE horse video stays a horse video when the audio-trained reader predicts speech. On the guitar recording, the video-trained reader predicts banjo. |
| **Compress Jina from 1,024D to 32D.** | The ranking between selected prompts and fixed Ridge reverses on ObjectNet. These are full-manifest scores, not the score of one example. |
| **Look at both image outcomes.** | A learned reader rescues the cellphone and regresses on the bath towel. Better mean accuracy does not mean every prediction improves. |

Everything shown is a saved held-out result. The page has no inference server, API calls, external fonts, trackers, or JavaScript dependencies. Media do not autoplay. The 16-frame horse preview plays at 5 fps; it is not native-rate playback or synchronized with another sound.

The examples are outcome-selected. [examples.json](examples.json) records the IDs, predictions, and selection rules; [MEDIA.md](MEDIA.md) records sources and display transformations. Population-level evidence is kept separate from the examples.

[Media attribution](MEDIA.md) · [Asset build script](../scripts/build_demo.py)
