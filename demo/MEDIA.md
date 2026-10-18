# Media and result attribution

These small files are presentation copies of existing benchmark inputs. Dataset and original-media rights remain with their respective owners; including a sample does not relicense the underlying media. No training data, embedding bank, or checkpoint is distributed with the demo.

| File | Source and transformation |
| --- | --- |
| `assets/can-opening.wav` | ESC-50 `1-101404-A-34.wav`; the original five-second, mono PCM samples, with container metadata stripped. |
| `assets/waveform.svg` | Amplitude envelope of that exact audio, using 240 display bins. No denoising or synthetic audio. |
| `assets/horse.mp4` | AVE test source `--I2373YpgM`; the same 16 sampled frames used for the experiment, resized to 640 pixels wide and encoded as a silent H.264 preview at 5 fps. |
| `assets/guitar.wav` | AVE test source `-44qKBR8bGQ`; the original event-audio PCM samples, with container metadata stripped. |
| `assets/cellphone.jpg` | ObjectNet eligible image `image_s0003472.jpg`, from the MMEB-V2 image release; resized for display and saved without EXIF. |
| `assets/bath-towel.jpg` | ObjectNet eligible image `image_s0006091.jpg`, from the same release; resized for display and saved without EXIF. |

The ESC-50 clip derives from **“Can Open.WAV” by LG**, [Freesound sound 101404](https://freesound.org/people/LG/sounds/101404/), credited as CC BY in the dataset's [clip-attribution list](https://github.com/karolpiczak/ESC-50/blob/master/LICENSE). ESC-50 as a dataset is distributed under [CC BY-NC 3.0](https://creativecommons.org/licenses/by-nc/3.0/). See [ESC-50](https://github.com/karolpiczak/ESC-50), [AVE](https://github.com/YapengTian/AVE-ECCV18), [ObjectNet](https://objectnet.dev/), and the [MMEB-V2 data release](https://huggingface.co/datasets/TIGER-Lab/MMEB-V2) for the original releases and terms.

The README overview and compression interaction are computed from [the bundled numerical outputs](../paper_results/results.json). Audio predictions are from the nested held-out natural-task analysis; AVE predictions use fixed logistic regression at C = 1; image predictions use nested Ridge. Each example's identity and measured predictions are in [examples.json](examples.json).

Selection is intentional and outcome-based. The audio selection rule used Gemma 2, Jina, and Omni Mini 0.9B across three split seeds; Omni Mini 2.3B is displayed as additional context. For AVE, each clip is the first test ID meeting the recorded same-modality-correct / cross-modality-wrong pattern. The cellphone rescue is a deliberate visual choice from 156 consistent three-seed rescues, rather than the first eligible item. The bath-towel regression is the first eligible regression in manifest order. These examples illustrate disagreements; they do not estimate their frequency.
