> **Archived.** Part of the NTOU senior capstone *Volleyball Match Analysis System Based on Deep Learning*.
> The code now lives in [volleyball-analysis/training/action-recognition](https://github.com/DL-Volleyball-Analysis/volleyball-analysis/tree/main/training/action-recognition); this repository is kept read-only as the record of the capstone version.

# Capstone: Volleyball Action Recognition | 排球動作辨識

A YOLOv11m detector for five player actions (block, receive, serve, set, spike) on full video frames, trained on
two merged Roboflow Universe datasets (24,806 images: 18,616 train, 3,636 validation, 2,554 test).

<p align="center"><img src="docs/action-boxes.jpg" width="720" alt="Action detections in the capstone web app"></p>

## Result
| | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|
| Validation split (reported in the capstone) | 0.945 | 0.755 |
| Test split (measured October 2026) | 0.957 | 0.790 |

Per class on the test split: block 0.987, receive 0.863, serve 0.967, set 0.976, spike 0.991. Every test image with
a source-video name comes from a video that is also in the training split, so these are not unseen-match scores;
receive has about a fifth of spike's training boxes and there is no dig class. Details:
[docs/results/actions.md](https://github.com/DL-Volleyball-Analysis/volleyball-analysis/blob/main/docs/results/actions.md).

## Contents
| Path | What it is |
|---|---|
| `train_volleyball.py` | training script (Ultralytics, 200 epochs, 640 px, SGD) |
| `requirements.txt` | dependencies |

The Ultralytics pretrained `yolo11m.pt` base weights were removed from the latest version to keep the repository light; they remain in the history at
[`aec7bc4`](https://github.com/DL-Volleyball-Analysis/action-recognition-yolov11/tree/aec7bc4).

## Data
[Volleyball Actions](https://universe.roboflow.com/actions-players/volleyball-actions/dataset/5) and
[Volleyball Action Recognition](https://universe.roboflow.com/vbanalyzer/volleyball-action-recognition-k6tqv/dataset/6)
(Roboflow Universe, CC BY 4.0).

## Team
Liang Yu-Jia 梁祐嘉 (lead), Tsai Pei-Ying 蔡佩穎, Chung Chia-Hsin 鍾佳芯; advisor Professor Ting Pei-Yi 丁培毅.
Department of Computer Science and Engineering, National Taiwan Ocean University.
MIT License.
