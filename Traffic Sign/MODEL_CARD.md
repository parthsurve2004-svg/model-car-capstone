# Model card: printed traffic signs

Trained 2026-10-30. This is an actual fine-tuned YOLOv10n model, not a placeholder or stock COCO label filter.

## Intended use

Controlled model-track demonstration with the included STOP, No Entry, straight compulsory-left and straight compulsory-right printable designs. Input is an image or selected camera/video frames. Camera recognition is not guaranteed outside the training domain. This is not a road-certified driving system.

## Artifacts and automatic selection

- `models/traffic_signs.pt`: trained four-class PyTorch checkpoint.
- `models/traffic_signs_ncnn_model/`: verified FP16-weight NCNN export at input size 320. The runtime selects this when NCNN is installed, otherwise the checkpoint.
- Class IDs: 0 `stop`, 1 `no_entry`, 2 `turn_left`, 3 `turn_right`.
- Default sign confidence threshold: 0.60. Three fresh observations confirm an instruction.
- Missing bundled weights now fail startup unless `--disable-traffic-signs` is explicitly selected.

## Actual training

- Initialization: repository's COCO-pretrained YOLOv10n checkpoint, not a new network trained from scratch.
- 25 completed epochs, input320, batch8, AdamW, initial learning rate0.001, freeze first10 layers, no horizontal/vertical flipping.
- Seed20261030; CPU4threads; approximately963.4 seconds training on cloud CPU. This is not Raspberry Pi timing.
- Best validation checkpoint from the completed run was selected. Both best/last checkpoints passed the canonical card checks after the shared postprocessor.
- Source: `training/generate_dataset.py`, `train_model.py`, `evaluate_model.py`.

## Dataset and independence

| Split | Images | Sign instances | Negative frames |
|---|---:|---:|---:|
| Train | 768 | 695 | 256 |
| Validation | 128 | 123 | 43 |
| Test | 128 | 118 | 43 |

These are **synthetically rendered signs**, not photographs of physical prints. Cards are composited onto repository video frames and procedural backgrounds, with scale, perspective, blur, brightness and negative-shape variations. Different videos/procedural seeds are used across splits. The same rendering family is shared, so the synthetic test has limited independence from training appearance.

No real Pi-camera photos of the user's prints were supplied. The model supports the included straight-arrow templates; it has not been trained for every curved/elbow-arrow sign variation. The complete generated dataset is supplied separately as Traffic-Sign-Training-Images.zip. Sample images/labels are inside training_samples/.

## Results actually measured

On the128-image held-out **synthetic** test set, the Ultralytics evaluator reported:

- Precision0.98462; recall0.99896.
- mAP@IoU0.50:0.99486; mAP@IoU0.50:0.95:0.97783.

These are synthetic detection metrics, not physical recognition accuracy or the confidence of every runtime decision. See validation/synthetic_evaluation.json for per-class values and exact scope.

At confidence0.60 with the actual runtime postprocessor:

- All8 canonical tests passed:4 native1024-card images plus4 cards pasted into640x480 frames.
-10 unfamiliar-font STOP variants:9 passed; a small padded OpenSans Condensed Bold STOP was missed.
-43 negative test images:1 false STOP, a red octagon reading SLOW at confidence about0.752. The failure image is included in validation/known_false_stop_slow_octagon.jpg.
- FP16 NCNN matched checkpoint decisions on all18 canonical/font challenge cases, including the known miss.

YOLOv10 emitted overlapping weaker alternate-class boxes on some large signs. The runtime removes duplicates at IoU>=0.60 only when the stronger cross-class confidence exceeds the weaker by>=0.10; near-tied conflicts remain ambiguous. Separate spatial signs are retained. The shared postprocessor is used in evaluation and runtime; raw and processed results are both retained.

## Recognition-to-action verification

`training/check_actions.py` ran actual checkpoint inference and actual NCNN inference through TrafficSignRuntime, using simulated clear sonar inputs, virtual time and an in-memory motor recorder. All4 cases passed for both formats:

- STOP -> STOP command and timed dwell.
- No Entry -> STOP command and sign-policy hold.
- Left -> TURN -90 at configured PWM210.
- Right -> TURN 90 at configured PWM210.

Reports: validation/action_check_pt.json and action_check_ncnn.json. These checks did not open serial ports or move a vehicle. They do not establish physical turn angle, stopping distance or camera recognition.

## Before physical operation

First inspect real photos/videos of the actual printed cards using detect.py with motors disconnected. Then inspect the CSI feed and dashboard in --bench-safe mode. Confirm capture orientation, sonar telemetry, battery/power conditions, turn polarity and Uno TURN_CAL before a low-speed controlled-track test. Never assume a print will stop a moving vehicle at an unmeasured distance.

The camera's apparent sign size/bearing gate is a heuristic, not calibrated metric ranging or a general intersection detector. Lane turns use visual departure/reacquisition; autonomous/follow sign turns remain timed because the existing IMU yaw is a stub. Pi inference speed and thermal behavior were not benchmarked here.

## Reproduction and integrity

The full training history is in validation/training_results.csv. The selected checkpoint and NCNN checksums are listed in models/SHA256SUMS. The pretrained initialization SHA256 was11287ed0735678e7ba1ac2a9b3098c049155b3fde123992e724c1264bcc16b6f. The dataset manifest SHA256 was81bc3f3d197b906e2e1794e00b9d251a695ddf9b11d17cc491432bce08e77aff.

Install the compatible Ultralytics/torch/NCNN runtime from the project requirements; use matching export/runtime versions if deployment reports an unsupported operator.
