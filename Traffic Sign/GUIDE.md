# Four-sign traffic recognition

This optional feature recognizes four sign classes for the model track. This trained package supplies custom weights; see MODEL_CARD.md for actual results. The existing COCO detector alone is not the four-class traffic-sign model. Physical sign-recognition accuracy and Raspberry Pi throughput remain unverified.

| Class name | Printed sign | Intended behavior |
|---|---|---|
| `stop` | Indian STOP octagon | Stop on confirmation, hold a timed dwell, then allow normal safety checks to decide whether motion can resume |
| `no_entry` | No Entry | Hold until an operator resets the sign policy |
| `turn_left` | Blue circular compulsory-left sign | Prepare a left maneuver when a confirmed sign becomes actionable |
| `turn_right` | Blue circular compulsory-right sign | Prepare a right maneuver when a confirmed sign becomes actionable |

Use compulsory-direction signs, not triangular bend-warning signs. The software cannot establish where a sign legally applies, the distance to a stopping line, or whether a physical intersection is safe. Automatic actions use apparent sign size and bearing as controlled-track heuristics, not a general intersection detector. Use consistent print sizes and sign placement, then tune against the actual track.

## Dataset and training

No actual camera photographs were supplied. A synthetic printed-sign dataset was generated and actual fine-tuning was performed; reproducible sources are in training/ and results are in MODEL_CARD.md. Real camera photographs remain necessary to evaluate and improve physical-domain performance. If an external collection is later selected, inspect its class names, bounding-box format, split structure and sample annotations before reuse. Do not assume numeric class IDs mean the four classes listed here.

Collect and label images of the actual Indian-standard printed signs used on the track. Include changes in scale, distance, lighting, tilt, background, motion blur and partial occlusion. Include negative images without signs and objects that resemble signs. Keep separate train/validation/test scenes and capture sessions; adjacent frames from the same recording must not be split across those sets.

Use YOLO detection annotations: one row per sign, `class_id center_x center_y width height`, with coordinates normalized to image dimensions. A dataset configuration can be:

```yaml
path: /absolute/path/to/traffic-sign-data
train: images/train
val: images/val
test: images/test
names:
  0: stop
  1: no_entry
  2: turn_left
  3: turn_right
```

The matching label directories are `labels/train`, `labels/val` and `labels/test`. Every image must have correct annotations; check small/far signs as well as large close signs. The model's saved class names must match the four names above, independently of class ID order.

For existing YOLO labels, read the source dataset YAML and remap class IDs deliberately when selecting these four signs. For Pascal VOC XML, convert pixel bounding boxes using each image's dimensions. For COCO JSON, resolve category IDs through the JSON category table and convert `[x, y, width, height]` boxes to normalized centers and sizes. Preserve the original dataset and inspect several converted labels overlaid on their images before training. Excluded sign categories should not silently become unannotated distractors in training images; choose and document the intended negative-image policy.

Train a nano detector on a workstation using the repository's Ultralytics-compatible environment. An example starting command, with an existing pretrained YOLOv10n checkpoint, is:

```bash
yolo detect train model=/path/to/yolov10n.pt data=/path/to/traffic_signs.yaml imgsz=320 epochs=50 fliplr=0.0 flipud=0.0
```

This is a starting configuration, not a measured training recipe. Left/right arrows are directional labels: ordinary horizontal flipping would create incorrectly labeled opposite-direction signs. Vertical flipping is also inappropriate for upright track signs. Evaluate augmentation settings visually before training.

Validate the resulting checkpoint on held-out data and inspect errors by class and sign size. In particular, check false STOP/No Entry detections and confusion between compulsory left/right signs. Confidence scores are not proof of accuracy. Choose the runtime threshold after measuring precision/recall on representative track images.

## Deployment

The runtime accepts a custom Ultralytics checkpoint or compatible NCNN export. A typical export command is:

```bash
yolo export model=/path/to/best.pt format=ncnn imgsz=320
```

Check compatibility between the exporting Ultralytics/NCNN versions and the Pi's ARM64 runtime. If tooling cannot export on the training workstation, perform export in a supported environment before the demonstration. Copy the complete exported directory, including metadata and model files. Preserve the four class names. An NCNN export must still be validated on the real Pi; its presence does not establish detection accuracy or FPS.

Traffic-sign recognition automatically uses the bundled trained model. Missing bundled weights stop startup unless --disable-traffic-signs is explicitly requested. It uses a separate, rate-limited predictor so that existing object and pothole model outputs remain separate from sign instructions. Restricting four versus seven output classes has much less effect on workload than changing model size, resolution, or inference cadence. Rate limiting reduces average work but a synchronous inference call can still delay a control-loop iteration.

From the repository root, enter this snapshot folder once suitable trained weights exist:

```bash
cd "Traffic Sign"
python run.py --mode lane --bench-safe
```

Use `--mode autonomous` for reactive cruising instead. Sign handling supports autonomous, lane and follow operation; direct manual browser movement is rejected while it is enabled. Use `--disable-traffic-signs` to retain operation without traffic instructions. Follow mode preserves the owner profile but requires reacquisition after sign stops/turns. The example uses bench-safe mode for initial recognition inspection, not a claim that physical movement has been validated.

Turn parameters can be set with `--traffic-turn-ms-per-degree` (default `10`, accepted range `1`–`100`) and `--traffic-turn-pwm` (default `210`, accepted range `0`–`255`). Match the time calibration to the actual firmware/chassis and choose PWM through physical low-risk bench testing; these values are not a measured vehicle-speed specification. Apparent-size/bearing gates are helper-function defaults, not additional command-line flags.

## Policy and track operation

The integration confirms signs across fresh observations; cached predictions must not count as new confirmations. A persistent sign is consumed once rather than issuing repeated stops or turns every camera frame. A No Entry restriction requires explicit reset. A turn sign prepares an instruction rather than immediately rotating the car upon its first distant appearance.

The initial defaults are a maximum of four sign-prediction passes per second, three fresh confirmations, a three-second STOP hold and a ten-second pending-turn lifetime. The dashboard's traffic-sign card shows the detected sign, horizontal bearing, action, pending direction and fault state, with a reset control. Sign boxes are overlaid on the MJPEG stream. The same state is available in the `traffic_signs` object in `/telemetry`.

The STOP dwell is nonblocking: camera, sonar, telemetry and emergency processing continue. It counts time commanded stopped rather than counting an active Arduino avoidance/reverse maneuver as a completed stop. There is no visual stopping-line detector, so STOP recognition does not promise a stop at an exact painted line. No Entry remains held until explicit reset; a continuously visible consumed sign is not repeatedly applied every frame.

Automatic behavior does not require an operator junction trigger. It still depends on fresh sign observations and safety clearance. Obstacle/reflex states, emergency latches, camera failure, and disabled motors must inhibit sign-driven movement. Resetting the traffic-sign policy must never release an independent emergency latch.

The camera baseline is the current OV5647 configuration at 640×480. Main capture already applies horizontal and vertical flips for the documented 180-degree mounting when its Picamera2 transform is available. Sign processing must use that same oriented frame; it must not add another mirror/rotation that reverses the arrows. Verify a known left-arrow print appears as left in the live image before enabling motion.

The automatic approach gate uses normalized bounding-box height as an apparent-size measure and a horizontal pinhole bearing derived from box center and the configured 53.5-degree horizontal field of view. These measurements do not recover metric range. There is no calibrated camera-pitch projection, known-size ranging, stopping-line detector or intersection map in this feature. A larger printed sign can trigger at a greater physical distance than a smaller one. Treat the thresholds as model-track settings requiring physical placement tests, not traffic-engineering guarantees.

By default a confirmed pending turn is eligible when its sign box is at least 12% of image height and its horizontal bearing is within ±22 degrees. On a 480-pixel-high frame the height gate is about 58 pixels. A smaller or peripheral sign can be detected and displayed while the car continues approaching under its ordinary navigation rules. The gate does not prove that the vehicle has reached the center of an intersection.

In autonomous mode the initial maneuver requests a ±90-degree turn at the configured turn PWM (default 210), bounded by a calibrated turn-time estimate. The initial calibration is 10 milliseconds per requested degree; no gyro/encoder feedback measures the achieved heading. Sign completion means the timed command has finished, not that actual yaw was verified.

In lane mode the turn is made through bounded 12-degree-command pulses while evaluating current lane geometry, with an explicit stop at each pulse boundary. Completion requires at least 72 commanded degrees and an observed departure from the original centered-lane condition. Three successive centered `DUAL_LOCKED` lane observations can then complete the maneuver: steering error must be within ±6 degrees and lateral offset within ±10%. Those confirmations occur after the pulse's STOP boundary on subsequent current frames, with a fresh obstacle check before continuing; a pre-STOP view is not accepted as proof of completion. At most 96 degrees are commanded in pulses; total maneuver time is limited to six seconds. These commanded degrees are accumulated requests, not measured yaw. An unchanged centered view of the original lane cannot complete the turn and eventually times out. The car must reacquire the centered two-boundary corridor before ordinary lane following resumes. A timeout or a safety interruption stops an active turn and holds a fault for explicit reset rather than silently continuing forward.

Starting or continuing a sign turn also requires both front sonar distances to be valid and at least 42 cm, sonar receipt age no more than 0.8 s, obstacle detections no older than 0.5 s, and sign observations no older than 0.75 s. Receipt age cannot establish when the Uno physically measured a distance; it only rejects stale serial telemetry at the Pi. Before a turn has begun, an obstacle makes the pending instruction wait for ordinary safety clearance, subject to its expiry, rather than initiating the maneuver. Interruption after movement has begun creates a turn fault requiring reset; the code does not resume a partly executed turn blindly.

To clear a No Entry hold or interrupted-turn fault, use `POST /traffic-signs/reset` on the configured dashboard host/port. Reset discards a pending turn and does not release an independent obstacle/emergency latch. The old `POST /traffic-signs/junction` endpoint returns `410 Gone`; it no longer triggers movement. These controls follow the existing local dashboard's access model; they do not add authenticated remote control.

Start with stationary recognition checks, followed by a controlled low-speed track demonstration. Confirm actual motor polarity and turn calibration before evaluating automatic behavior. Keep physical Pi performance results separate from synthetic or mocked software test results.

## Snapshot layout and tests

This folder contains an isolated copy of the traffic-sign vehicle runtime. Launching `run.py` selects this folder's `car` package. General obstacle and pothole model paths intentionally resolve to the original repository root (`../benchmarking/models` and `../pothole_detection`); traffic-sign weights must be supplied explicitly. Trained sign weights and printable cards are included in this updated package. Training images are available as a separate generated-data archive; virtual environments and generated caches are excluded.

From this folder, with the repository virtual environment already installed, run:

```bash
../.venv/bin/python -m pytest -q tests
../.venv/bin/python run.py --help
```

These checks use mocks and synthetic images. They do not establish physical sign accuracy, Raspberry Pi inference throughput or calibrated chassis yaw.

## Recognition checks before movement

Use `python detect.py --source printable_cards/stop.png --output stop-check` from this folder for motor-free inference. Inspect the annotated image and JSON, then repeat using a photograph of your physical print at the actual camera angle. `--source 0` is an OpenCV webcam; the Picamera2 CSI path is in run.py. A digital-card result and synthetic held-out scores do not establish physical reliability.

Print cards are in printable_cards/. Actual training examples and their labels are in training_samples/. Read MODEL_CARD.md for exact model files, training duration/epochs, test counts, operational-threshold checks and known mistakes.
