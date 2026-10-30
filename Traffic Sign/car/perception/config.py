"""
config.py — Runtime Configuration for Pi 5 Perception & Navigation
==================================================================
Owned by perception-agent.
Central place for sensor pins, model paths, thresholds, and I2C addresses.
"""

from pathlib import Path

# ── Model & Inference ────────────────────────────────────────────────────────
# Relative to repo root or car/ runtime directory
# Share original repository models; this snapshot adds one path level.
REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = str(REPO_ROOT / "benchmarking" / "models" / "yolov10n_ncnn_model")
POTHOLE_MODEL_PATH = str(REPO_ROOT / "pothole_detection" / "best.pt")
IMGSZ = 320                            # Optimal resolution identified in benchmarks
POTHOLE_IMGSZ = 416                    # Resolution for D40 pothole model (matches 4:3 GPU fine-tuning)
CONFIDENCE_THRESHOLD = 0.25            # Detection threshold for positive obstacles
POTHOLE_CONFIDENCE_THRESHOLD = 0.20    # High-recall/precision threshold for GPU fine-tuned D40 printout potholes
POTHOLE_LOWER_Y_MIN_FRAC = 0.40        # Potholes must lie in the lower section of the camera (yc >= 40% height)

# ── Serial Link (Pi 5 ↔ Arduino Uno) ─────────────────────────────────────────
# Default transport: USB-serial (/dev/ttyUSB0 or /dev/ttyACM0).
# For GPIO UART: /dev/ttyAMA0
SERIAL_PORT = "/dev/ttyUSB0"
SERIAL_BAUD = 115200                   # Matches arduino/PROTOCOL.md

# ── Camera (CSI — OmniVision OV5647 5MP Rev 1.3) ──────────────────────────────
# Replaces broken Global Shutter; connected via 22-pin to 15-pin FPC cable.
# Managed via picamera2 on Pi 5
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_HFOV_DEG = 53.5                  # Horizontal Field of View (~53.5° for 3.6mm lens)
CAMERA_AWB_MODE = "Daylight"            # White balance profile matching daylight.jpg

# ── Ultrasonic Sensors (2× HC-SR04P, both front-facing) ──────────────────────
# Polling is owned by the Arduino Uno to avoid Pi OS GPIO jitter.
# Distances are received as '[TELEM] US LEFT=x RIGHT=y' serial messages
# (parsed by MotorController.read_telemetry → SensorManager.read_ultrasonic).
EMERGENCY_STOP_DISTANCE_CM = 25.0      # Stop immediately if obstacle closer than 25cm (Architecture v2)

# ── IMU (ICM20948 over I2C) ──────────────────────────────────────────────────
IMU_I2C_BUS = 1
IMU_I2C_ADDRESS = 0x68

# ── Follow-Me & Person Re-Identification ─────────────────────────────────────
FOLLOW_CALIBRATION_FRAMES = 8          # Multi-frame samples to build target signature
FOLLOW_SIMILARITY_THRESHOLD = 0.58     # Min combined score (clothing + figure) to match target
FOLLOW_TRACK_VERIFY_THRESHOLD = 0.50   # Min score during active tracking before dropping to RE_ACQUIRE
FOLLOW_ANTI_CONFUSION_MARGIN = 0.08    # Required margin over overlapping distractor person
FOLLOW_SEARCH_STEP_DEG = 28            # Degrees per stepped turn toward last_seen_side
FOLLOW_SEARCH_TURN_SPEED = 165         # PWM speed for search turns during RE_ACQUIRE
FOLLOW_SEARCH_MAX_STEPS = 4            # Max directional search turns before holding stationary
FOLLOW_SEARCH_INTERVAL_S = 0.85        # Delay (s) between search turns to stabilize camera & scan

# ── Chart-Paper Lane Detection & Following (Option A) ────────────────────────
LANE_ROI_Y_MIN_FRAC = 0.45             # Scan lower 55% of frame for chart paper lines
LANE_LOOKAHEAD_Y_FRAC = 0.68           # Vertical lookahead row for steering error (0=top, 1=bottom)
LANE_DEFAULT_WIDTH_FRAC = 0.52         # Expected normalized spacing between left & right lines at reference row
LANE_EMA_ALPHA = 0.45                  # Temporal smoothing factor for polynomial coefficients
LANE_MAX_JUMP_FRAC = 0.15              # Max frame-to-frame lateral jump before rejecting outlier
LANE_LOST_MAX_FRAMES = 8               # Coast frames allowed before stopping when both lines disappear
LANE_CRUISE_SPEED = 165                # Forward PWM speed when centered in lane
LANE_CURVE_SPEED = 145                 # PWM speed when executing corrective lane turns
LANE_STEER_DEADBAND_DEG = 6            # Deadband (deg) within which car drives straight
LANE_MAX_STEER_DEG = 38                # Maximum skid-steer correction angle for sharp paper curves
LANE_COLOR_MODE = "dark"               # "dark" (black lines on white chart paper), "yellow", "blue", "white"
