#!/usr/bin/env python3
"""
live_detect.py — Fullscreen Live Perception & Autonomous Control System
NMIMS MPSTME Capstone — Raspberry Pi 5 + Arduino Uno
Optimized: GPU Fine-Tuned Printout Pothole Model + Lower Camera Section ROI Filtering
Feature: Manual Control Obstacle Override Toggle (Web UI & 'O' Key)
"""

import io
import json
import logging
import socketserver
import threading
import time
import urllib.parse
from http import server
from pathlib import Path
from threading import Condition, Lock
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
try:
    from picamera2 import Picamera2
except Exception:
    Picamera2 = None
from ultralytics import YOLO
from car.perception.lane_detector import ChartLaneDetector, LaneDetectionResult
from car.perception.config import REPO_ROOT

# Standalone legacy runtime retained for compatibility/regression tests.
# This is not the traffic-sign launcher; use ../run.py for sign behavior.

# ── 1. Model & Path Resolution ───────────────────────────────────────────────
def find_model_path(rel_path: str) -> Optional[Path]:
    candidates = [
        REPO_ROOT / rel_path,  # Share original repository assets from this isolated snapshot.
        Path("/home/capstone/model") / rel_path,
        Path.home() / "model" / rel_path,
        Path.home() / "model-car-capstone" / rel_path,
        Path.cwd() / rel_path,
        Path(__file__).resolve().parent / rel_path,
        Path(__file__).resolve().parent.parent / rel_path,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

obs_candidates = [
    "benchmarking/models/yolov10n_ncnn_model",
    "benchmarking/models/yolov10n.pt",
    "yolov10n.pt",
]
obs_path = None
for cand in obs_candidates:
    p = find_model_path(cand)
    if p and p.exists():
        obs_path = p
        break

if not obs_path:
    obs_path = Path("/home/capstone/model/yolov10n_ncnn_model")

print(f"📦 Loading YOLOv10n obstacle model: {obs_path}")
try:
    obs_model = YOLO(str(obs_path), task="detect")
except Exception as e:
    print(f"⚠️ Failed to load primary obstacle model ({obs_path}): {e}")
    pt_fallback = find_model_path("yolov10n.pt") or find_model_path("benchmarking/models/yolov10n.pt")
    if pt_fallback:
        print(f"📦 Falling back to PyTorch weights: {pt_fallback}")
        obs_model = YOLO(str(pt_fallback), task="detect")
    else:
        obs_model = None

# Pothole Model Candidates (prioritizing GPU printout-fine-tuned best.pt)
pot_candidates = [
    "pothole_best.pt",
    "pothole_detection/best.pt",
    "pothole_detection/runs_rdd/printout_gpu_pothole_model/weights/best.pt",
    "pothole_detection/best_ncnn_model",
    "pothole_detection/runs_rdd/custom_pothole_model/weights/best.pt",
    "pothole_detection/rdd2022_yolov8n.pt",
]

pot_path = None
for cand in pot_candidates:
    p = find_model_path(cand)
    if p and p.exists():
        pot_path = p
        break

pot_model = None
if pot_path and pot_path.exists():
    print(f"📦 Loading GPU Fine-Tuned Pothole model: {pot_path}")
    try:
        pot_model = YOLO(str(pot_path), task="detect")
    except Exception as e:
        print(f"⚠️ Could not load pothole model ({pot_path}): {e}")
else:
    print("ℹ️ Using classical CV pothole detector as fallback (no YOLO pothole weights found).")

def find_daylight_ref() -> Optional[Path]:
    for p in [Path("/home/capstone/daylight.jpg"), Path("/home/capstone/model/daylight.jpg"), Path.home() / "daylight.jpg"]:
        if p.exists(): return p
    return None

CAM_WIDTH = 320
CAM_HEIGHT = 240
IMGSZ = 320
POTHOLE_IMGSZ = 416                     # Matches 4:3 GPU fine-tuning resolution for held printouts
OBSTACLE_CLASSES = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck
CONF_OBSTACLE = 0.35
CONF_POTHOLE = 0.20                     # Balanced recall/precision threshold for D40 potholes on held printouts
POTHOLE_LOWER_Y_MIN_FRAC = 0.40         # Pothole vertical center must lie in lower 60% of camera (yc >= 0.40*h)
POTHOLE_PERSIST_FRAMES = 2              # Consecutive pothole detections required before autonomous swerve
POTHOLE_ONLY = True                     # STRICT FILTER: Only detect D40 potholes
POTHOLE_EVERY = 2

BBOX_STOP_HEIGHT_FRAC = 0.38
EMERGENCY_STOP_DIST_CM = 25.0           # Critical ultrasonic auto-brake distance (< 25 cm)
US_AVOID_DIST_CM = 42.0                 # Directional ultrasonic avoidance steer zone (< 42 cm)
US_SLOW_DIST_CM = 60.0                  # Cautionary speed reduction zone (< 60 cm)
CRUISE_SPEED = 200                      # Normal cruising forward PWM
SLOW_SPEED = 135                        # Cautionary PWM when approaching obstacle

# ── Remote Control & Operating Modes ─────────────────────────────────────────
control_mode = "autonomous"             # "autonomous", "lane", or "manual"
manual_override_obstacles = False       # True = Ignore all obstacles & ultrasonic brake in manual mode
manual_speed = 185                      # Default manual drive PWM (100 - 255)
manual_action = "IDLE"                  # Current manual direction / status
lane_color_mode = "dark"
lane_detector = ChartLaneDetector(color_mode=lane_color_mode)
last_lane_res = None

# ── 2. Camera Controls & Daylight White Balance ──────────────────────────────
try:
    from libcamera import controls, Transform
    AWB_MODES = {
        "daylight": controls.AwbModeEnum.Daylight,
        "auto": controls.AwbModeEnum.Auto,
        "indoor": controls.AwbModeEnum.Indoor,
        "cloudy": controls.AwbModeEnum.Cloudy,
        "tungsten": controls.AwbModeEnum.Tungsten,
    }
except Exception:
    Transform = None
    AWB_MODES = {"auto": 0, "indoor": 4, "daylight": 5, "cloudy": 6, "tungsten": 2}

current_awb = "daylight"
swap_rb = True
hardware_rotated = False
cam_instance = None

def set_awb_mode(cam, mode_name: str) -> bool:
    global current_awb
    mode_name = mode_name.lower().strip()
    if mode_name not in AWB_MODES: return False
    val = AWB_MODES[mode_name]
    if cam is None:
        current_awb = mode_name
        return True
    try:
        cam.set_controls({"AwbMode": val})
        current_awb = mode_name
        print(f"☀️ Camera AWB set to: {mode_name.upper()} (matches daylight.jpg)")
        return True
    except Exception as e:
        print(f"⚠️ Failed to set AwbMode: {e}")
        return False

# ── 3. Noise-Resilient CLAHE & Classical CV Fallback Pothole Detection ───────
def enhance_contrast(roi: np.ndarray) -> np.ndarray:
    """CLAHE contrast normalization for held-up road photos, printouts, and noisy feeds."""
    if roi is None or getattr(roi, "size", 0) == 0:
        return roi
    denoised = cv2.GaussianBlur(roi, (3, 3), 0)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_c, a_c, b_c = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge((clahe.apply(l_c), a_c, b_c)), cv2.COLOR_LAB2BGR)

def cv_detect_potholes(frame: np.ndarray) -> List[Dict[str, Any]]:
    """Classical CV pothole fallback detector (used ONLY when YOLO pot_model is None)."""
    if frame is None or getattr(frame, "size", 0) == 0:
        return []
    h, w = frame.shape[:2]
    roi_y = int(h * 0.45)
    roi = frame[roi_y:, :]
    if roi.size == 0:
        return []

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    lap = cv2.Laplacian(blur, cv2.CV_64F)
    lap_u = np.uint8(np.clip(np.absolute(lap), 0, 255))
    _, thresh = cv2.threshold(lap_u, 36, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dets = []
    roi_area = roi.shape[0] * roi.shape[1]
    min_area = max(250, int(roi_area * 0.01))
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > roi_area * 0.22:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / max(ch, 1)
        if not (0.5 < aspect < 2.5):
            continue
        dets.append({
            "bbox": [x, y + roi_y, x + cw, y + roi_y + ch],
            "conf": 0.45,
            "label": "POTHOLE (CV)",
            "is_pothole": True
        })
    return dets

# ── 4. Arduino Uno Parallel Serial Bridge ────────────────────────────────────
class ArduinoBridge:
    def __init__(self):
        self.serial_port = None
        self.port_name = "SIMULATED"
        self.connected = False
        self.lock = Lock()
        self.latest_sonar = {"left": 999.0, "right": 999.0}
        self.latest_state = "IDLE"
        self.motors_enabled = False
        self.pi_latched = False
        self.invert_motors = True
        self.last_status_time = 0.0

    def auto_connect(self):
        try:
            import serial
        except ImportError:
            print("⚠️ pyserial not installed. Operating in DRY-RUN mode.")
            return False

        for p in ["/dev/ttyUSB0", "/dev/ttyACM0", "/dev/ttyUSB1", "/dev/ttyACM1"]:
            try:
                s = serial.Serial(p, 115200, timeout=0.1)
                time.sleep(2.0)
                s.reset_input_buffer()
                s.reset_output_buffer()
                self.serial_port = s
                self.port_name = p
                self.connected = True
                print(f"🔌 Arduino connected on {p} @ 115200 baud")

                self.send("TUNE SET TELEM_STREAM 1")
                self.send("TUNE SET CLIFF_ENABLE 0")
                self.send("PI_GO")
                self.send("STATUS")
                return True
            except Exception:
                continue
        print("⚠️ No physical Arduino detected. Operating in DRY-RUN mode.")
        self.connected = False
        return False

    def send(self, cmd: str) -> bool:
        if not self.connected or not self.serial_port: return False
        with self.lock:
            try:
                self.serial_port.write((cmd.strip() + "\n").encode("utf-8"))
                self.serial_port.flush()
                return True
            except Exception:
                self.connected = False
                return False

    def read_telemetry(self):
        if not self.connected or not self.serial_port: return
        now = time.time()
        if now - self.last_status_time > 1.5:
            self.send("STATUS")
            self.last_status_time = now

        try:
            while self.serial_port.in_waiting:
                line = self.serial_port.readline().decode("utf-8", errors="ignore").strip()
                if not line: continue
                if ("LEFT=" in line or "US_LEFT=" in line) and ("RIGHT=" in line or "US_RIGHT=" in line):
                    norm = line.replace("US_LEFT=", "LEFT=").replace("US_RIGHT=", "RIGHT=")
                    for pt in norm.split():
                        if pt.startswith("LEFT="):
                            try: self.latest_sonar["left"] = float(pt.split("=")[1])
                            except: pass
                        elif pt.startswith("RIGHT="):
                            try: self.latest_sonar["right"] = float(pt.split("=")[1])
                            except: pass
                if "STATE=" in line:
                    for pt in line.split():
                        if pt.startswith("STATE="):
                            self.latest_state = pt.split("=")[1]
        except Exception:
            pass

    def pi_stop(self):
        self.pi_latched = True
        self.send("PI_STOP")

    def pi_go(self):
        self.pi_latched = False
        self.send("PI_GO")

    def pi_avoid(self, direction: str):
        eff_dir = direction.upper()
        if self.invert_motors:
            eff_dir = "RIGHT" if eff_dir == "LEFT" else "LEFT"
        self.send(f"PI_AVOID {eff_dir}")

    def drive(self, speed: int, duration_ms: int, force: bool = False):
        if self.motors_enabled and (force or not self.pi_latched):
            cmd = f"REV {speed} {duration_ms}" if self.invert_motors else f"DRIVE {speed} {duration_ms}"
            self.send(cmd)

    def reverse(self, speed: int, duration_ms: int, force: bool = False):
        if self.motors_enabled and (force or not self.pi_latched):
            cmd = f"DRIVE {speed} {duration_ms}" if self.invert_motors else f"REV {speed} {duration_ms}"
            self.send(cmd)

    def turn(self, angle: int, speed: int = 150, force: bool = False):
        if self.motors_enabled and (force or not self.pi_latched):
            eff_angle = -angle if self.invert_motors else angle
            self.send(f"TURN {eff_angle} {speed}")

    def stop(self):
        self.send("STOP")

arduino = ArduinoBridge()
active_action_label = "IDLE (Safety Hold)"

def is_valid_us(dist: float) -> bool:
    """Returns True if reading is a valid obstacle reflection (filters out 0.0 noise & 999 timeout)."""
    return 1.0 < dist < 850.0

def evaluate_autonomous_action(frame_w: int, frame_h: int, obs_boxes: list, pot_dets: list, frame: Optional[np.ndarray] = None):
    global active_action_label, control_mode, manual_action, manual_override_obstacles, last_lane_res, lane_detector

    # 1. Camera Obstacle Check
    obstacle_ahead = False
    for b in obs_boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        height_frac = (y2 - y1) / max(frame_h, 1)
        cx = (x1 + x2) / 2
        if 0.20 * frame_w < cx < 0.80 * frame_w and height_frac > BBOX_STOP_HEIGHT_FRAC:
            obstacle_ahead = True
            break

    # ── MANUAL REMOTE CONTROL MODE ───────────────────────────────────────────
    if control_mode == "manual":
        us_l = arduino.latest_sonar["left"]
        us_r = arduino.latest_sonar["right"]
        v_l = is_valid_us(us_l)
        v_r = is_valid_us(us_r)
        front_blocked = (v_l and us_l < EMERGENCY_STOP_DIST_CM) or (v_r and us_r < EMERGENCY_STOP_DIST_CM)

        if manual_override_obstacles:
            # Full manual override: driver has complete authority through obstacles/potholes
            if front_blocked or obstacle_ahead:
                active_action_label = f"⚠️ MANUAL [OVERRIDE]: {manual_action} (Obstacle Ignored)"
            else:
                active_action_label = f"🎮 MANUAL [OVERRIDE]: {manual_action} (PWM {manual_speed})"
        else:
            if front_blocked:
                active_action_label = f"🛑 MANUAL SAFETY BRAKE (L:{us_l:.0f}cm R:{us_r:.0f}cm)"
                arduino.stop()
                manual_action = "SAFETY_BRAKE"
                return
            elif obstacle_ahead:
                active_action_label = f"⚠️ MANUAL WARN: Obstacle Ahead | {manual_action}"
            else:
                active_action_label = f"🎮 MANUAL REMOTE: {manual_action} (PWM {manual_speed})"
        return

    # ── CHART-PAPER LANE FOLLOWING MODE ──────────────────────────────────────
    if control_mode == "lane":
        exclude_boxes = []
        for b in obs_boxes:
            exclude_boxes.append({"bbox": list(map(int, b.xyxy[0].tolist()))})
        for p in pot_dets:
            if "bbox" in p:
                exclude_boxes.append({"bbox": p["bbox"]})

        lane_res = lane_detector.detect(frame, exclude_boxes=exclude_boxes) if frame is not None else LaneDetectionResult(status="LANE_LOST")
        last_lane_res = lane_res

        # Tier 1 Camera Obstacle Check
        if obstacle_ahead:
            active_action_label = "🛑 PI_STOP: Obstacle Ahead in Lane"
            arduino.pi_stop()
            return

        if arduino.pi_latched and not obstacle_ahead:
            active_action_label = "✅ PI_GO: Lane Cleared"
            arduino.pi_go()

        # Tier 2 Ultrasonic Auto-Brake
        us_l = arduino.latest_sonar["left"]
        us_r = arduino.latest_sonar["right"]
        v_l = is_valid_us(us_l)
        v_r = is_valid_us(us_r)
        if (v_l and us_l < EMERGENCY_STOP_DIST_CM) or (v_r and us_r < EMERGENCY_STOP_DIST_CM):
            active_action_label = f"🛑 US AUTO-BRAKE! (L:{us_l:.0f}cm R:{us_r:.0f}cm)"
            arduino.stop()
            return

        # Tier 2.5 In-Lane Obstacle/Pothole Bypass
        danger_left = sum(1.0 for p in pot_dets if p.get("is_pothole") and (p["bbox"][0] + p["bbox"][2]) / 2 < frame_w * 0.5)
        danger_right = sum(1.0 for p in pot_dets if p.get("is_pothole") and (p["bbox"][0] + p["bbox"][2]) / 2 >= frame_w * 0.5)
        if danger_left > 0 or danger_right > 0:
            if danger_left > danger_right:
                active_action_label = "🔀 LANE BYPASS: Steering RIGHT around Pothole"
                arduino.pi_avoid("RIGHT")
                if arduino.motors_enabled:
                    arduino.turn(32, 145)
            else:
                active_action_label = "🔀 LANE BYPASS: Steering LEFT around Pothole"
                arduino.pi_avoid("LEFT")
                if arduino.motors_enabled:
                    arduino.turn(-32, 145)
            return

        # Tier 3 Nominal Lane Centering via Pure Pursuit
        if lane_res.status == "LANE_LOST" and (lane_res.lost_frames > lane_detector.lost_max_frames or lane_res.center_coeffs is None):
            active_action_label = f"⚠️ LANE LOST ({lane_res.lost_frames}f): Holding Safe Stop"
            arduino.stop()
            return

        delta = lane_res.steer_angle_deg
        is_coasting = (lane_res.status == "LANE_LOST")

        if abs(delta) <= lane_detector.steer_deadband_deg:
            if is_coasting:
                active_action_label = f"⚠️ LANE COASTING ({lane_res.lost_frames}/{lane_detector.lost_max_frames}): Straight"
            else:
                active_action_label = f"🛣️ LANE LOCKED ({lane_res.status}): Straight"
            if arduino.motors_enabled:
                arduino.drive(165, 250)
        else:
            steer_cmd = int(round(delta))
            steer_dir = "RIGHT" if steer_cmd > 0 else "LEFT"
            if is_coasting:
                active_action_label = f"⚠️ LANE COASTING ({lane_res.lost_frames}/{lane_detector.lost_max_frames}): Steer {steer_dir} ({abs(steer_cmd)}°)"
            else:
                active_action_label = f"🛣️ LANE STEER {steer_dir} ({abs(steer_cmd)}°)"
            if arduino.motors_enabled:
                arduino.turn(steer_cmd, 145)
        return

    # ── AUTONOMOUS NAVIGATION MODE ───────────────────────────────────────────
    if obstacle_ahead:
        active_action_label = "🛑 PI_STOP: Obstacle Imminent (Camera)"
        arduino.pi_stop()
        return

    if arduino.pi_latched and not obstacle_ahead:
        active_action_label = "✅ PI_GO: Path Cleared (Camera)"
        arduino.pi_go()

    # 2. Ultrasonic Sensor Active Processing
    us_l = arduino.latest_sonar["left"]
    us_r = arduino.latest_sonar["right"]
    v_l = is_valid_us(us_l)
    v_r = is_valid_us(us_r)

    # 2a. Ultrasonic Critical Safety Auto-Brake (< EMERGENCY_STOP_DIST_CM)
    if (v_l and us_l < EMERGENCY_STOP_DIST_CM) or (v_r and us_r < EMERGENCY_STOP_DIST_CM):
        active_action_label = f"🛑 US AUTO-BRAKE! (L:{us_l:.0f}cm R:{us_r:.0f}cm)"
        arduino.stop()
        return

    # 2b. Front Corridor Blocked: Both sensors detect obstacle (< US_AVOID_DIST_CM)
    if v_l and v_r and us_l < US_AVOID_DIST_CM and us_r < US_AVOID_DIST_CM:
        active_action_label = f"⚠️ US OBSTACLE AHEAD (L:{us_l:.0f}cm R:{us_r:.0f}cm) -> STOP"
        arduino.stop()
        return

    # 2c. Ultrasonic Directional Avoidance: Left obstacle closer -> Steer RIGHT
    if v_l and us_l < US_AVOID_DIST_CM and (not v_r or us_l < us_r - 6.0):
        active_action_label = f"🔀 US AVOID: Left Obs ({us_l:.0f}cm) -> Steering RIGHT"
        arduino.pi_avoid("RIGHT")
        if arduino.motors_enabled:
            arduino.turn(35, 160)
        return

    # 2d. Ultrasonic Directional Avoidance: Right obstacle closer -> Steer LEFT
    if v_r and us_r < US_AVOID_DIST_CM and (not v_l or us_r < us_l - 6.0):
        active_action_label = f"🔀 US AVOID: Right Obs ({us_r:.0f}cm) -> Steering LEFT"
        arduino.pi_avoid("LEFT")
        if arduino.motors_enabled:
            arduino.turn(-35, 160)
        return

    # 3. Camera Pothole Avoidance (GPU Fine-Tuned YOLO, Lower Camera Section)
    danger_left = sum(1.0 for p in pot_dets if p.get("is_pothole") and (p["bbox"][0] + p["bbox"][2]) / 2 < frame_w * 0.5)
    danger_right = sum(1.0 for p in pot_dets if p.get("is_pothole") and (p["bbox"][0] + p["bbox"][2]) / 2 >= frame_w * 0.5)

    if danger_left > 0 or danger_right > 0:
        if danger_left > danger_right:
            active_action_label = "🔀 AVOIDING POTHOLE: Steering RIGHT"
            arduino.pi_avoid("RIGHT")
            if arduino.motors_enabled:
                arduino.turn(30, 160)
        else:
            active_action_label = "🔀 AVOIDING POTHOLE: Steering LEFT"
            arduino.pi_avoid("LEFT")
            if arduino.motors_enabled:
                arduino.turn(-30, 160)
        return

    # 4. Clear Path Cruising
    if arduino.motors_enabled:
        if (v_l and us_l < US_SLOW_DIST_CM) or (v_r and us_r < US_SLOW_DIST_CM):
            active_action_label = f"🟡 CRUISING CAUTIOUSLY (PWM {SLOW_SPEED}) [US Near]"
            arduino.drive(SLOW_SPEED, 250)
        else:
            active_action_label = f"🟢 CRUISING FORWARD (PWM {CRUISE_SPEED})"
            arduino.drive(CRUISE_SPEED, 250)
    else:
        active_action_label = "⏸️ MOTORS DISABLED (Bench Test Safe)"

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Capstone Model Car — Autonomous Perception & Control</title>
  <style>
    :root {
      --bg: #000000;
      --card-glass: rgba(18, 22, 30, 0.75);
      --border-glass: rgba(255, 255, 255, 0.12);
      --accent: #58a6ff; --success: #3fb950; --danger: #f85149; --warning: #d29922; --text: #f0f6fc; --muted: #8b949e;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html, body {
      background: var(--bg); color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      width: 100vw; height: 100vh; overflow: hidden;
      display: flex; justify-content: center; align-items: center; user-select: none;
    }
    #viewport {
      position: absolute; top: 0; left: 0; width: 100vw; height: 100vh;
      display: flex; justify-content: center; align-items: center; background: #000;
    }
    img#stream { width: 100%; height: 100%; object-fit: contain; display: block; }
    header {
      position: absolute; top: 14px; left: 50%; transform: translateX(-50%);
      background: var(--card-glass); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
      border: 1px solid var(--border-glass); border-radius: 30px; padding: 6px 18px;
      display: flex; align-items: center; gap: 14px; box-shadow: 0 8px 32px rgba(0,0,0,0.6); z-index: 100;
    }
    .badge { padding: 3px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 700; }
    .badge-daylight { background: rgba(227, 179, 65, 0.2); color: #e3b341; border: 1px solid #e3b341; }
    .badge-motors-on { background: rgba(63, 185, 80, 0.2); color: var(--success); border: 1px solid var(--success); }
    .badge-motors-off { background: rgba(248, 81, 73, 0.2); color: var(--danger); border: 1px solid var(--danger); }
    .badge-invert-on { background: rgba(88, 166, 255, 0.2); color: var(--accent); border: 1px solid var(--accent); }
    .badge-invert-off { background: rgba(139, 148, 158, 0.2); color: var(--muted); border: 1px solid var(--muted); }
    .badge-override-off { background: rgba(139, 148, 158, 0.2); color: var(--muted); border: 1px solid var(--muted); }
    .badge-override-on { background: rgba(227, 179, 65, 0.35); color: #e3b341; border: 1px solid #e3b341; animation: pulse-override 1.8s infinite; font-weight: 800; }
    @keyframes pulse-override { 0% { opacity: 0.85; } 50% { opacity: 1; box-shadow: 0 0 12px rgba(227, 179, 65, 0.55); } 100% { opacity: 0.85; } }
    .badge-us { background: rgba(63, 185, 80, 0.2); color: var(--success); border: 1px solid var(--success); }
    .badge-mode-auto { background: rgba(88, 166, 255, 0.2); color: #58a6ff; border: 1px solid #58a6ff; cursor: pointer; }
    .badge-mode-manual { background: rgba(63, 185, 80, 0.25); color: #56d364; border: 1px solid #56d364; cursor: pointer; }

    .controls-dock {
      position: absolute; bottom: 16px; left: 50%; transform: translateX(-50%);
      background: var(--card-glass); backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
      border: 1px solid var(--border-glass); border-radius: 28px; padding: 6px 14px;
      display: flex; align-items: center; gap: 8px; box-shadow: 0 8px 32px rgba(0,0,0,0.6); z-index: 100;
      transition: all 0.3s ease;
    }
    .controls-dock.hidden { opacity: 0; pointer-events: none; transform: translateX(-50%) translateY(20px); }
    button {
      background: rgba(255, 255, 255, 0.08); color: var(--text); border: 1px solid var(--border-glass);
      padding: 7px 14px; border-radius: 18px; font-size: 0.8rem; font-weight: 600; cursor: pointer;
      display: inline-flex; align-items: center; gap: 6px;
    }
    button:hover { background: rgba(255, 255, 255, 0.18); }
    button.btn-active { background: #1f6feb; border-color: #58a6ff; }
    button.btn-warning-active { background: rgba(227, 179, 65, 0.35); border-color: #e3b341; color: #fff; }
    button.btn-danger { background: rgba(248, 81, 73, 0.25); border-color: var(--danger); color: #ff7b72; }
    button.btn-success { background: rgba(63, 185, 80, 0.25); border-color: var(--success); color: #56d364; }
    .shortcut-tip { position: absolute; bottom: 6px; right: 12px; font-size: 0.7rem; color: var(--muted); opacity: 0.6; }

    /* Remote Control Glass Panel */
    .remote-panel {
      position: absolute; right: 20px; bottom: 74px; z-index: 100;
      background: var(--card-glass); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
      border: 1px solid var(--border-glass); border-radius: 20px; padding: 12px 14px;
      display: flex; flex-direction: column; align-items: center; gap: 8px;
      box-shadow: 0 12px 40px rgba(0,0,0,0.7); width: 194px;
      transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .remote-panel.minimized {
      transform: translateY(20px) scale(0.9);
      opacity: 0; pointer-events: none;
    }
    .remote-header {
      width: 100%; display: flex; justify-content: space-between; align-items: center;
      font-size: 0.75rem; font-weight: 700; color: var(--accent);
    }
    .speed-pills {
      display: flex; gap: 4px; background: rgba(255,255,255,0.06); border-radius: 12px; padding: 2px;
    }
    .btn-speed {
      padding: 3px 7px; font-size: 0.65rem; border-radius: 10px; border: none; background: transparent; color: var(--muted); cursor: pointer; font-weight: 700;
    }
    .btn-speed.active {
      background: var(--accent); color: #fff;
    }
    .dpad-grid {
      display: grid; grid-template-columns: repeat(3, 48px); grid-template-rows: repeat(3, 48px); gap: 6px;
      justify-content: center; align-items: center;
    }
    .dpad-btn {
      width: 48px; height: 48px; padding: 0; border-radius: 14px;
      display: flex; flex-direction: column; justify-content: center; align-items: center;
      font-size: 1.05rem; font-weight: 700; border: 1px solid var(--border-glass);
      background: rgba(255, 255, 255, 0.06); color: var(--text); cursor: pointer;
      transition: all 0.1s ease; touch-action: manipulation;
    }
    .dpad-btn .key-hint {
      font-size: 0.58rem; color: var(--muted); font-weight: 600; margin-top: -2px;
    }
    .dpad-btn:hover { background: rgba(255, 255, 255, 0.16); }
    .dpad-btn:active, .dpad-btn.active {
      transform: scale(0.92);
    }
    .dpad-btn.btn-drive-active {
      background: rgba(63, 185, 80, 0.4) !important; border-color: var(--success) !important; color: #fff !important;
      box-shadow: 0 0 16px rgba(63, 185, 80, 0.55);
    }
    .dpad-btn.btn-steer-active {
      background: rgba(88, 166, 255, 0.4) !important; border-color: var(--accent) !important; color: #fff !important;
      box-shadow: 0 0 16px rgba(88, 166, 255, 0.55);
    }
    .dpad-btn.btn-rev-active {
      background: rgba(227, 179, 65, 0.4) !important; border-color: #e3b341 !important; color: #fff !important;
      box-shadow: 0 0 16px rgba(227, 179, 65, 0.55);
    }
    .dpad-btn.dpad-stop {
      background: rgba(248, 81, 73, 0.15); border-color: rgba(248, 81, 73, 0.4); color: #ff7b72;
    }
    .dpad-btn.dpad-stop:active, .dpad-btn.dpad-stop.active {
      background: rgba(248, 81, 73, 0.5) !important; border-color: var(--danger) !important; color: #fff !important;
      box-shadow: 0 0 16px rgba(248, 81, 73, 0.6);
    }
    .remote-footer {
      font-size: 0.65rem; color: var(--muted); text-align: center;
    }

    .modal {
      display: none; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
      background: rgba(0,0,0,0.85); z-index: 1000; justify-content: center; align-items: center; padding: 20px;
    }
    .modal.show { display: flex; }
    .modal-box { background: #161b22; border: 1px solid #30363d; border-radius: 10px; max-width: 680px; width: 100%; padding: 16px; }
    .modal-box img { width: 100%; border-radius: 6px; margin-top: 10px; display: block; }
  </style>
</head>
<body>
  <div id="viewport">
    <img id="stream" src="stream.mjpg" alt="Autonomous Perception Stream">
  </div>

  <header id="topbar">
    <div style="font-weight:700; font-size:0.85rem">🏎️ NMIMS MPSTME Capstone</div>
    <div class="badge badge-mode-auto" id="mode-badge" onclick="toggleControlMode()" title="Click to toggle Mode (C)">MODE: AUTO 🤖</div>
    <div class="badge badge-override-off" id="override-badge" onclick="toggleOverride()" style="cursor:pointer" title="Toggle Manual Obstacle Override (O)">OVERRIDE: OFF 🛡️</div>
    <div class="badge badge-daylight" id="awb-badge">AWB: DAYLIGHT ☀️</div>
    <div class="badge badge-motors-off" id="motors-badge">MOTORS: OFF ⏸️</div>
    <div class="badge badge-invert-on" id="invert-badge">POLARITY: INVERTED (FWD FIX) 🔄</div>
    <div class="badge badge-us" id="sonar-badge">US: L=--- R=---</div>
  </header>

  <div class="controls-dock" id="dock">
    <button id="btn-motors" class="btn-danger" onclick="toggleMotors()">🚗 Enable Motors (M)</button>
    <button id="btn-mode" onclick="toggleControlMode()">🎮 Mode: Manual (C)</button>
    <button id="btn-lane" onclick="setMode('lane')">🛣️ Lane Mode</button>
    <button id="btn-lane-color" onclick="cycleLaneColor()">🎨 Lane: Dark</button>
    <button id="btn-override" onclick="toggleOverride()">🔓 Override Obstacles (O)</button>
    <button id="btn-remote" class="btn-active" onclick="toggleRemoteWidget()">🕹️ Remote</button>
    <button id="btn-invert" class="btn-active" onclick="toggleInvertMotors()">🔄 Polarity: Inverted (I)</button>
    <button class="btn-active" id="btn-daylight" onclick="setAwb('daylight')">☀️ Daylight</button>
    <button onclick="setAwb('auto')">🔄 Auto</button>
    <button onclick="setAwb('indoor')">💡 Indoor</button>
    <button id="btn-pothole-mode" class="btn-active" onclick="togglePotholeMode()">🕳️ Potholes Only</button>
    <button onclick="emergencyStop()" class="btn-danger">🛑 E-STOP (Space)</button>
    <button onclick="toggleFullscreen()">⛶ Fullscreen (F)</button>
    <button onclick="openRef()">📷 Ref</button>
    <button onclick="toggleDock()" style="padding:7px 10px">👁️</button>
  </div>

  <!-- Floating Remote Control Widget (D-Pad) -->
  <div class="remote-panel" id="remote-panel">
    <div class="remote-header">
      <span>🎮 Remote Control</span>
      <div class="speed-pills">
        <button class="btn-speed" onclick="setSpeed(130)" id="spd-130">130</button>
        <button class="btn-speed active" onclick="setSpeed(185)" id="spd-185">185</button>
        <button class="btn-speed" onclick="setSpeed(235)" id="spd-235">235</button>
      </div>
    </div>
    <div class="dpad-grid">
      <div></div>
      <button class="dpad-btn" id="dpad-up" onmousedown="startDrive('forward')" onmouseup="stopDrive()" onmouseleave="stopDrive()" ontouchstart="startDrive('forward')" ontouchend="stopDrive()">▲<span class="key-hint">W</span></button>
      <div></div>
      <button class="dpad-btn" id="dpad-left" onmousedown="startDrive('left')" onmouseup="stopDrive()" onmouseleave="stopDrive()" ontouchstart="startDrive('left')" ontouchend="stopDrive()">◀<span class="key-hint">A</span></button>
      <button class="dpad-btn dpad-stop" id="dpad-stop" onclick="stopDrive()">■<span class="key-hint">Space</span></button>
      <button class="dpad-btn" id="dpad-right" onmousedown="startDrive('right')" onmouseup="stopDrive()" onmouseleave="stopDrive()" ontouchstart="startDrive('right')" ontouchend="stopDrive()">▶<span class="key-hint">D</span></button>
      <div></div>
      <button class="dpad-btn" id="dpad-down" onmousedown="startDrive('reverse')" onmouseup="stopDrive()" onmouseleave="stopDrive()" ontouchstart="startDrive('reverse')" ontouchend="stopDrive()">▼<span class="key-hint">S</span></button>
      <div></div>
    </div>
    <div class="remote-footer">
      <span>Drive: <b>WASD</b> or <b>Arrow Keys</b></span>
    </div>
  </div>

  <div class="shortcut-tip"><b>WASD/Arrows</b> Drive &bull; <b>C</b> Mode &bull; <b>O</b> Override Obstacles &bull; <b>1/2/3</b> Speed &bull; <b>Space</b> E-Stop &bull; <b>M</b> Motors &bull; <b>I</b> Invert &bull; <b>H</b> Dock</div>

  <div class="modal" id="modal" onclick="closeRef()">
    <div class="modal-box" onclick="event.stopPropagation()">
      <div style="display:flex; justify-content:space-between; font-weight:600">
        <span>Reference Image: daylight.jpg</span>
        <span style="cursor:pointer" onclick="closeRef()">&times;</span>
      </div>
      <img src="daylight.jpg" alt="daylight.jpg reference">
    </div>
  </div>

  <script>
    let currentMode = 'autonomous';
    let currentSpeed = 185;
    let driveInterval = null;
    let activeAction = null;
    let manualOverride = false;
    let keysPressed = new Set();

    function toggleFullscreen() {
      if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch(() => {});
      } else {
        document.exitFullscreen().catch(() => {});
      }
    }

    function toggleMotors() {
      fetch('/toggle_motors').then(r => r.json()).then(d => {
        const btn = document.getElementById('btn-motors');
        const badge = document.getElementById('motors-badge');
        if (d.motors_enabled) {
          btn.innerText = '🛑 Disable Motors (M)';
          btn.className = 'btn-success';
          badge.innerText = 'MOTORS: ACTIVE ⚡';
          badge.className = 'badge badge-motors-on';
        } else {
          btn.innerText = '🚗 Enable Motors (M)';
          btn.className = 'btn-danger';
          badge.innerText = 'MOTORS: OFF ⏸️';
          badge.className = 'badge badge-motors-off';
        }
      });
    }

    function toggleInvertMotors() {
      fetch('/toggle_invert_motors').then(r => r.json()).then(d => {
        const btn = document.getElementById('btn-invert');
        const badge = document.getElementById('invert-badge');
        if (d.invert_motors) {
          btn.innerText = '🔄 Polarity: Inverted (I)';
          btn.className = 'btn-active';
          badge.innerText = 'POLARITY: INVERTED (FWD FIX) 🔄';
          badge.className = 'badge badge-invert-on';
        } else {
          btn.innerText = '🔄 Polarity: Normal (I)';
          btn.className = '';
          badge.innerText = 'POLARITY: NORMAL ➡️';
          badge.className = 'badge badge-invert-off';
        }
      });
    }

    function toggleOverride() {
      fetch('/toggle_override').then(r => r.json()).then(d => {
        updateOverrideUI(d.override);
      });
    }

    function updateOverrideUI(enabled) {
      manualOverride = enabled;
      const b = document.getElementById('override-badge');
      const btn = document.getElementById('btn-override');
      if (enabled) {
        if (b) { b.innerText = 'OVERRIDE: ON 🔓'; b.className = 'badge badge-override-on'; }
        if (btn) { btn.innerText = '🛡️ Override: ON (O)'; btn.className = 'btn-warning-active'; }
      } else {
        if (b) { b.innerText = 'OVERRIDE: OFF 🛡️'; b.className = 'badge badge-override-off'; }
        if (btn) { btn.innerText = '🔓 Override Obstacles (O)'; btn.className = ''; }
      }
    }

    function setSpeed(spd) {
      currentSpeed = spd;
      fetch('/set_speed?speed=' + spd).then(r => r.json()).then(d => {
        document.querySelectorAll('.btn-speed').forEach(b => b.classList.remove('active'));
        const activeBtn = document.getElementById('spd-' + spd);
        if (activeBtn) activeBtn.classList.add('active');
      });
    }

    function toggleControlMode() {
      fetch('/toggle_control_mode').then(r => r.json()).then(d => {
        updateModeUI(d.mode);
      });
    }

    function setMode(mode) {
      fetch('/set_mode?mode=' + mode).then(r => r.json()).then(d => {
        updateModeUI(d.mode);
      });
    }

    function cycleLaneColor() {
      const colors = ['dark', 'yellow', 'blue', 'white'];
      const btn = document.getElementById('btn-lane-color');
      let curr = (btn ? btn.getAttribute('data-color') : '') || 'dark';
      let nextIdx = (colors.indexOf(curr) + 1) % colors.length;
      let nextCol = colors[nextIdx];
      fetch('/set_lane_color?color=' + nextCol).then(r => r.json()).then(d => {
        if (btn) {
          btn.setAttribute('data-color', d.color);
          btn.innerText = '🎨 Lane: ' + d.color.toUpperCase();
        }
      });
    }

    function updateModeUI(mode) {
      currentMode = mode;
      const mb = document.getElementById('mode-badge');
      const btnMode = document.getElementById('btn-mode');
      if (mode === 'manual') {
        mb.innerText = 'MODE: MANUAL REMOTE 🎮';
        mb.className = 'badge badge-mode-manual';
        btnMode.innerText = '🤖 Mode: Auto (C)';
        btnMode.className = 'btn-active';
      } else if (mode === 'lane') {
        mb.innerText = 'MODE: LANE 🛣️';
        mb.className = 'badge badge-mode-auto';
        btnMode.innerText = '🎮 Mode: Manual (C)';
        btnMode.className = 'btn-active';
      } else {
        mb.innerText = 'MODE: AUTO 🤖';
        mb.className = 'badge badge-mode-auto';
        btnMode.innerText = '🛣️ Mode: Lane (C)';
        btnMode.className = '';
      }
    }

    function toggleRemoteWidget() {
      const rp = document.getElementById('remote-panel');
      rp.classList.toggle('minimized');
      const btn = document.getElementById('btn-remote');
      if (rp.classList.contains('minimized')) {
        btn.classList.remove('btn-active');
      } else {
        btn.classList.add('btn-active');
      }
    }

    function sendDriveCmd(action) {
      fetch(`/control?action=${action}&speed=${currentSpeed}`).catch(() => {});
    }

    function startDrive(action) {
      if (currentMode !== 'manual') {
        fetch('/set_mode?mode=manual').then(r => r.json()).then(d => updateModeUI(d.mode));
      }
      if (activeAction === action && driveInterval) return;
      activeAction = action;
      highlightDpad(action, true);
      sendDriveCmd(action);
      clearInterval(driveInterval);
      driveInterval = setInterval(() => {
        sendDriveCmd(action);
      }, 140);
    }

    function stopDrive() {
      clearInterval(driveInterval);
      driveInterval = null;
      if (activeAction) {
        highlightDpad(activeAction, false);
      }
      activeAction = null;
      fetch('/control?action=stop').catch(() => {});
    }

    function highlightDpad(action, on) {
      const map = {
        'forward': { id: 'dpad-up', cls: 'btn-drive-active' },
        'reverse': { id: 'dpad-down', cls: 'btn-rev-active' },
        'left': { id: 'dpad-left', cls: 'btn-steer-active' },
        'right': { id: 'dpad-right', cls: 'btn-steer-active' },
        'stop': { id: 'dpad-stop', cls: 'active' }
      };
      const info = map[action];
      if (info) {
        const el = document.getElementById(info.id);
        if (el) {
          if (on) el.classList.add(info.cls);
          else el.classList.remove(info.cls);
        }
      }
    }

    function emergencyStop() {
      stopDrive();
      fetch('/emergency_stop').then(r => r.json()).then(d => {
        const btn = document.getElementById('btn-motors');
        const badge = document.getElementById('motors-badge');
        btn.innerText = '🚗 Enable Motors (M)';
        btn.className = 'btn-danger';
        badge.innerText = 'MOTORS: OFF (E-STOP) 🛑';
        badge.className = 'badge badge-motors-off';
      });
    }

    function togglePotholeMode() {
      fetch('/toggle_pothole_mode').then(r => r.json()).then(d => {
        const btn = document.getElementById('btn-pothole-mode');
        if (d.pothole_only) {
          btn.innerText = '🕳️ Potholes Only';
          btn.className = 'btn-active';
        } else {
          btn.innerText = '🛣️ Potholes + Cracks';
          btn.className = '';
        }
      });
    }

    function setAwb(mode) {
      fetch('/set_awb?mode=' + mode).then(r => r.json()).then(d => {
        if (d.status === 'ok') {
          document.getElementById('awb-badge').innerText = 'AWB: ' + d.mode.toUpperCase() + (d.mode === 'daylight' ? ' ☀️' : '');
          document.querySelectorAll('.controls-dock button').forEach(b => {
            if (b.innerText.toLowerCase().includes(mode)) b.classList.add('btn-active');
            else if (b.innerText.includes('☀️') || b.innerText.includes('🔄') || b.innerText.includes('💡')) b.classList.remove('btn-active');
          });
        }
      });
    }

    function toggleDock() {
      const dock = document.getElementById('dock');
      const topbar = document.getElementById('topbar');
      dock.classList.toggle('hidden');
      topbar.style.opacity = dock.classList.contains('hidden') ? '0' : '1';
    }

    function openRef() { document.getElementById('modal').classList.add('show'); }
    function closeRef() { document.getElementById('modal').classList.remove('show'); }

    // Live Telemetry Poller
    function pollTelemetry() {
      fetch('/telemetry').then(r => r.json()).then(d => {
        const sb = document.getElementById('sonar-badge');
        if (sb && d.sonar) {
          const l = (d.sonar.left <= 1.0 || d.sonar.left >= 850.0) ? 'CLR' : d.sonar.left.toFixed(0) + 'cm';
          const r = (d.sonar.right <= 1.0 || d.sonar.right >= 850.0) ? 'CLR' : d.sonar.right.toFixed(0) + 'cm';
          sb.innerText = `US: L=${l}  R=${r}`;
          if ((d.sonar.left > 1.0 && d.sonar.left < 25.0) || (d.sonar.right > 1.0 && d.sonar.right < 25.0)) {
            sb.style.background = 'rgba(248, 81, 73, 0.35)';
            sb.style.borderColor = 'var(--danger)';
            sb.style.color = '#ff7b72';
          } else if ((d.sonar.left > 1.0 && d.sonar.left < 45.0) || (d.sonar.right > 1.0 && d.sonar.right < 45.0)) {
            sb.style.background = 'rgba(227, 179, 65, 0.35)';
            sb.style.borderColor = '#e3b341';
            sb.style.color = '#e3b341';
          } else {
            sb.style.background = 'rgba(63, 185, 80, 0.2)';
            sb.style.borderColor = 'var(--success)';
            sb.style.color = 'var(--success)';
          }
        }
        if (d.control_mode && d.control_mode !== currentMode) {
          updateModeUI(d.control_mode);
        }
        if (d.manual_override_obstacles !== undefined && d.manual_override_obstacles !== manualOverride) {
          updateOverrideUI(d.manual_override_obstacles);
        }
      }).catch(() => {});
    }
    setInterval(pollTelemetry, 300);

    // Laptop Keyboard Event Listeners for WASD / Arrow Keys & O for Override
    window.addEventListener('keydown', (e) => {
      const k = e.key.toLowerCase();
      if (e.target.tagName === 'INPUT') return;

      if (k === 'w' || e.key === 'ArrowUp') {
        e.preventDefault();
        keysPressed.add('up');
        startDrive('forward');
      } else if (k === 's' || e.key === 'ArrowDown') {
        e.preventDefault();
        keysPressed.add('down');
        startDrive('reverse');
      } else if (k === 'a' || e.key === 'ArrowLeft') {
        e.preventDefault();
        keysPressed.add('left');
        startDrive('left');
      } else if (k === 'd' || e.key === 'ArrowRight') {
        e.preventDefault();
        keysPressed.add('right');
        startDrive('right');
      } else if (k === ' ' || e.key === 'Spacebar') {
        e.preventDefault();
        emergencyStop();
      } else if (k === 'c') {
        toggleControlMode();
      } else if (k === 'o') {
        toggleOverride();
      } else if (k === '1') {
        setSpeed(130);
      } else if (k === '2') {
        setSpeed(185);
      } else if (k === '3') {
        setSpeed(235);
      } else if (k === 'f') {
        toggleFullscreen();
      } else if (k === 'm') {
        toggleMotors();
      } else if (k === 'i') {
        toggleInvertMotors();
      } else if (k === 'h') {
        toggleDock();
      } else if (k === 'p') {
        togglePotholeMode();
      }
    });

    window.addEventListener('keyup', (e) => {
      const k = e.key.toLowerCase();
      if (k === 'w' || e.key === 'ArrowUp') keysPressed.delete('up');
      if (k === 's' || e.key === 'ArrowDown') keysPressed.delete('down');
      if (k === 'a' || e.key === 'ArrowLeft') keysPressed.delete('left');
      if (k === 'd' || e.key === 'ArrowRight') keysPressed.delete('right');

      if (keysPressed.size === 0 && activeAction) {
        stopDrive();
      }
    });
  </script>
</body>
</html>
"""

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

def draw_boxes(frame, result, color, prefix=""):
    if result is None or getattr(result, "boxes", None) is None:
        return
    w = frame.shape[1]
    f_scale = 0.50 if w >= 480 else 0.38
    thick = 2 if w >= 480 else 1
    for b in result.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        label = f"{prefix}{result.names[int(b.cls[0])]} {float(b.conf[0]):.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        cv2.putText(frame, label, (x1, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, f_scale, color, thick)

def detect_loop(cam, output):
    global current_awb, swap_rb, POTHOLE_ONLY, hardware_rotated
    n = 0
    last_pot_dets = []
    confirmed_pot_dets = []
    pothole_streak = 0
    fps = 0.0
    t_prev = time.time()

    print("🚀 Parallel Detection & Control Loop Active! (Lower-Section Printout Pothole Tuning)")
    while True:
        arduino.read_telemetry()

        if cam is not None:
            frame_raw = cam.capture_array()
            if swap_rb:
                frame = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2BGR)
            else:
                frame = frame_raw.copy()
        else:
            frame = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)

        if not hardware_rotated:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        h, w = frame.shape[:2]

        # 1. Obstacle Detection (YOLOv10n)
        r = None
        if obs_model is not None:
            try:
                r = obs_model.predict(frame, imgsz=IMGSZ, conf=CONF_OBSTACLE,
                                      classes=OBSTACLE_CLASSES, verbose=False)[0]
            except Exception:
                r = None

        # 2. Pothole Detection with GPU Printout-Fine-Tuned Model (Lower Camera Section)
        if n % POTHOLE_EVERY == 0:
            raw_pot_dets = []
            if pot_model is not None:
                try:
                    # Run on full 4:3 frame (matching training aspect ratio) and filter boxes to lower section
                    frame_area = float(h * w)
                    res_p = pot_model.predict(frame, imgsz=POTHOLE_IMGSZ, conf=CONF_POTHOLE, verbose=False)[0]

                    for box in res_p.boxes:
                        cls_id = int(box.cls[0])
                        name = res_p.names.get(cls_id, "pothole").lower()
                        c_val = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                        is_pothole = (cls_id == 0 or "pothole" in name or "d40" in name)
                        if POTHOLE_ONLY and not is_pothole:
                            continue

                        # Lower-section & bounding-box geometry sanity filter
                        bw = x2 - x1
                        bh = y2 - y1
                        yc = (y1 + y2) / 2.0
                        box_area = float(bw * bh)

                        # 1. Must be in the lower section of the camera frame
                        if yc < h * POTHOLE_LOWER_Y_MIN_FRAC or y2 < h * 0.45:
                            continue
                        # 2. Reject giant whole-printout border boxes or tiny specks
                        if bw > w * 0.62 or bh > h * 0.48 or box_area > frame_area * 0.22:
                            continue
                        if bw < w * 0.025 or bh < h * 0.02 or box_area < frame_area * 0.0015:
                            continue
                        # 3. Aspect ratio check
                        aspect = bw / max(bh, 1.0)
                        if not (0.25 <= aspect <= 4.5):
                            continue

                        lbl = f"POTHOLE (D40) {c_val:.2f}" if is_pothole else f"CRACK {c_val:.2f}"
                        raw_pot_dets.append({
                            "bbox": [x1, y1, x2, y2],
                            "conf": c_val,
                            "label": lbl,
                            "is_pothole": is_pothole
                        })
                except Exception:
                    pass
            else:
                # Only use classical CV fallback if no YOLO pothole model is loaded
                raw_pot_dets = cv_detect_potholes(frame)

            last_pot_dets = raw_pot_dets
            if any(p.get("is_pothole") for p in raw_pot_dets):
                pothole_streak += 1
            else:
                pothole_streak = 0

            # Temporal persistence filter: require POTHOLE_PERSIST_FRAMES consecutive scans for steering
            confirmed_pot_dets = last_pot_dets if pothole_streak >= POTHOLE_PERSIST_FRAMES else []

        # 3. Decision Engine (uses temporally confirmed potholes to prevent 1-frame false swerves)
        obs_boxes = r.boxes if r is not None else []
        evaluate_autonomous_action(w, h, obs_boxes, confirmed_pot_dets, frame=frame)

        # 4. Render Lane Corridor & Curves if in Lane Mode
        if control_mode == "lane" and last_lane_res is not None:
            if last_lane_res.corridor_pts is not None and len(last_lane_res.corridor_pts) >= 4:
                overlay = frame.copy()
                cv2.fillPoly(overlay, [last_lane_res.corridor_pts], (40, 180, 50))
                cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
            if last_lane_res.left_pts is not None and len(last_lane_res.left_pts) >= 2:
                cv2.polylines(frame, [last_lane_res.left_pts], False, (255, 255, 0), thick + 1)
            if last_lane_res.right_pts is not None and len(last_lane_res.right_pts) >= 2:
                cv2.polylines(frame, [last_lane_res.right_pts], False, (0, 215, 255), thick + 1)
            if last_lane_res.center_pts is not None and len(last_lane_res.center_pts) >= 2:
                cv2.polylines(frame, [last_lane_res.center_pts], False, (255, 0, 255), thick)
            if last_lane_res.lookahead_point is not None and (last_lane_res.status != "LANE_LOST" or last_lane_res.center_coeffs is not None):
                lx, ly = last_lane_res.lookahead_point
                cv2.drawMarker(frame, (lx, ly), (0, 0, 255), cv2.MARKER_CROSS, 14, thick + 1)
                cv2.circle(frame, (lx, ly), 6, (0, 255, 255), thick)

        # 5. Render Obstacle Boxes (Green)
        if r is not None:
            draw_boxes(frame, r, (0, 255, 0))

        # 6. Render Lower-Section Pothole Scan Horizon & Pothole Boxes
        f_scale = 0.50 if w >= 480 else 0.38
        hud_scale1 = 0.65 if w >= 480 else 0.45
        hud_scale2 = 0.55 if w >= 480 else 0.40
        thick = 2 if w >= 480 else 1

        roi_line_y = int(h * POTHOLE_LOWER_Y_MIN_FRAC)
        cv2.line(frame, (0, roi_line_y), (w, roi_line_y), (75, 75, 75), 1)

        for p in last_pot_dets:
            x1, y1, x2, y2 = p["bbox"]
            color = (0, 69, 255) if p.get("is_pothole") else (255, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
            cv2.putText(frame, p["label"], (x1, max(14, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, f_scale, color, thick)

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now

        # HUD Overlays
        ov_text = " [OVERRIDE]" if (control_mode == "manual" and manual_override_obstacles) else ""
        mode_tag = f"[{control_mode.upper()}{ov_text}]"
        cv2.putText(frame, f"{fps:.1f} FPS | {mode_tag} | {current_awb.upper()} AWB", (10, 20 if w < 480 else 26),
                    cv2.FONT_HERSHEY_SIMPLEX, hud_scale1, (0, 255, 255), thick)
        action_col = (0, 255, 0) if ("CRUISING" in active_action_label or "MANUAL" in active_action_label or "LANE" in active_action_label) else (0, 165, 255)
        if "BRAKE" in active_action_label or "STOP" in active_action_label:
            action_col = (0, 0, 255)
        cv2.putText(frame, f"Action: {active_action_label}", (10, 38 if w < 480 else 52),
                    cv2.FONT_HERSHEY_SIMPLEX, hud_scale1, action_col, thick)

        if control_mode == "lane" and last_lane_res is not None:
            if last_lane_res.status != "LANE_LOST":
                lane_str = f"LANE: {last_lane_res.status} | Off: {last_lane_res.lateral_offset_pct:+.1f}% | δ: {last_lane_res.steer_angle_deg:+.0f}°"
                cv2.putText(frame, lane_str, (10, 56 if w < 480 else 76),
                            cv2.FONT_HERSHEY_SIMPLEX, hud_scale1, (0, 255, 200), thick)
            elif last_lane_res.center_coeffs is not None:
                lane_str = f"LANE: COASTING ({last_lane_res.lost_frames}) | Off: {last_lane_res.lateral_offset_pct:+.1f}% | δ: {last_lane_res.steer_angle_deg:+.0f}°"
                cv2.putText(frame, lane_str, (10, 56 if w < 480 else 76),
                            cv2.FONT_HERSHEY_SIMPLEX, hud_scale1, (0, 165, 255), thick)

        # Ultrasonic live display
        us_l = arduino.latest_sonar["left"]
        us_r = arduino.latest_sonar["right"]
        us_l_disp = f"{us_l:.0f}cm" if (1.0 < us_l < 850.0) else "CLEAR"
        us_r_disp = f"{us_r:.0f}cm" if (1.0 < us_r < 850.0) else "CLEAR"
        us_str = f"US: L={us_l_disp} | R={us_r_disp}"

        us_color = (0, 255, 0)
        if (1.0 < us_l < EMERGENCY_STOP_DIST_CM) or (1.0 < us_r < EMERGENCY_STOP_DIST_CM):
            us_color = (0, 0, 255)
            us_str += " [BRAKE]"
        elif (1.0 < us_l < US_AVOID_DIST_CM) or (1.0 < us_r < US_AVOID_DIST_CM):
            us_color = (0, 140, 255)
            us_str += " [AVOID]"
        elif (1.0 < us_l < US_SLOW_DIST_CM) or (1.0 < us_r < US_SLOW_DIST_CM):
            us_color = (0, 255, 255)
            us_str += " [SLOW]"

        ard_str = f"Arduino: {arduino.port_name} [{arduino.latest_state}] | Polarity: {'INVERTED' if arduino.invert_motors else 'NORMAL'}"
        cv2.putText(frame, us_str, (10, h - (22 if w < 480 else 36)),
                    cv2.FONT_HERSHEY_SIMPLEX, hud_scale2, us_color, thick)
        cv2.putText(frame, ard_str, (10, h - (8 if w < 480 else 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, hud_scale2, (200, 200, 200), 1)

        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            output.write(jpg.tobytes())

        n += 1

# ── 7. Web Server Handler ────────────────────────────────────────────────────
class Handler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        global POTHOLE_ONLY, control_mode, manual_speed, manual_action, manual_override_obstacles, lane_color_mode, lane_detector
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path in ('/', '/index.html'):
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        elif path == '/toggle_motors':
            arduino.motors_enabled = not arduino.motors_enabled
            if not arduino.motors_enabled:
                arduino.stop()
            print(f"🚗 Motors toggled: {'ENABLED' if arduino.motors_enabled else 'DISABLED'}")
            resp = json.dumps({"status": "ok", "motors_enabled": arduino.motors_enabled}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        elif path == '/toggle_invert_motors':
            arduino.invert_motors = not arduino.invert_motors
            print(f"🔄 Motor polarity toggled: {'INVERTED (FWD Fix)' if arduino.invert_motors else 'NORMAL'}")
            resp = json.dumps({"status": "ok", "invert_motors": arduino.invert_motors}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        elif path == '/toggle_override':
            manual_override_obstacles = not manual_override_obstacles
            print(f"🔓 Manual obstacle override toggled: {'ENABLED (Driver Override)' if manual_override_obstacles else 'DISABLED (Safety Auto-Brake)'}")
            resp = json.dumps({"status": "ok", "override": manual_override_obstacles}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        elif path == '/emergency_stop':
            arduino.motors_enabled = False
            arduino.stop()
            arduino.pi_stop()
            print("🛑 Emergency Stop triggered!")
            resp = json.dumps({"status": "ok", "motors_enabled": False}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        elif path == '/telemetry':
            data = {
                "sonar": arduino.latest_sonar,
                "state": arduino.latest_state,
                "motors_enabled": arduino.motors_enabled,
                "invert_motors": arduino.invert_motors,
                "control_mode": control_mode,
                "manual_override_obstacles": manual_override_obstacles,
                "manual_speed": manual_speed,
                "manual_action": manual_action,
                "action": active_action_label,
                "lane_status": last_lane_res.status if last_lane_res else "OFF",
                "lane_offset": round(last_lane_res.lateral_offset_pct, 1) if last_lane_res else 0.0,
                "lane_steer": round(last_lane_res.steer_angle_deg, 1) if last_lane_res else 0.0,
                "lane_color": lane_color_mode,
            }
            resp = json.dumps(data).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        elif path == '/toggle_control_mode':
            if control_mode == "autonomous":
                control_mode = "lane"
            elif control_mode == "lane":
                control_mode = "manual"
            else:
                control_mode = "autonomous"
            if control_mode == "manual":
                arduino.stop()
                manual_action = "IDLE"
            else:
                arduino.pi_go()
                manual_action = "IDLE"
            print(f"🎮 Control mode toggled: {control_mode.upper()}")
            resp = json.dumps({"status": "ok", "mode": control_mode}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        elif path == '/set_mode':
            req_mode = query.get('mode', ['autonomous'])[0].lower()
            if req_mode in ('autonomous', 'lane', 'manual'):
                control_mode = req_mode
                if control_mode == "manual":
                    arduino.stop()
                    manual_action = "IDLE"
                else:
                    arduino.pi_go()
                    manual_action = "IDLE"
            print(f"🎮 Control mode set: {control_mode.upper()}")
            resp = json.dumps({"status": "ok", "mode": control_mode}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        elif path == '/set_lane_color':
            col = query.get('color', ['dark'])[0].lower()
            if col in ('dark', 'yellow', 'blue', 'white'):
                lane_color_mode = col
                lane_detector.set_color_mode(col)
            print(f"🎨 Lane color mode set: {lane_color_mode.upper()}")
            resp = json.dumps({"status": "ok", "color": lane_color_mode}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        elif path == '/set_speed':
            try:
                spd = int(query.get('speed', [manual_speed])[0])
                manual_speed = max(100, min(255, spd))
            except Exception:
                pass
            resp = json.dumps({"status": "ok", "speed": manual_speed}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        elif path == '/control':
            action = query.get('action', ['stop'])[0].lower()
            if 'speed' in query:
                try:
                    manual_speed = max(100, min(255, int(query['speed'][0])))
                except Exception:
                    pass

            control_mode = "manual"
            manual_action = action.upper()

            # Ultrasonic safety check
            us_l = arduino.latest_sonar["left"]
            us_r = arduino.latest_sonar["right"]
            front_blocked = (is_valid_us(us_l) and us_l < EMERGENCY_STOP_DIST_CM) or (is_valid_us(us_r) and us_r < EMERGENCY_STOP_DIST_CM)

            if action == 'forward':
                if front_blocked and not manual_override_obstacles:
                    arduino.stop()
                    manual_action = "BLOCKED_BRAKE"
                else:
                    arduino.drive(manual_speed, 350, force=manual_override_obstacles)
            elif action == 'reverse':
                arduino.reverse(manual_speed, 350, force=manual_override_obstacles)
            elif action == 'left':
                arduino.turn(-35, manual_speed, force=manual_override_obstacles)
            elif action == 'right':
                arduino.turn(35, manual_speed, force=manual_override_obstacles)
            elif action == 'stop':
                arduino.stop()
                manual_action = "STOP"

            resp = json.dumps({
                "status": "ok",
                "action": manual_action,
                "mode": control_mode,
                "speed": manual_speed,
                "override": manual_override_obstacles
            }).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        elif path == '/toggle_pothole_mode':
            POTHOLE_ONLY = not POTHOLE_ONLY
            print(f"🕳️ Pothole filtering mode: {'POTHOLES ONLY (D40)' if POTHOLE_ONLY else 'ALL ROAD DAMAGE'}")
            resp = json.dumps({"status": "ok", "pothole_only": POTHOLE_ONLY}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        elif path == '/set_awb':
            mode = query.get('mode', ['daylight'])[0]
            success = set_awb_mode(cam_instance, mode)
            resp = json.dumps({"status": "ok" if success else "error", "mode": current_awb}).encode('utf-8')
            self.send_response(200 if success else 400)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        elif path in ('/daylight.jpg', '/ref_image'):
            ref_path = find_daylight_ref()
            if ref_path and ref_path.exists():
                data = ref_path.read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_error(404, "daylight.jpg not found on Raspberry Pi")

        elif path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(frame)))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                logging.warning('Client stream closed: %s', e)
        else:
            self.send_error(404)

class Server(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

def main():
    global output, cam_instance, hardware_rotated
    logging.basicConfig(level=logging.INFO)

    arduino.auto_connect()

    cam = None
    if Picamera2 is not None:
        try:
            print(f"🎥 Initializing Picamera2 in BGR888 mode ({CAM_WIDTH}x{CAM_HEIGHT}, 180° rotation)...")
            cam = Picamera2()
            cam_instance = cam

            video_config = None
            hardware_rotated = False
            if Transform is not None:
                try:
                    video_config = cam.create_video_configuration(
                        main={"size": (CAM_WIDTH, CAM_HEIGHT), "format": "BGR888"},
                        transform=Transform(hflip=1, vflip=1)
                    )
                    hardware_rotated = True
                    print("🔄 Hardware 180° rotation (hflip + vflip via ISP) enabled.")
                except Exception as e:
                    print(f"⚠️ Could not set hardware transform: {e}")

            if video_config is None:
                video_config = cam.create_video_configuration(main={"size": (CAM_WIDTH, CAM_HEIGHT), "format": "BGR888"})

            cam.configure(video_config)
            cam.start()
            set_awb_mode(cam, "daylight")
        except Exception as e:
            print(f"⚠️ Picamera2 initialization failed: {e}")
            cam = None

    output = StreamingOutput()
    worker = threading.Thread(target=detect_loop, args=(cam, output), daemon=True)
    worker.start()

    port = 8000
    print(f"🚀 Live perception ready at: http://<pi-ip>:{port}")
    try:
        Server(('', port), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Stopping system...")
    finally:
        arduino.stop()
        if cam is not None:
            cam.stop()
            cam.close()
        print("Hardware safely released.")

if __name__ == '__main__':
    main()
