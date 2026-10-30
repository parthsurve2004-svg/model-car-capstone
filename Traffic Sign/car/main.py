"""
main.py — Autonomous RC Car Entry Point & Perception-Control Coordination
==========================================================================
Owned by integration-agent.

Wires together:
  Camera capture (Picamera2 / VideoCapture) → YOLOv10n (NCNN) + classical CV
  → Zone Scorer + Decision Engine → Motor Controller (serial → Arduino Uno)
  → Headless HTTP Live Stream with Diagnostic HUD overlay (port 8080)

Operating Modes:
  autonomous (default):
    Avoid all obstacles. Persons are treated as obstacles.
    Tier 1 Pi Camera Latch (PI_STOP/PI_GO) > Tier 1.5 Cliff Reflex >
    Tier 2 Ultrasonic Avoidance Reflex > Tier 3 Normal Driving.

  follow:
    Lock onto closest person at startup. Follow them with PID steering.
    Non-target obstacles still trigger Tier 1 PI_STOP.
    If tracker drops confidence → stop, enter re-acquisition.

Reference Specifications:
  arduino/PROTOCOL.md (serial command & telemetry contract)
  arduino/point_to_point_control/point_to_point_control.ino (firmware FSM)
  AGENTS.md (two-layer Pi 5 + Arduino Uno architecture)
"""

import argparse
import http.server
import io
import json
import logging
import math
import os
import signal
import socketserver
import sys
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

from car.perception.config import (
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_FPS,
    EMERGENCY_STOP_DISTANCE_CM,
    SERIAL_PORT,
    SERIAL_BAUD,
    FOLLOW_CALIBRATION_FRAMES,
    FOLLOW_SIMILARITY_THRESHOLD,
    FOLLOW_TRACK_VERIFY_THRESHOLD,
    FOLLOW_ANTI_CONFUSION_MARGIN,
    FOLLOW_SEARCH_STEP_DEG,
    FOLLOW_SEARCH_TURN_SPEED,
    FOLLOW_SEARCH_MAX_STEPS,
    FOLLOW_SEARCH_INTERVAL_S,
    LANE_ROI_Y_MIN_FRAC,
    LANE_LOOKAHEAD_Y_FRAC,
    LANE_DEFAULT_WIDTH_FRAC,
    LANE_EMA_ALPHA,
    LANE_MAX_JUMP_FRAC,
    LANE_LOST_MAX_FRAMES,
    LANE_CRUISE_SPEED,
    LANE_CURVE_SPEED,
    LANE_STEER_DEADBAND_DEG,
    LANE_MAX_STEER_DEG,
    LANE_COLOR_MODE,
)
from car.perception.detector import Detector
from car.perception.sensors import SensorManager
from car.perception.avoidance import compute_zone_scores, decide_avoidance
from car.perception.tracker import ObjectTracker
from car.perception.lane_detector import ChartLaneDetector, LaneCurve, LaneDetectionResult
from car.control.motor_controller import MotorController
from car.control.navigation import NavigationSystem
from car.control.pid import PID
from car.control.traffic_sign_policy import TrafficSignPolicy
from car.perception.sign_model import resolve_sign_model

# ── Architecture §6 Rule 2 — Bounding-box height thresholds ──────────────────
BBOX_STOP_THRESHOLD = 0.40   # > 40% of frame height → PI_STOP
BBOX_SLOW_THRESHOLD = 0.15   # 15–40% → caution / reduce PWM

# Cruise and slow PWM values (0–255)
CRUISE_SPEED = 180
SLOW_SPEED   = 120

# ── Follow-me speed modulation (bbox height vs frame height) ─────────────────
FOLLOW_TOO_CLOSE    = 0.50   # > 50% → stop / hold
FOLLOW_IDEAL_MIN    = 0.25   # 25–50% → cruise
FOLLOW_GETTING_FAR  = 0.10   # 10–25% → speed up
FOLLOW_CRUISE_SPEED = 160
FOLLOW_FAST_SPEED   = 200
FOLLOW_SLOW_SPEED   = 80

# ── Command Throttling & Cadence ─────────────────────────────────────────────
# Prevents flooding the ATmega328P 115200 baud serial rx buffer.
DRIVE_CMD_DURATION_MS = 300     # Duration in ms sent with DRIVE commands
DRIVE_THROTTLE_INTERVAL_S = 0.18 # Interval between repeated DRIVE commands (~5.5 Hz)

# ── Camera Controls & AWB Definitions ────────────────────────────────────────
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


# =============================================================================
# Helper Utilities
# =============================================================================

def is_valid_us(dist: Any) -> bool:
    """Returns True if reading is a valid obstacle reflection (filters out 0.0 noise, 999 timeout, None, booleans, nan, inf)."""
    if dist is None or isinstance(dist, bool) or not isinstance(dist, (int, float)):
        return False
    if math.isnan(dist) or math.isinf(dist):
        return False
    return 1.0 < dist < 850.0


def find_daylight_ref() -> Optional[Path]:
    """Finds daylight reference image path if available."""
    candidates = [
        Path("/home/capstone/daylight.jpg"),
        Path("/home/capstone/model/daylight.jpg"),
        Path.home() / "daylight.jpg",
        Path.cwd() / "daylight.jpg",
        Path(__file__).resolve().parent.parent / "daylight.jpg",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def set_awb_mode(cam, mode_name: str) -> bool:
    """Applies camera AWB mode to picamera2 instance."""
    mode_name = mode_name.lower().strip()
    if mode_name not in AWB_MODES:
        return False
    if cam is None:
        return True
    val = AWB_MODES[mode_name]
    try:
        cam.set_controls({"AwbMode": val})
        print(f"☀️ Camera AWB set to: {mode_name.upper()} (matches daylight.jpg)")
        return True
    except Exception as e:
        print(f"⚠️ Failed to set AwbMode on camera: {e}")
        return False


def _bbox_height_frac(bbox, frame_height: int) -> float:
    """Returns bounding box height as a fraction of frame height."""
    x1, y1, x2, y2 = bbox
    return (y2 - y1) / max(frame_height, 1)


def _bbox_center_x(bbox) -> float:
    """Returns horizontal center coordinate of a bounding box."""
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0


def _check_obstacle_ahead(detections: list, frame_w: int, frame_h: int) -> Tuple[str, Optional[int]]:
    """
    Checks YOLO and pothole detections for obstacles in the forward driving path.
    Rule 1 & Rule 2:
      - Stop if obstacle is in forward path (center 20%-80%) and height > 40% of frame.
      - Slow if obstacle is in forward path and height > 15%.
      - Zone score check: if center zone is blocked and both sides dangerous -> STOP.
    """
    slow_candidate = False
    valid_dets = []

    for det in detections:
        if not isinstance(det, dict):
            continue
        bbox = det.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        try:
            frac = _bbox_height_frac(bbox, frame_h)
            cx = _bbox_center_x(bbox)
        except (TypeError, ValueError, ZeroDivisionError):
            continue

        valid_dets.append(det)
        in_center_corridor = 0.20 * frame_w <= cx <= 0.80 * frame_w

        if in_center_corridor:
            if frac > BBOX_STOP_THRESHOLD:
                return "STOP", None
            elif frac > BBOX_SLOW_THRESHOLD:
                slow_candidate = True

    if valid_dets:
        scores = compute_zone_scores(valid_dets, frame_w, frame_h)
        decision = decide_avoidance(scores)
        if decision == "STOP":
            return "STOP", None

    if slow_candidate:
        return "SLOW", SLOW_SPEED

    return "CLEAR", None


# =============================================================================
# Command Throttler
# =============================================================================

class CommandThrottler:
    """Throttles serial DRIVE, TURN, and PI_AVOID commands to prevent bus flooding and buffer delays."""

    def __init__(self, throttle_interval_s: float = DRIVE_THROTTLE_INTERVAL_S, drive_duration_ms: int = DRIVE_CMD_DURATION_MS):
        self.throttle_interval = throttle_interval_s
        self.drive_duration_ms = drive_duration_ms
        self.last_drive_time: float = 0.0
        self.last_speed: Optional[int] = None
        self.last_cmd: Optional[str] = None
        self.last_turn_time: float = 0.0
        self.last_turn_angle: Optional[int] = None
        self.last_turn_speed: Optional[int] = None
        self.last_avoid_hint: Optional[str] = None
        self.last_avoid_hint_time: float = 0.0
        self.latch_start_time: float = 0.0

    def should_send_drive(self, target_speed: int) -> bool:
        now = time.monotonic()
        if self.last_cmd != "DRIVE":
            return True
        if self.last_speed != target_speed:
            return True
        if (now - self.last_drive_time) >= self.throttle_interval:
            return True
        return False

    def record_drive(self, speed: int):
        self.last_drive_time = time.monotonic()
        self.last_speed = speed
        self.last_cmd = "DRIVE"

    def should_send_turn(self, target_angle: int, target_speed: int, interval_s: float = 0.12) -> bool:
        now = time.monotonic()
        if self.last_cmd != "TURN":
            return True
        if self.last_turn_speed is not None and self.last_turn_speed != target_speed:
            return True
        if self.last_turn_angle is None or abs(target_angle - self.last_turn_angle) >= 5:
            return True
        if (now - self.last_turn_time) >= interval_s:
            return True
        return False

    def record_turn(self, angle: int, speed: int):
        self.last_turn_time = time.monotonic()
        self.last_turn_angle = angle
        self.last_turn_speed = speed
        self.last_cmd = "TURN"
        self.last_speed = None

    def should_send_avoid_hint(self, hint: str, interval_s: float = 1.0) -> bool:
        now = time.monotonic()
        if self.last_avoid_hint != hint:
            return True
        if (now - self.last_avoid_hint_time) >= interval_s:
            return True
        return False

    def record_avoid_hint(self, hint: str):
        self.last_avoid_hint = hint
        self.last_avoid_hint_time = time.monotonic()

    def record_other(self, cmd: str):
        self.last_cmd = cmd
        self.last_speed = None


# =============================================================================
# System Runtime State
# =============================================================================

class SystemState:
    """Thread-safe runtime state container shared between perception and HTTP server."""

    def __init__(self):
        self.lock = threading.Lock()
        self.motors_enabled: bool = True     # False = Bench-test safe mode (no physical drive)
        self.invert_motors: bool = False     # True = TT motor polarity inversion fix
        self.active_action_label: str = "IDLE (Safety Hold)"
        self.current_awb: str = "daylight"
        self.current_mode: str = "autonomous"
        self.control_mode: str = "autonomous"  # "autonomous" or "manual"
        self.manual_override_obstacles: bool = False  # True = Ignore camera/ultrasonic obstacle stops in manual mode
        self.manual_speed: int = 185           # Manual drive PWM (130, 185, 235)
        self.manual_action: str = "IDLE"       # Current manual remote action
        self.pi_latched: bool = False
        self.fps: float = 0.0
        self.running: bool = True
        self.cam_instance: Any = None
        self.motor: Optional[MotorController] = None
        self.streaming_output: Optional['StreamingOutput'] = None

        # Follow-Me & Re-ID State
        self.follow_state: str = "WAIT_CALIBRATION"
        self.pending_select_point: Optional[Tuple[float, float]] = None
        self.pending_calibrate_center: bool = False
        self.pending_reset_target: bool = False
        self.target_bbox: Optional[Tuple[int, int, int, int]] = None
        self.target_similarity: float = 0.0
        self.calibration_progress: float = 0.0
        self.last_seen_side: str = "CENTER"
        self.non_target_persons: list = []

        # Lane Detection & Following State
        self.lane_color_mode: str = LANE_COLOR_MODE
        self.lane_result: Optional[LaneDetectionResult] = None

        self.traffic_signs = None
        self.traffic_sign_status = {"enabled": False}
        self.traffic_reset_requested = False


# =============================================================================
# Headless HTTP Live Stream with Diagnostic HUD
# =============================================================================

class StreamingOutput(io.BufferedIOBase):
    """Thread-safe MJPEG frame buffer using condition variable notification."""

    def __init__(self):
        self.frame: Optional[bytes] = None
        self.condition = threading.Condition()

    def write(self, buf: bytes):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Autonomous RC Car — Perception & Diagnostic HUD</title>
  <style>
    :root {
      --bg: #0d1117;
      --card-glass: rgba(18, 22, 30, 0.85);
      --border-glass: rgba(255, 255, 255, 0.15);
      --accent: #58a6ff; --success: #3fb950; --danger: #f85149; --warning: #d29922; --text: #f0f6fc; --muted: #8b949e;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html, body {
      background: var(--bg); color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      width: 100vw; height: 100vh; overflow: hidden;
      display: flex; justify-content: center; align-items: center; user-select: none;
    }
    #viewport {
      position: absolute; top: 0; left: 0; width: 100vw; height: 100vh;
      display: flex; justify-content: center; align-items: center; background: #000;
    }
    img#stream { width: 100%; height: 100%; object-fit: contain; display: block; }
    header {
      position: absolute; top: 12px; left: 50%; transform: translateX(-50%);
      background: var(--card-glass); backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
      border: 1px solid var(--border-glass); border-radius: 30px; padding: 6px 18px;
      display: flex; align-items: center; gap: 12px; box-shadow: 0 8px 32px rgba(0,0,0,0.6); z-index: 100;
    }
    .badge { padding: 4px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 700; }
    .badge-mode { background: rgba(88, 166, 255, 0.2); color: var(--accent); border: 1px solid var(--accent); }
    .badge-daylight { background: rgba(227, 179, 65, 0.2); color: #e3b341; border: 1px solid #e3b341; }
    .badge-motors-on { background: rgba(63, 185, 80, 0.2); color: var(--success); border: 1px solid var(--success); }
    .badge-motors-off { background: rgba(248, 81, 73, 0.2); color: var(--danger); border: 1px solid var(--danger); }
    .badge-invert-on { background: rgba(88, 166, 255, 0.2); color: var(--accent); border: 1px solid var(--accent); }
    .badge-invert-off { background: rgba(139, 148, 158, 0.2); color: var(--muted); border: 1px solid var(--muted); }
    .badge-us { background: rgba(63, 185, 80, 0.2); color: var(--success); border: 1px solid var(--success); }
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
      display: inline-flex; align-items: center; gap: 6px; transition: background 0.2s;
    }
    button:hover { background: rgba(255, 255, 255, 0.18); }
    button.btn-active { background: #1f6feb; border-color: #58a6ff; }
    button.btn-danger { background: rgba(248, 81, 73, 0.25); border-color: var(--danger); color: #ff7b72; }
    button.btn-success { background: rgba(63, 185, 80, 0.25); border-color: var(--success); color: #56d364; }
    .shortcut-tip { position: absolute; bottom: 6px; right: 12px; font-size: 0.7rem; color: var(--muted); opacity: 0.6; }
    .remote-panel {
      position: absolute; bottom: 80px; right: 20px;
      background: var(--card-glass); backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
      border: 1px solid var(--border-glass); border-radius: 16px; padding: 12px;
      display: flex; flex-direction: column; gap: 8px; z-index: 99;
      box-shadow: 0 8px 32px rgba(0,0,0,0.6);
      transition: all 0.3s ease;
    }
    .remote-panel.hidden { opacity: 0; pointer-events: none; transform: translateY(20px); }
    .remote-header {
      display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem; font-weight: 700;
    }
    .speed-pills { display: flex; gap: 4px; }
    .btn-speed {
      padding: 3px 8px; font-size: 0.7rem; border-radius: 10px; background: rgba(255,255,255,0.08); border: 1px solid var(--border-glass); color: var(--text); cursor: pointer;
    }
    .btn-speed.active { background: #1f6feb; border-color: #58a6ff; font-weight: 700; }
    .dpad-grid {
      display: grid; grid-template-columns: repeat(3, 44px); grid-template-rows: repeat(3, 44px); gap: 6px; justify-content: center;
    }
    .dpad-btn {
      width: 44px; height: 44px; padding: 0; border-radius: 10px; font-size: 1rem; font-weight: 700;
      background: rgba(255,255,255,0.08); border: 1px solid var(--border-glass); color: var(--text);
      display: flex; flex-direction: column; align-items: center; justify-content: center; cursor: pointer;
    }
    .dpad-btn:hover { background: rgba(255,255,255,0.2); }
    .dpad-btn:active { background: #1f6feb; border-color: #58a6ff; transform: scale(0.95); }
    .dpad-stop { background: rgba(248,81,73,0.2); border-color: var(--danger); color: #ff7b72; }
    .key-hint { font-size: 0.55rem; color: var(--muted); margin-top: 1px; }
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

  <aside id="traffic-sign-card" style="position:absolute;top:84px;right:16px;z-index:98;max-width:290px;padding:12px;background:rgba(13,17,23,.92);border:1px solid #58a6ff;border-radius:10px;font-size:.8rem">
    <strong>Traffic signs</strong>
    <div id="traffic-sign-label">Disabled</div>
    <div id="traffic-sign-action">Start with trained sign weights to enable.</div>
    <div id="traffic-sign-pending"></div>
    <div id="traffic-sign-fault" style="color:#ff7b72"></div>
    <button id="traffic-sign-reset" style="display:none;margin-top:8px" onclick="fetch('/traffic-signs/reset',{method:'POST'})">Reset sign hold / turn fault</button>
  </aside>

  <header id="topbar">
    <div style="font-weight:700; font-size:0.85rem">🏎️ MPSTME Capstone — Pi 5 ↔ Uno</div>
    <div class="badge badge-mode" id="mode-badge" onclick="toggleControlMode()" style="cursor:pointer" title="Click to toggle Mode (C)">MODE: AUTONOMOUS 🤖</div>
    <div class="badge badge-invert-off" id="target-badge" onclick="calibrateTarget()" style="cursor:pointer" title="Click to Calibrate (L)">TARGET: UNCALIBRATED 🎯</div>
    <div class="badge badge-invert-off" id="override-badge" onclick="toggleOverride()" style="cursor:pointer" title="Toggle Manual Obstacle Override (O)">OVERRIDE: OFF 🛡️</div>
    <div class="badge badge-daylight" id="awb-badge">AWB: DAYLIGHT ☀️</div>
    <div class="badge badge-motors-on" id="motors-badge">MOTORS: ACTIVE ⚡</div>
    <div class="badge badge-invert-off" id="invert-badge">POLARITY: NORMAL ➡️</div>
    <div class="badge badge-us" id="sonar-badge">US: L=--- R=---</div>
  </header>

  <div class="controls-dock" id="dock">
    <button id="btn-motors" class="btn-success" onclick="toggleMotors()">🛑 Disable Motors (M)</button>
    <button id="btn-mode" onclick="toggleControlMode()">🎮 Mode: Toggle (C)</button>
    <button id="btn-lane" onclick="setMode('lane')">🛣️ Lane Mode</button>
    <button id="btn-lane-color" onclick="cycleLaneColor()">🎨 Lane: Dark</button>
    <button id="btn-follow" onclick="setMode('follow')">🎯 Follow Mode</button>
    <button id="btn-cal-target" class="btn-active" onclick="calibrateTarget()">🎯 Calibrate Target (L)</button>
    <button id="btn-reset-target" onclick="resetTarget()">🔄 Reset Target</button>
    <button id="btn-override" onclick="toggleOverride()">🔓 Override Obstacles (O)</button>
    <button id="btn-remote" onclick="toggleRemoteWidget()">🕹️ D-Pad</button>
    <button id="btn-invert" onclick="toggleInvertMotors()">🔄 Polarity: Normal (I)</button>
    <button class="btn-active" id="btn-daylight" onclick="setAwb('daylight')">☀️ Daylight</button>
    <button onclick="setAwb('auto')">🔄 Auto</button>
    <button onclick="setAwb('indoor')">💡 Indoor</button>
    <button onclick="releaseLatch()" class="btn-active">✅ Release Latch (PI_GO)</button>
    <button onclick="emergencyStop()" class="btn-danger">🛑 E-STOP (Space)</button>
    <button onclick="testPulse()">🏎️ Test Pulse</button>
    <button onclick="testTurn(45)">🔄 Test Turn</button>
    <button onclick="toggleFullscreen()">⛶ Fullscreen</button>
    <button onclick="openRef()">📷 Ref daylight.jpg</button>
    <button onclick="toggleDock()" style="padding:7px 10px">👁️</button>
  </div>

  <!-- Floating Remote Control Widget (D-Pad) -->
  <div class="remote-panel hidden" id="remote-panel">
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
    <div style="font-size:0.65rem; color:var(--muted); text-align:center;">WASD or Arrow Keys to Drive</div>
  </div>

  <div class="shortcut-tip"><b>F</b> Fullscreen &bull; <b>M</b> Motors &bull; <b>I</b> Polarity &bull; <b>Space</b> E-Stop &bull; <b>D</b> Daylight &bull; <b>H</b> Dock</div>

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
    let currentControlMode = 'autonomous';
    let manualOverride = false;
    let currentSpeed = 185;
    let driveInterval = null;
    let activeAction = null;

    function toggleOverride() {
      fetch('/toggle_override').then(r => r.json()).then(d => {
        updateOverrideBadge(d.override);
      });
    }

    function updateOverrideBadge(enabled) {
      manualOverride = enabled;
      const b = document.getElementById('override-badge');
      const btn = document.getElementById('btn-override');
      if (enabled) {
        if (b) { b.innerText = 'OVERRIDE: ON 🔓'; b.className = 'badge badge-invert-on'; }
        if (btn) { btn.innerText = '🛡️ Override: ON (O)'; btn.className = 'btn-active'; }
      } else {
        if (b) { b.innerText = 'OVERRIDE: OFF 🛡️'; b.className = 'badge badge-invert-off'; }
        if (btn) { btn.innerText = '🔓 Override Obstacles (O)'; btn.className = ''; }
      }
    }

    function toggleControlMode() {
      fetch('/toggle_control_mode').then(r => r.json()).then(d => {
        currentControlMode = d.control_mode;
        updateModeBadge(d.control_mode);
      });
    }

    function setMode(mode) {
      fetch('/set_mode?mode=' + mode).then(r => r.json()).then(d => {
        currentControlMode = d.control_mode || d.mode;
        updateModeBadge(currentControlMode);
      });
    }

    function calibrateTarget() {
      fetch('/calibrate_target').then(r => r.json());
    }

    function resetTarget() {
      fetch('/reset_target').then(r => r.json());
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

    function updateModeBadge(mode) {
      const badge = document.getElementById('mode-badge');
      const btn = document.getElementById('btn-mode');
      const panel = document.getElementById('remote-panel');
      if (mode === 'manual') {
        badge.innerText = 'MODE: MANUAL 🎮';
        badge.className = 'badge badge-motors-on';
        if (btn) { btn.innerText = '🤖 Mode: Auto (C)'; btn.className = 'btn-active'; }
        if (panel) panel.classList.remove('hidden');
      } else if (mode === 'lane') {
        badge.innerText = 'MODE: LANE 🛣️';
        badge.className = 'badge badge-motors-on';
        if (btn) { btn.innerText = '🎯 Mode: Follow (C)'; btn.className = 'btn-active'; }
        if (panel) panel.classList.add('hidden');
      } else if (mode === 'follow') {
        badge.innerText = 'MODE: FOLLOW 🎯';
        badge.className = 'badge badge-invert-on';
        if (btn) { btn.innerText = '🎮 Mode: Manual (C)'; btn.className = ''; }
        if (panel) panel.classList.add('hidden');
      } else {
        badge.innerText = 'MODE: AUTONOMOUS 🤖';
        badge.className = 'badge badge-mode';
        if (btn) { btn.innerText = '🛣️ Mode: Lane (C)'; btn.className = ''; }
        if (panel) panel.classList.add('hidden');
      }
    }

    window.addEventListener('load', () => {
      const streamImg = document.getElementById('stream');
      if (streamImg) {
        streamImg.addEventListener('click', (e) => {
          const rect = streamImg.getBoundingClientRect();
          const imgAspect = 640.0 / 480.0;
          const elemAspect = rect.width / rect.height;
          let renderedW = rect.width;
          let renderedH = rect.height;
          let offsetX = 0;
          let offsetY = 0;
          if (elemAspect > imgAspect) {
            renderedW = rect.height * imgAspect;
            offsetX = (rect.width - renderedW) / 2.0;
          } else {
            renderedH = rect.width / imgAspect;
            offsetY = (rect.height - renderedH) / 2.0;
          }
          const clickX = e.clientX - rect.left - offsetX;
          const clickY = e.clientY - rect.top - offsetY;
          if (clickX >= 0 && clickX <= renderedW && clickY >= 0 && clickY <= renderedH) {
            const normX = Math.max(0.0, Math.min(1.0, clickX / renderedW));
            const normY = Math.max(0.0, Math.min(1.0, clickY / renderedH));
            fetch('/select_target?x=' + normX.toFixed(4) + '&y=' + normY.toFixed(4)).then(r => r.json());
          }
        });
      }
    });

    function toggleRemoteWidget() {
      const panel = document.getElementById('remote-panel');
      if (panel) panel.classList.toggle('hidden');
    }

    function setSpeed(spd) {
      fetch('/set_speed?speed=' + spd).then(r => r.json()).then(d => {
        currentSpeed = d.manual_speed;
        [130, 185, 235].forEach(s => {
          const b = document.getElementById('spd-' + s);
          if (b) b.className = (s === currentSpeed) ? 'btn-speed active' : 'btn-speed';
        });
      });
    }

    function startDrive(action) {
      if (activeAction === action) return;
      activeAction = action;
      if (driveInterval) clearInterval(driveInterval);
      fetch('/manual_drive?cmd=' + action + '&speed=' + currentSpeed);
      driveInterval = setInterval(() => {
        fetch('/manual_drive?cmd=' + action + '&speed=' + currentSpeed);
      }, 220);
    }

    function stopDrive() {
      if (driveInterval) {
        clearInterval(driveInterval);
        driveInterval = null;
      }
      activeAction = null;
      fetch('/manual_stop');
    }

    function toggleFullscreen() {
      if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch(() => {});
      } else {
        document.exitFullscreen().catch(() => {});
      }
    }

    function toggleMotors() {
      fetch('/toggle_motors').then(r => r.json()).then(d => {
        updateMotorsBadge(d.motors_enabled);
      });
    }

    function updateMotorsBadge(enabled) {
      const btn = document.getElementById('btn-motors');
      const badge = document.getElementById('motors-badge');
      if (enabled) {
        btn.innerText = '🛑 Disable Motors (M)';
        btn.className = 'btn-success';
        badge.innerText = 'MOTORS: ACTIVE ⚡';
        badge.className = 'badge badge-motors-on';
      } else {
        btn.innerText = '🚗 Enable Motors (M)';
        btn.className = 'btn-danger';
        badge.innerText = 'MOTORS: BENCH SAFE ⏸️';
        badge.className = 'badge badge-motors-off';
      }
    }

    function toggleInvertMotors() {
      fetch('/toggle_invert_motors').then(r => r.json()).then(d => {
        updateInvertBadge(d.invert_motors);
      });
    }

    function updateInvertBadge(inverted) {
      const btn = document.getElementById('btn-invert');
      const badge = document.getElementById('invert-badge');
      if (inverted) {
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
    }

    function emergencyStop() {
      fetch('/emergency_stop').then(r => r.json()).then(d => {
        updateMotorsBadge(false);
      });
    }

    function releaseLatch() {
      fetch('/release_latch').then(r => r.json());
    }

    function testPulse() {
      fetch('/test_drive').then(r => r.json());
    }

    function testTurn(angle = 45) {
      fetch('/test_turn?angle=' + angle).then(r => r.json());
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

    function pollTelemetry() {
      fetch('/telemetry').then(r => r.json()).then(d => {
        const sb = document.getElementById('sonar-badge');
        if (sb && d.sonar) {
          const l = (d.sonar.left <= 1.0 || d.sonar.left >= 850.0) ? 'CLR' : d.sonar.left.toFixed(0) + 'cm';
          const r = (d.sonar.right <= 1.0 || d.sonar.right >= 850.0) ? 'CLR' : d.sonar.right.toFixed(0) + 'cm';
          let usText = `US: L=${l}  R=${r}`;
          if (d.sonar.rear && d.sonar.rear > 1.0 && d.sonar.rear < 850.0) {
            usText += `  B=${d.sonar.rear.toFixed(0)}cm`;
          }
          sb.innerText = usText;
          if ((d.sonar.left > 1.0 && d.sonar.left < 25.0) || (d.sonar.right > 1.0 && d.sonar.right < 25.0)) {
            sb.style.background = 'rgba(248, 81, 73, 0.35)';
            sb.style.borderColor = 'var(--danger)';
            sb.style.color = '#ff7b72';
          } else if ((d.sonar.left > 1.0 && d.sonar.left < 45.0) || (d.sonar.right > 1.0 && d.sonar.right < 45.0)) {
            sb.style.background = 'rgba(227, 179, 65, 0.35)';
            sb.style.borderColor = 'var(--warning)';
            sb.style.color = '#e3b341';
          } else {
            sb.style.background = 'rgba(63, 185, 80, 0.2)';
            sb.style.borderColor = 'var(--success)';
            sb.style.color = 'var(--success)';
          }
        }
        if (d.follow_state) {
          const tb = document.getElementById('target-badge');
          if (tb) {
            if (d.follow_state === 'TRACKING') {
              tb.innerText = `🎯 LOCKED (${(d.target_similarity || 0).toFixed(2)}) [${d.last_seen_side || 'CTR'}]`;
              tb.className = 'badge badge-motors-on';
            } else if (d.follow_state === 'CALIBRATING') {
              tb.innerText = `⏳ CALIBRATING (${Math.round((d.calibration_progress || 0) * 100)}%)`;
              tb.className = 'badge badge-daylight';
            } else if (d.follow_state === 'RE_ACQUIRE') {
              tb.innerText = `🔍 SEARCHING [${d.last_seen_side || 'CTR'}]`;
              tb.className = 'badge badge-us';
            } else {
              tb.innerText = 'TARGET: CLICK TO SELECT 🎯';
              tb.className = 'badge badge-invert-off';
            }
          }
        }
        const signs = d.traffic_signs || {enabled:false};
        const names = {stop:'STOP', no_entry:'No Entry', turn_left:'Left', turn_right:'Right'};
        const label = signs.latest_sign ? (names[signs.latest_sign] || signs.latest_sign) : 'No current sign';
        const bearing = typeof signs.bearing_deg === 'number' ? ` | bearing ${signs.bearing_deg.toFixed(1)}°` : '';
        document.getElementById('traffic-sign-label').textContent = signs.enabled ? label + bearing : 'Disabled';
        document.getElementById('traffic-sign-action').textContent = signs.status || 'Traffic-sign model unavailable';
        document.getElementById('traffic-sign-pending').textContent = signs.pending_turn ? 'Pending: ' + (names[signs.pending_turn] || signs.pending_turn) : '';
        document.getElementById('traffic-sign-fault').textContent = signs.fault || '';
        document.getElementById('traffic-sign-reset').style.display = signs.enabled && (signs.no_entry || signs.fault) ? 'inline-block' : 'none';
        if (d.control_mode) {
          currentControlMode = d.control_mode;
          updateModeBadge(d.control_mode);
        } else if (d.mode) {
          document.getElementById('mode-badge').innerText = 'MODE: ' + d.mode.toUpperCase();
        }
        if (d.manual_override_obstacles !== undefined) updateOverrideBadge(d.manual_override_obstacles);
        if (d.motors_enabled !== undefined) updateMotorsBadge(d.motors_enabled);
        if (d.invert_motors !== undefined) updateInvertBadge(d.invert_motors);
      }).catch(() => {});
    }
    setInterval(pollTelemetry, 300);

    window.addEventListener('keydown', (e) => {
      const k = e.key.toLowerCase();
      if (k === 'f') toggleFullscreen();
      if (k === 'm') toggleMotors();
      if (k === 'i') toggleInvertMotors();
      if (k === 'c') toggleControlMode();
      if (k === 'l') calibrateTarget();
      if (k === 'o') toggleOverride();
      if (k === 'r') toggleRemoteWidget();
      if (k === 't') testPulse();
      if (k === ' ') { e.preventDefault(); emergencyStop(); stopDrive(); }
      if (k === 'h') toggleDock();
      if (k === 'd') setAwb('daylight');
      if (currentControlMode === 'manual') {
        if (k === 'w' || e.key === 'ArrowUp') { e.preventDefault(); startDrive('forward'); }
        if (k === 's' || e.key === 'ArrowDown') { e.preventDefault(); startDrive('reverse'); }
        if (k === 'a' || e.key === 'ArrowLeft') { e.preventDefault(); startDrive('left'); }
        if (k === 'd' || e.key === 'ArrowRight') { e.preventDefault(); startDrive('right'); }
        if (k === '1') setSpeed(130);
        if (k === '2') setSpeed(185);
        if (k === '3') setSpeed(235);
      }
    });

    window.addEventListener('keyup', (e) => {
      const k = e.key.toLowerCase();
      if (['w', 's', 'a', 'd', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright'].includes(k)) {
        stopDrive();
      }
    });
  </script>
</body>
</html>
"""


class DiagnosticHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler for video stream and diagnostic controls."""

    server: 'StreamingServer'

    def log_message(self, format, *args):
        # Suppress standard logging to prevent terminal flooding
        return

    def _json_response(self, status, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        state = self.server.sys_state
        path = urllib.parse.urlparse(self.path).path
        if path == "/traffic-signs/junction":
            self._json_response(410, {"error": "turns are automatic; junction trigger retired"})
            return
        if path != "/traffic-signs/reset":
            self._json_response(404, {"error": "unknown endpoint"})
            return
        with state.lock:
            if state.traffic_signs is None:
                self._json_response(409, {"error": "traffic sign feature is disabled"})
                return
            state.traffic_reset_requested = True
        self._json_response(202, {"status": "queued"})

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        sys_state: SystemState = self.server.sys_state

        # Sign mode owns movement arbitration. Direct HTTP motor commands and
        # manual entry would bypass that arbitration, so reject it.
        restricted = {"/manual_drive", "/test_drive", "/test_turn",
                      "/toggle_control_mode"}
        if sys_state.traffic_signs is not None and (
            path in restricted or (path == "/set_mode" and
            query.get("mode", ["autonomous"])[0] not in ("autonomous", "lane", "follow"))
        ):
            self._json_response(409, {"error": "sign mode supports autonomous/lane/follow only"})
            return

        if path in ('/', '/index.html'):
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        elif path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                out = sys_state.streaming_output
                while sys_state.running and out is not None:
                    with out.condition:
                        out.condition.wait(timeout=1.0)
                        frame_bytes = out.frame
                    if frame_bytes is None:
                        continue
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(frame_bytes)))
                    self.end_headers()
                    self.wfile.write(frame_bytes)
                    self.wfile.write(b'\r\n')
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                logging.warning("Stream client disconnected: %s", e)

        elif path == '/telemetry':
            sonar = sys_state.motor.latest_sonar.copy() if sys_state.motor else {"left": 999.0, "right": 999.0}
            state_str = sys_state.motor.latest_state if sys_state.motor else "UNKNOWN"
            port_name = sys_state.motor.port if sys_state.motor else "DISCONNECTED"
            connected = sys_state.motor.is_connected() if sys_state.motor else False
            with sys_state.lock:
                motors_enabled = sys_state.motors_enabled
                invert_motors = sys_state.invert_motors
                action = sys_state.active_action_label
                pi_latched = sys_state.pi_latched
                mode = sys_state.current_mode
                control_mode = sys_state.control_mode
                manual_override_obstacles = sys_state.manual_override_obstacles
                manual_speed = sys_state.manual_speed
                manual_action = sys_state.manual_action
                awb = sys_state.current_awb
                fps = round(sys_state.fps, 1)
                follow_state = sys_state.follow_state
                target_bbox = sys_state.target_bbox
                target_similarity = round(sys_state.target_similarity, 2)
                calibration_progress = round(sys_state.calibration_progress, 2)
                last_seen_side = sys_state.last_seen_side
                lane_res_snap = sys_state.lane_result
                lane_status = lane_res_snap.status if lane_res_snap else "OFF"
                lane_offset = round(lane_res_snap.lateral_offset_pct, 1) if lane_res_snap else 0.0
                lane_steer = round(lane_res_snap.steer_angle_deg, 1) if lane_res_snap else 0.0
                lane_color = sys_state.lane_color_mode
                traffic_signs = dict(sys_state.traffic_sign_status)
            data = {
                "sonar": sonar,
                "state": state_str,
                "port": port_name,
                "connected": connected,
                "motors_enabled": motors_enabled,
                "invert_motors": invert_motors,
                "action": action,
                "pi_latched": pi_latched,
                "mode": mode,
                "control_mode": control_mode,
                "manual_override_obstacles": manual_override_obstacles,
                "manual_speed": manual_speed,
                "manual_action": manual_action,
                "awb": awb,
                "fps": fps,
                "follow_state": follow_state,
                "target_bbox": target_bbox,
                "target_similarity": target_similarity,
                "calibration_progress": calibration_progress,
                "last_seen_side": last_seen_side,
                "lane_status": lane_status,
                "lane_offset": lane_offset,
                "lane_steer": lane_steer,
                "lane_color": lane_color,
                "traffic_signs": traffic_signs,
            }
            resp = json.dumps(data).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        elif path == '/toggle_override':
            with sys_state.lock:
                sys_state.manual_override_obstacles = not sys_state.manual_override_obstacles
                enabled = sys_state.manual_override_obstacles
            print(f"🔓 Manual obstacle override: {'ENABLED (Driver Full Authority)' if enabled else 'DISABLED (Safety Auto-Brake)'}")
            resp = json.dumps({"status": "ok", "override": enabled}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        elif path == '/toggle_motors':
            with sys_state.lock:
                sys_state.motors_enabled = not sys_state.motors_enabled
                enabled = sys_state.motors_enabled
            if not enabled and sys_state.motor:
                sys_state.motor.stop()
            print(f"🚗 Motors toggled: {'ENABLED' if enabled else 'BENCH SAFE (DISABLED)'}")
            resp = json.dumps({"status": "ok", "motors_enabled": enabled}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        elif path == '/toggle_invert_motors':
            with sys_state.lock:
                sys_state.invert_motors = not sys_state.invert_motors
                inverted = sys_state.invert_motors
            if sys_state.motor:
                sys_state.motor.invert_motors = inverted
            print(f"🔄 Motor polarity toggled: {'INVERTED' if inverted else 'NORMAL'}")
            resp = json.dumps({"status": "ok", "invert_motors": inverted}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        elif path == '/emergency_stop':
            with sys_state.lock:
                sys_state.motors_enabled = False
                sys_state.pi_latched = True
            if sys_state.motor:
                sys_state.motor.stop()
                sys_state.motor.pi_stop()
            print("🛑 Emergency Stop triggered via Web UI!")
            resp = json.dumps({"status": "ok", "motors_enabled": False, "pi_latched": True}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        elif path == '/release_latch':
            with sys_state.lock:
                sys_state.pi_latched = False
            if sys_state.motor:
                sys_state.motor.pi_go()
            print("✅ Latch release (PI_GO) triggered via Web UI!")
            resp = json.dumps({"status": "ok", "pi_latched": False}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        elif path == '/test_drive':
            executed = False
            with sys_state.lock:
                can_drive = sys_state.motors_enabled and not sys_state.pi_latched
            if can_drive and sys_state.motor:
                sys_state.motor.drive(CRUISE_SPEED, 300)
                executed = True
            resp = json.dumps({"status": "ok", "executed": executed}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        elif path == '/test_turn':
            try:
                angle = int(query.get('angle', [45])[0])
            except (ValueError, TypeError):
                angle = 45
            executed = False
            with sys_state.lock:
                can_drive = sys_state.motors_enabled and not sys_state.pi_latched
            if can_drive and sys_state.motor:
                sys_state.motor.turn(angle, speed=150)
                executed = True
            resp = json.dumps({"status": "ok", "executed": executed, "angle": angle}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        elif path == '/toggle_control_mode':
            with sys_state.lock:
                # Cycle: autonomous -> lane -> follow -> manual -> autonomous
                if sys_state.control_mode == "autonomous":
                    sys_state.control_mode = "lane"
                    sys_state.current_mode = "lane"
                elif sys_state.control_mode == "lane":
                    sys_state.control_mode = "follow"
                    sys_state.current_mode = "follow"
                    if sys_state.follow_state not in ("TRACKING", "CALIBRATING"):
                        sys_state.follow_state = "WAIT_CALIBRATION"
                elif sys_state.control_mode == "follow":
                    sys_state.control_mode = "manual"
                else:
                    sys_state.control_mode = "autonomous"
                    sys_state.current_mode = "autonomous"
                mode_now = sys_state.control_mode
            if mode_now == "manual" and sys_state.motor:
                sys_state.motor.stop()
            print(f"🎮 Control mode toggled: {mode_now.upper()}")
            resp = json.dumps({"status": "ok", "control_mode": mode_now, "mode": mode_now}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        elif path == '/set_mode':
            new_mode = query.get('mode', ['autonomous'])[0].lower().strip()
            with sys_state.lock:
                if new_mode == "manual":
                    sys_state.control_mode = "manual"
                elif new_mode in ("autonomous", "lane", "follow"):
                    sys_state.control_mode = new_mode
                    sys_state.current_mode = new_mode
                    if new_mode == "follow" and sys_state.follow_state not in ("TRACKING", "CALIBRATING"):
                        sys_state.follow_state = "WAIT_CALIBRATION"
                mode_now = sys_state.control_mode
            if mode_now == "manual" and sys_state.motor:
                sys_state.motor.stop()
            print(f"🔄 Mode switched to: {mode_now.upper()}")
            resp = json.dumps({"status": "ok", "mode": mode_now, "control_mode": mode_now}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        elif path == '/set_lane_color':
            color = query.get('color', ['dark'])[0].lower().strip()
            if color in ("dark", "yellow", "blue", "white"):
                with sys_state.lock:
                    sys_state.lane_color_mode = color
                    if hasattr(self.server, "lane_detector") and self.server.lane_detector:
                        self.server.lane_detector.set_color_mode(color)
            print(f"🎨 Lane color mode set to: {color.upper()}")
            resp = json.dumps({"status": "ok", "color": color}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        elif path == '/select_target':
            try:
                nx = float(query.get('x', [0.5])[0])
                ny = float(query.get('y', [0.5])[0])
            except (ValueError, TypeError):
                nx, ny = 0.5, 0.5
            with sys_state.lock:
                sys_state.current_mode = "follow"
                sys_state.control_mode = "follow"
                sys_state.pending_select_point = (nx, ny)
            print(f"🎯 Target select queued at ({nx:.2f}, {ny:.2f})")
            resp = json.dumps({"status": "ok", "point": [nx, ny]}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        elif path == '/calibrate_target':
            with sys_state.lock:
                sys_state.current_mode = "follow"
                sys_state.control_mode = "follow"
                sys_state.pending_calibrate_center = True
            print("🎯 Calibrate target queued")
            resp = json.dumps({"status": "ok"}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        elif path == '/reset_target':
            with sys_state.lock:
                sys_state.pending_reset_target = True
                sys_state.follow_state = "WAIT_CALIBRATION"
                sys_state.target_bbox = None
                sys_state.target_similarity = 0.0
                sys_state.calibration_progress = 0.0
                sys_state.last_seen_side = "CENTER"
            if sys_state.motor:
                sys_state.motor.stop()
            print("🔄 Target profile reset queued")
            resp = json.dumps({"status": "ok"}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        elif path == '/set_speed':
            try:
                spd = int(query.get('speed', [185])[0])
            except (ValueError, TypeError):
                spd = 185
            spd = max(80, min(255, spd))
            with sys_state.lock:
                sys_state.manual_speed = spd
            resp = json.dumps({"status": "ok", "manual_speed": spd}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        elif path == '/manual_drive':
            cmd = query.get('cmd', [''])[0].lower().strip()
            spd = int(query.get('speed', [sys_state.manual_speed])[0]) if 'speed' in query else sys_state.manual_speed
            executed = False
            blocked = False
            reason = "ok"

            with sys_state.lock:
                override = sys_state.manual_override_obstacles
                can_drive = sys_state.motors_enabled and (override or not sys_state.pi_latched)
                sys_state.control_mode = "manual"

            if can_drive and sys_state.motor:
                sonar = sys_state.motor.latest_sonar
                us_l = sonar.get("left", 999.0)
                us_r = sonar.get("right", 999.0)
                us_rear = sonar.get("rear", 999.0)

                if cmd in ("forward", "fwd"):
                    if not override and ((is_valid_us(us_l) and us_l < EMERGENCY_STOP_DISTANCE_CM) or (is_valid_us(us_r) and us_r < EMERGENCY_STOP_DISTANCE_CM)):
                        blocked = True
                        reason = "us_front_blocked"
                        sys_state.motor.stop()
                    else:
                        sys_state.motor.drive(spd, 350)
                        executed = True
                elif cmd in ("reverse", "rev"):
                    if not override and is_valid_us(us_rear) and us_rear < EMERGENCY_STOP_DISTANCE_CM:
                        blocked = True
                        reason = "us_rear_blocked"
                        sys_state.motor.stop()
                    else:
                        sys_state.motor.reverse(spd, 350)
                        executed = True
                elif cmd in ("left", "turn_left"):
                    sys_state.motor.turn(-40, speed=160)
                    executed = True
                elif cmd in ("right", "turn_right"):
                    sys_state.motor.turn(40, speed=160)
                    executed = True
                elif cmd in ("stop", "halt"):
                    sys_state.motor.stop()
                    executed = True

            with sys_state.lock:
                sys_state.manual_action = cmd.upper() if executed else ("BLOCKED" if blocked else "IDLE")

            resp = json.dumps({"status": "ok" if executed else "blocked", "cmd": cmd, "executed": executed, "reason": reason}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        elif path == '/manual_stop':
            if sys_state.motor:
                sys_state.motor.stop()
            with sys_state.lock:
                sys_state.manual_action = "STOP"
            resp = json.dumps({"status": "ok", "action": "STOP"}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        elif path == '/set_awb':
            mode = query.get('mode', ['daylight'])[0]
            success = set_awb_mode(sys_state.cam_instance, mode)
            if success:
                sys_state.current_awb = mode
            resp = json.dumps({"status": "ok" if success else "error", "mode": sys_state.current_awb}).encode('utf-8')
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
                self.send_error(404, "daylight.jpg not found on filesystem")
        else:
            self.send_error(404)


class StreamingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Multi-threaded HTTP Server for low-latency streaming."""
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address, RequestHandlerClass, sys_state: SystemState):
        super().__init__(server_address, RequestHandlerClass)
        self.sys_state = sys_state


# =============================================================================
# Diagnostic HUD Overlay Renderer
# =============================================================================

def draw_hud(
    frame: np.ndarray,
    detections: list,
    action_label: str,
    latest_sonar: Optional[dict],
    latest_state: Optional[str],
    port_name: Optional[str],
    fps: float,
    awb_mode: str,
    mode: str,
    motors_enabled: bool,
    invert_motors: bool,
    pi_latched: bool,
    target_bbox: Optional[Tuple[int, int, int, int]] = None,
    target_similarity: float = 0.0,
    calibration_progress: float = 0.0,
    last_seen_side: str = "CENTER",
    follow_state: str = "WAIT_CALIBRATION",
    non_target_persons: Optional[list] = None,
    lane_result: Optional[LaneDetectionResult] = None,
) -> np.ndarray:
    """Renders active bounding boxes, zone markers, and live diagnostic telemetry overlay."""
    if frame is None or getattr(frame, "size", 0) == 0:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

    annotated = frame.copy()
    h, w = annotated.shape[:2]

    if latest_sonar is None or not isinstance(latest_sonar, dict):
        latest_sonar = {}
    if latest_state is None:
        latest_state = "UNKNOWN"
    if port_name is None:
        port_name = "DISCONNECTED"
    if fps is None or not isinstance(fps, (int, float)):
        fps = 0.0
    action_label = str(action_label or "IDLE")
    mode = str(mode or "autonomous")
    awb_mode = str(awb_mode or "daylight")

    # Dynamic scaling for resolution (supports 320x240, 640x480, etc.)
    is_hd = (w >= 480)
    box_scale  = 0.45 if is_hd else 0.35
    hud_scale1 = 0.55 if is_hd else 0.40
    hud_scale2 = 0.45 if is_hd else 0.32
    thick      = 2 if is_hd else 1

    # ── 1. Zone Divider Markers (0.33 and 0.67 of width) ─────────────────────
    z_left  = int(w * 0.33)
    z_right = int(w * 0.67)
    cv2.line(annotated, (z_left, 0), (z_left, h), (70, 70, 70), 1)
    cv2.line(annotated, (z_right, 0), (z_right, h), (70, 70, 70), 1)
    cv2.putText(annotated, "L", (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
    cv2.putText(annotated, "C", (int(w * 0.48), h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
    cv2.putText(annotated, "R", (max(10, w - 20), h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

    # ── 1b. Render Lane Corridor, Polylines & Target Reticle (Lane Mode) ─────
    if lane_result is not None:
        if lane_result.corridor_pts is not None and len(lane_result.corridor_pts) >= 4:
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [lane_result.corridor_pts], (40, 180, 50))
            cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)

        if lane_result.left_pts is not None and len(lane_result.left_pts) >= 2:
            cv2.polylines(annotated, [lane_result.left_pts], False, (255, 255, 0), thick + 1)

        if lane_result.right_pts is not None and len(lane_result.right_pts) >= 2:
            cv2.polylines(annotated, [lane_result.right_pts], False, (0, 215, 255), thick + 1)

        if lane_result.center_pts is not None and len(lane_result.center_pts) >= 2:
            cv2.polylines(annotated, [lane_result.center_pts], False, (255, 0, 255), thick)

        if lane_result.lookahead_point is not None and (lane_result.status != "LANE_LOST" or lane_result.center_coeffs is not None):
            lx, ly = lane_result.lookahead_point
            cv2.drawMarker(annotated, (lx, ly), (0, 0, 255), cv2.MARKER_CROSS, 14, thick + 1)
            cv2.circle(annotated, (lx, ly), 6, (0, 255, 255), thick)

    # ── 2. Render Detections (YOLO Obstacles, Potholes & Bystanders) ─────────
    non_target_set = set()
    if non_target_persons:
        for ntp in non_target_persons:
            if isinstance(ntp, dict) and ntp.get("bbox"):
                non_target_set.add(tuple(ntp["bbox"]))

    for det in (detections or []):
        if not isinstance(det, dict):
            continue
        bbox = det.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
        except (ValueError, TypeError):
            continue
        is_pothole = det.get("type") == "negative" or "pothole" in det.get("class_name", "").lower()
        is_bystander = tuple(bbox) in non_target_set or (det.get("class_name") == "person" and non_target_persons and det in non_target_persons)

        if is_bystander:
            color = (0, 140, 255)  # Orange for Bystander Obstacles
            lbl = "BYSTANDER [OBSTACLE]"
        elif is_pothole:
            color = (0, 69, 255)   # Red/Orange for Potholes
            lbl = det.get("class_name", "pothole")
        else:
            color = (0, 255, 0)    # Green for General Obstacles
            lbl = det.get("class_name", "obstacle")

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thick)
        conf = det.get("confidence", 0.0)
        text = f"{lbl.upper()} {conf:.2f}"
        cv2.putText(annotated, text, (max(0, x1), max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, box_scale, color, thick)

    # ── 3. Render Target Lock Box with Corner Reticles ────────────────────────
    if target_bbox is not None and follow_state in ("TRACKING", "CALIBRATING"):
        tx, ty, tw, th = [int(v) for v in target_bbox]
        tx2, ty2 = tx + tw, ty + th
        tgt_color = (0, 215, 255)  # Gold
        cv2.rectangle(annotated, (tx, ty), (tx2, ty2), tgt_color, thick)

        # Reticles on 4 corners
        rlen = max(6, min(22, int(min(tw, th) * 0.22)))
        cv2.line(annotated, (tx, ty), (tx + rlen, ty), (255, 255, 0), thick + 1)
        cv2.line(annotated, (tx, ty), (tx, ty + rlen), (255, 255, 0), thick + 1)
        cv2.line(annotated, (tx2, ty), (tx2 - rlen, ty), (255, 255, 0), thick + 1)
        cv2.line(annotated, (tx2, ty), (tx2, ty + rlen), (255, 255, 0), thick + 1)
        cv2.line(annotated, (tx, ty2), (tx + rlen, ty2), (255, 255, 0), thick + 1)
        cv2.line(annotated, (tx, ty2), (tx, ty2 - rlen), (255, 255, 0), thick + 1)
        cv2.line(annotated, (tx2, ty2), (tx2 - rlen, ty2), (255, 255, 0), thick + 1)
        cv2.line(annotated, (tx2, ty2), (tx2, ty2 - rlen), (255, 255, 0), thick + 1)

        if follow_state == "CALIBRATING":
            bar_w = max(50, tw)
            bar_h = 6
            bx = tx
            by = max(14, ty - 14)
            cv2.rectangle(annotated, (bx, by), (bx + bar_w, by + bar_h), (40, 40, 40), -1)
            fill_w = int(bar_w * max(0.0, min(1.0, calibration_progress)))
            cv2.rectangle(annotated, (bx, by), (bx + fill_w, by + bar_h), (0, 255, 0), -1)
            cal_lbl = f"CALIBRATING ({int(calibration_progress * FOLLOW_CALIBRATION_FRAMES)}/{FOLLOW_CALIBRATION_FRAMES})"
            cv2.putText(annotated, cal_lbl, (bx, max(12, by - 4)), cv2.FONT_HERSHEY_SIMPLEX, box_scale, (0, 255, 255), thick)
        else:
            tgt_lbl = f"TARGET (SIM: {target_similarity:.2f}) [{last_seen_side}]"
            cv2.putText(annotated, tgt_lbl, (max(0, tx), max(12, ty - 6)), cv2.FONT_HERSHEY_SIMPLEX, box_scale, (0, 255, 255), thick)

    # ── 4. Diagnostic HUD Text Overlays ──────────────────────────────────────
    head_str = f"{fps:.1f} FPS | {mode.upper()} | {awb_mode.upper()} AWB"
    cv2.putText(annotated, head_str, (10, 20 if is_hd else 15),
                cv2.FONT_HERSHEY_SIMPLEX, hud_scale1, (0, 255, 255), thick)

    # Action line
    act_color = (0, 255, 0) if "CRUISING" in action_label or "LOCKED" in action_label else (0, 165, 255)
    if "STOP" in action_label or "BRAKE" in action_label or "BLOCKED" in action_label:
        act_color = (0, 0, 255)
    cv2.putText(annotated, f"Action: {action_label}", (10, 42 if is_hd else 30),
                cv2.FONT_HERSHEY_SIMPLEX, hud_scale1, act_color, thick)

    # Directional Search Banner when in RE_ACQUIRE
    if follow_state == "RE_ACQUIRE":
        search_str = f"RE_ACQUIRE: SEARCHING {last_seen_side}"
        cv2.putText(annotated, search_str, (10, 64 if is_hd else 48),
                    cv2.FONT_HERSHEY_SIMPLEX, hud_scale1, (0, 165, 255), thick)
    elif lane_result is not None:
        if lane_result.status != "LANE_LOST":
            lane_str = f"LANE: {lane_result.status} | Off: {lane_result.lateral_offset_pct:+.1f}% | δ: {lane_result.steer_angle_deg:+.0f}°"
            cv2.putText(annotated, lane_str, (10, 64 if is_hd else 48),
                        cv2.FONT_HERSHEY_SIMPLEX, hud_scale1, (0, 255, 200), thick)
        elif lane_result.center_coeffs is not None:
            lane_str = f"LANE: COASTING ({lane_result.lost_frames}) | Off: {lane_result.lateral_offset_pct:+.1f}% | δ: {lane_result.steer_angle_deg:+.0f}°"
            cv2.putText(annotated, lane_str, (10, 64 if is_hd else 48),
                        cv2.FONT_HERSHEY_SIMPLEX, hud_scale1, (0, 165, 255), thick)

    # Ultrasonic telemetry display
    us_l = latest_sonar.get("left", 999.0)
    us_r = latest_sonar.get("right", 999.0)
    us_rear = latest_sonar.get("rear", 999.0)
    v_l = is_valid_us(us_l)
    v_r = is_valid_us(us_r)
    v_rear = is_valid_us(us_rear)
    us_l_disp = f"{us_l:.0f}cm" if v_l else "CLR"
    us_r_disp = f"{us_r:.0f}cm" if v_r else "CLR"
    us_str = f"US: L={us_l_disp} | R={us_r_disp}"
    if v_rear:
        us_str += f" | B={us_rear:.0f}cm"

    us_color = (0, 255, 0)
    if (v_l and us_l < EMERGENCY_STOP_DISTANCE_CM) or (v_r and us_r < EMERGENCY_STOP_DISTANCE_CM):
        us_color = (0, 0, 255)
        us_str += " [BRAKE]"
    elif (v_l and us_l < 42.0) or (v_r and us_r < 42.0):
        us_color = (0, 140, 255)
        us_str += " [AVOID]"
    elif (v_l and us_l < 60.0) or (v_r and us_r < 60.0):
        us_color = (0, 255, 255)
        us_str += " [SLOW]"

    # Bottom status lines
    cv2.putText(annotated, us_str, (10, h - (32 if is_hd else 24)),
                cv2.FONT_HERSHEY_SIMPLEX, hud_scale2, us_color, thick)

    motor_status = "ACTIVE ⚡" if motors_enabled else "BENCH SAFE ⏸️"
    ard_str = f"Arduino: {port_name} [{latest_state}] | Motors: {motor_status} | Pol: {'INV' if invert_motors else 'NORM'}"
    cv2.putText(annotated, ard_str, (10, h - (14 if is_hd else 10)),
                cv2.FONT_HERSHEY_SIMPLEX, hud_scale2, (200, 200, 200), 1)

    if pi_latched:
        cv2.putText(annotated, "[PI_STOP LATCH ACTIVE]", (max(10, w - (230 if is_hd else 160)), 20 if is_hd else 15),
                    cv2.FONT_HERSHEY_SIMPLEX, hud_scale2, (0, 0, 255), thick)

    return annotated


# =============================================================================
# Chart-Paper Lane Following Mode (Option A)
# =============================================================================

def _run_lane(
    motor: MotorController,
    sensors: SensorManager,
    lane_detector: ChartLaneDetector,
    detections: list,
    us_blocked: bool,
    frame: np.ndarray,
    pi_latched: bool,
    motors_enabled: bool,
    throttler: CommandThrottler,
    sys_state: Optional[SystemState] = None,
    lane_result: Optional[LaneDetectionResult] = None,
) -> Tuple[bool, str, LaneDetectionResult]:
    """
    Executes one loop iteration of Chart-Paper Lane Following mode (Option A).
    Priority Stack:
      Tier 1: Imminent obstacle / pothole camera stop (PI_STOP latch)
      Tier 1.5 & Tier 2: Cliff reflex & Ultrasonic auto-brake / local Arduino FSM
      Tier 2.5: In-lane active obstacle/pothole bypass (steer to clear side)
      Tier 3: Nominal chart-paper lane centering via pure pursuit
    """
    frame_h, frame_w = frame.shape[:2]
    us = sensors.read_ultrasonic()
    arduino_state = (motor.latest_state or "UNKNOWN").upper()

    # Filter valid detections
    valid_detections = [
        d for d in (detections or [])
        if isinstance(d, dict) and d.get("bbox") and len(d.get("bbox")) == 4
    ]

    # Run Lane Detection (excluding bounding boxes of active obstacles & potholes)
    lane_res = lane_result if lane_result is not None else lane_detector.detect(frame, exclude_boxes=valid_detections)
    if sys_state:
        sys_state.lane_result = lane_res

    # ── 1. Tier 1: Camera Obstacle Latch (Rule 1 & Rule 2) ───────────────────
    action, slow_speed = _check_obstacle_ahead(valid_detections, frame_w, frame_h)

    if action == "STOP":
        if not pi_latched:
            print("🛑 PI_STOP: Obstacle/Pothole ahead in lane (camera latch engaged)")
            motor.pi_stop()
            pi_latched = True
            throttler.record_other("PI_STOP")
            return pi_latched, "🛑 PI_STOP: Obstacle/Pothole Ahead in Lane", lane_res
        else:
            throttler.record_other("PI_STOPPED")
            return pi_latched, "🛑 PI_STOPPED (Lane Obstacle Latch Active)", lane_res

    # ── 2. Camera Latch Release ──────────────────────────────────────────────
    if pi_latched:
        if action == "CLEAR" and not us_blocked:
            print("✅ PI_GO: Lane corridor clear, releasing camera latch.")
            motor.pi_go()
            pi_latched = False
            arduino_state = "IDLE"
        else:
            throttler.record_other("PI_STOPPED")
            return pi_latched, "🛑 PI_STOPPED (Lane Latch Active)", lane_res

    # ── 3. Respect Arduino Autonomous Reflex States (Tier 2 & 1.5) ────────────
    if arduino_state in ("AVOIDING", "STATE_AVOIDING", "CLIFF_REVERSING", "STATE_CLIFF_REVERSING"):
        throttler.record_other(arduino_state)
        return pi_latched, f"🔀 ARDUINO {arduino_state} (Yielding)", lane_res

    if arduino_state in ("CLIFF_STOPPED", "STATE_CLIFF_STOPPED"):
        throttler.record_other(arduino_state)
        if action == "CLEAR" and not us_blocked and is_valid_us(us.get("left")) and is_valid_us(us.get("right")):
            print("✅ PI_GO: Cliff cleared, releasing latch.")
            motor.pi_go()
            return False, "✅ PI_GO: Cliff Cleared", lane_res
        return pi_latched, "⚠️ CLIFF_STOPPED (Latched Drop-Off)", lane_res

    if arduino_state in ("PI_STOPPED", "STATE_PI_STOPPED"):
        throttler.record_other(arduino_state)
        if action == "CLEAR" and not us_blocked:
            print("✅ PI_GO: Releasing stale Arduino PI_STOPPED state.")
            motor.pi_go()
            arduino_state = "IDLE"
        else:
            pi_latched = True
            return pi_latched, "🛑 PI_STOPPED (Arduino Latched)", lane_res

    # ── 4. Ultrasonic Critical Stop (< EMERGENCY_STOP_DISTANCE_CM) ───────────
    if us_blocked:
        throttler.record_other("STOP")
        if motors_enabled:
            motor.stop()
        return pi_latched, f"🛑 US AUTO-BRAKE (L:{us.get('left', 999):.0f}cm R:{us.get('right', 999):.0f}cm)", lane_res

    # ── 5. Tier 2.5: In-Lane Active Obstacle / Pothole Bypass ────────────────
    # Check if there are active obstacles or potholes inside the driving corridor
    scores = compute_zone_scores(valid_detections, frame_w, frame_h, sonar=us)
    avoid_decision = decide_avoidance(scores)
    us_l, us_r = us.get("left", 999.0), us.get("right", 999.0)

    if avoid_decision in ("LEFT", "RIGHT"):
        hint = avoid_decision
        if throttler.should_send_avoid_hint(hint):
            motor.pi_avoid(hint)
            throttler.record_avoid_hint(hint)
        turn_angle = 32 if hint == "RIGHT" else -32
        turn_speed = LANE_CURVE_SPEED
        if motors_enabled and throttler.should_send_turn(turn_angle, turn_speed):
            motor.turn(turn_angle, speed=turn_speed)
            throttler.record_turn(turn_angle, turn_speed)
        return pi_latched, f"🔀 LANE BYPASS: Avoiding Hazard -> Steering {hint}", lane_res

    # Close ultrasonic asymmetric avoidance
    if is_valid_us(us_l) and us_l < 42.0 and (not is_valid_us(us_r) or us_l < us_r - 5.0):
        if throttler.should_send_avoid_hint("RIGHT"):
            motor.pi_avoid("RIGHT")
            throttler.record_avoid_hint("RIGHT")
        if motors_enabled and throttler.should_send_turn(32, LANE_CURVE_SPEED):
            motor.turn(32, speed=LANE_CURVE_SPEED)
            throttler.record_turn(32, LANE_CURVE_SPEED)
        return pi_latched, f"🔀 LANE BYPASS: Left Obstacle ({us_l:.0f}cm) -> Steering RIGHT", lane_res

    if is_valid_us(us_r) and us_r < 42.0 and (not is_valid_us(us_l) or us_r < us_l - 5.0):
        if throttler.should_send_avoid_hint("LEFT"):
            motor.pi_avoid("LEFT")
            throttler.record_avoid_hint("LEFT")
        if motors_enabled and throttler.should_send_turn(-32, LANE_CURVE_SPEED):
            motor.turn(-32, speed=LANE_CURVE_SPEED)
            throttler.record_turn(-32, LANE_CURVE_SPEED)
        return pi_latched, f"🔀 LANE BYPASS: Right Obstacle ({us_r:.0f}cm) -> Steering LEFT", lane_res

    # ── 6. Tier 3: Nominal Chart-Paper Lane Following ────────────────────────
    # If lane is lost and either exceeding coasting limit OR no prior lane lock:
    if lane_res.status == "LANE_LOST" and (lane_res.lost_frames > lane_detector.lost_max_frames or lane_res.center_coeffs is None):
        if motors_enabled:
            motor.stop()
        throttler.record_other("STOP")
        return pi_latched, f"⚠️ LANE LOST ({lane_res.lost_frames}f): Holding Safe Stop", lane_res

    # Either locked onto lane (DUAL_LOCKED, SINGLE_LEFT, SINGLE_RIGHT) or active coasting with valid trajectory
    delta = lane_res.steer_angle_deg
    is_coasting = (lane_res.status == "LANE_LOST")

    if abs(delta) <= lane_detector.steer_deadband_deg:
        target_speed = LANE_CRUISE_SPEED
        if motors_enabled:
            if throttler.should_send_drive(target_speed):
                motor.drive(target_speed, throttler.drive_duration_ms)
                throttler.record_drive(target_speed)
        else:
            target_speed = 0
        if is_coasting:
            action_label = f"⚠️ LANE COASTING ({lane_res.lost_frames}/{lane_detector.lost_max_frames}): Straight"
        else:
            action_label = f"🛣️ LANE LOCKED ({lane_res.status}): Straight (PWM {LANE_CRUISE_SPEED})"
    else:
        steer_cmd = int(round(delta))
        target_speed = LANE_CURVE_SPEED
        if motors_enabled:
            if throttler.should_send_turn(steer_cmd, target_speed):
                motor.turn(steer_cmd, speed=target_speed)
                throttler.record_turn(steer_cmd, target_speed)
        steer_dir = "RIGHT" if steer_cmd > 0 else "LEFT"
        if is_coasting:
            action_label = f"⚠️ LANE COASTING ({lane_res.lost_frames}/{lane_detector.lost_max_frames}): Steer {steer_dir} ({abs(steer_cmd)}°)"
        else:
            action_label = f"🛣️ LANE STEER {steer_dir} ({abs(steer_cmd)}° | PWM {LANE_CURVE_SPEED})"

    return pi_latched, action_label, lane_res


# =============================================================================
# Autonomous Navigation Mode
# =============================================================================

def _run_autonomous(
    motor: MotorController,
    sensors: SensorManager,
    detections: list,
    us_blocked: bool,
    frame: np.ndarray,
    pi_latched: bool,
    motors_enabled: bool,
    throttler: CommandThrottler,
    fresh_inference: bool = True,
) -> Tuple[bool, str]:
    """
    One loop iteration of autonomous navigation.
    Priority stack:
      Tier 1   — Pi Camera Latch (PI_STOP / PI_GO)
      Tier 1.5 — Arduino Cliff Reflex (CLIFF_REVERSING / CLIFF_STOPPED)
      Tier 2   — Arduino Ultrasonic Avoidance (AVOIDING)
      Tier 3   — Normal Driving (DRIVE / SLOW / STOP)

    Returns:
      (updated_pi_latched, action_label)
    """
    frame_h, frame_w = frame.shape[:2]
    us = sensors.read_ultrasonic()
    arduino_state = (motor.latest_state or "UNKNOWN").upper()

    valid_detections = [
        d for d in (detections or [])
        if isinstance(d, dict) and d.get("bbox") and len(d.get("bbox")) == 4
    ]

    # ── 1. Check for Imminent Obstacles Ahead (Tier 1 Camera Latch) ───────────
    # Tier 1 camera latch is absolute: overrides everything including mid-avoidance
    # and mid-cliff reverse per PROTOCOL.md.
    action, slow_speed = _check_obstacle_ahead(valid_detections, frame_w, frame_h)

    if action == "STOP":
        if not pi_latched:
            print("🛑 PI_STOP: Imminent obstacle ahead (camera)")
            motor.pi_stop()
            pi_latched = True
            throttler.record_other("PI_STOP")
            throttler.latch_start_time = time.monotonic()
            return pi_latched, "🛑 PI_STOP: Obstacle Imminent (Camera)"
        else:
            # Check if stationary obstacle allows auto-unstick / pivot escape
            now_m = time.monotonic()
            latch_age = now_m - getattr(throttler, "latch_start_time", 0.0)
            if latch_age > 1.2 and valid_detections:
                scores = compute_zone_scores(valid_detections, frame_w, frame_h, sonar=us)
                decision = decide_avoidance(scores)
                us_l, us_r = us.get("left", 999.0), us.get("right", 999.0)
                score_l, score_r = scores.get("left", 0.0), scores.get("right", 0.0)

                # Prefer side with substantially lower danger / better clearance
                if (decision == "RIGHT" or score_r < score_l) and (not is_valid_us(us_r) or us_r > 32.0):
                    print("🔀 Auto-Unstick: Right side clear, releasing latch and turning right")
                    motor.pi_go()
                    pi_latched = False
                    throttler.latch_start_time = 0.0
                    if motors_enabled:
                        motor.turn(45, speed=165)
                        throttler.record_turn(45, 165)
                    return pi_latched, "🔀 ESCAPE: Pivoting RIGHT around obstacle"
                elif (decision == "LEFT" or score_l < score_r) and (not is_valid_us(us_l) or us_l > 32.0):
                    print("🔀 Auto-Unstick: Left side clear, releasing latch and turning left")
                    motor.pi_go()
                    pi_latched = False
                    throttler.latch_start_time = 0.0
                    if motors_enabled:
                        motor.turn(-45, speed=165)
                        throttler.record_turn(-45, 165)
                    return pi_latched, "🔀 ESCAPE: Pivoting LEFT around obstacle"

            throttler.record_other("PI_STOPPED")
            return pi_latched, "🛑 PI_STOPPED (Obstacle Imminent)"

    # ── 2. Handle Existing Camera Latch ───────────────────────────────────────
    if pi_latched:
        if action == "CLEAR" and not us_blocked:
            print("✅ PI_GO: Path clear, releasing camera latch.")
            motor.pi_go()
            pi_latched = False
            throttler.latch_start_time = 0.0
            arduino_state = "IDLE"
        else:
            throttler.record_other("PI_STOPPED")
            return pi_latched, "🛑 PI_STOPPED (Camera Latch Active)"

    # ── 3. Respect Arduino Autonomous Reflex States (Tier 2 & 1.5) ────────────
    if arduino_state in ("AVOIDING", "STATE_AVOIDING", "CLIFF_REVERSING", "STATE_CLIFF_REVERSING"):
        throttler.record_other(arduino_state)
        return pi_latched, f"🔀 ARDUINO {arduino_state} (Yielding)"

    # ── 4. Handle Cliff Latched Stop (Tier 1.5) ───────────────────────────────
    if arduino_state in ("CLIFF_STOPPED", "STATE_CLIFF_STOPPED"):
        throttler.record_other(arduino_state)
        # Release cliff latch only if path is clear on camera AND ground reflection is valid
        if action == "CLEAR" and not us_blocked and is_valid_us(us.get("left")) and is_valid_us(us.get("right")):
            print("✅ PI_GO: Cliff drop-off cleared, releasing latch.")
            motor.pi_go()
            return False, "✅ PI_GO: Cliff Cleared"
        return pi_latched, "⚠️ CLIFF_STOPPED (Latched Drop-Off)"

    # ── 4b. Sync Arduino PI_STOPPED state if unlatched on Pi ─────────────────
    if arduino_state in ("PI_STOPPED", "STATE_PI_STOPPED"):
        throttler.record_other(arduino_state)
        if action == "CLEAR" and not us_blocked:
            print("✅ PI_GO: Releasing stale Arduino PI_STOPPED state.")
            motor.pi_go()
            arduino_state = "IDLE"
        else:
            pi_latched = True
            return pi_latched, "🛑 PI_STOPPED (Arduino Latched)"

    # ── 5. Ultrasonic Critical Stop (< EMERGENCY_STOP_DISTANCE_CM) ───────────
    if us_blocked:
        throttler.record_other("STOP")
        motor.stop()
        return pi_latched, f"🛑 US AUTO-BRAKE (L:{us.get('left', 999):.0f}cm R:{us.get('right', 999):.0f}cm)"

    # ── 6. Active Obstacle Avoidance & Steering ──────────────────────────────
    scores = compute_zone_scores(valid_detections, frame_w, frame_h, sonar=us)
    decision = decide_avoidance(scores)
    us_l, us_r = us.get("left", 999.0), us.get("right", 999.0)

    # 6a. Fused Camera + Ultrasonic Directional Bypass
    if decision in ("LEFT", "RIGHT"):
        if throttler.should_send_avoid_hint(decision):
            motor.pi_avoid(decision)
            throttler.record_avoid_hint(decision)

        # Active skid-steer bypass turn when obstacle is detected ahead
        turn_angle = 35 if decision == "RIGHT" else -35
        turn_speed = 160
        if motors_enabled and throttler.should_send_turn(turn_angle, turn_speed):
            motor.turn(turn_angle, speed=turn_speed)
            throttler.record_turn(turn_angle, turn_speed)
            dir_str = "RIGHT" if decision == "RIGHT" else "LEFT"
            return pi_latched, f"🔀 AVOIDING OBSTACLE: Steering {dir_str} ({abs(turn_angle)}°)"

    # 6b. Close Ultrasonic Asymmetric Avoidance (sensor-only fallback)
    if is_valid_us(us_l) and us_l < 42.0 and (not is_valid_us(us_r) or us_l < us_r - 5.0):
        if throttler.should_send_avoid_hint("RIGHT"):
            motor.pi_avoid("RIGHT")
            throttler.record_avoid_hint("RIGHT")
        if motors_enabled and throttler.should_send_turn(35, 160):
            motor.turn(35, speed=160)
            throttler.record_turn(35, 160)
            return pi_latched, f"🔀 US AVOID: Left Obs ({us_l:.0f}cm) -> Steering RIGHT"

    if is_valid_us(us_r) and us_r < 42.0 and (not is_valid_us(us_l) or us_r < us_l - 5.0):
        if throttler.should_send_avoid_hint("LEFT"):
            motor.pi_avoid("LEFT")
            throttler.record_avoid_hint("LEFT")
        if motors_enabled and throttler.should_send_turn(-35, 160):
            motor.turn(-35, speed=160)
            throttler.record_turn(-35, 160)
            return pi_latched, f"🔀 US AVOID: Right Obs ({us_r:.0f}cm) -> Steering LEFT"

    # ── 7. Forward Driving (Cruising or Cautionary) ───────────────────────────
    in_us_caution = ((is_valid_us(us_l) and us_l < 60.0) or
                     (is_valid_us(us_r) and us_r < 60.0))
    if action == "SLOW" or in_us_caution:
        target_speed = slow_speed if slow_speed is not None else SLOW_SPEED
        action_label = f"🟡 CRUISING CAUTIOUSLY (PWM {target_speed})"
    else:
        target_speed = CRUISE_SPEED
        action_label = f"🟢 CRUISING FORWARD (PWM {target_speed})"

    # Execute drive with throttling and bench-safe check
    if motors_enabled:
        if throttler.should_send_drive(target_speed):
            motor.drive(target_speed, throttler.drive_duration_ms)
            throttler.record_drive(target_speed)
    else:
        action_label = f"⏸️ BENCH SAFE: DRIVE PWM {target_speed} (Suppressed)"

    return pi_latched, action_label


# =============================================================================
# Follow-Me Mode
# =============================================================================

def _run_follow(
    motor: MotorController,
    sensors: SensorManager,
    tracker: ObjectTracker,
    pid: PID,
    detections: list,
    us_blocked: bool,
    frame: np.ndarray,
    follow_state: str,
    pi_latched: bool,
    motors_enabled: bool,
    throttler: CommandThrottler,
    fresh_inference: bool = True,
    sys_state: Optional[SystemState] = None,
) -> Tuple[str, bool, str]:
    """
    One loop iteration of follow-me mode.
    Priority: Tier 1 non-target obstacle stop > PID person steering + active obstacle bypass.

    Returns:
      (updated_follow_state, updated_pi_latched, action_label)
    """
    frame_h, frame_w = frame.shape[:2]
    us = sensors.read_ultrasonic()
    arduino_state = (motor.latest_state or "UNKNOWN").upper()

    # ── Handle Web UI / Pending Triggers ─────────────────────────────────────
    if sys_state and getattr(sys_state, "pending_reset_target", False):
        tracker.reset()
        follow_state = "WAIT_CALIBRATION"
        sys_state.pending_reset_target = False
        sys_state.target_bbox = None
        sys_state.target_similarity = 0.0
        sys_state.calibration_progress = 0.0
        sys_state.last_seen_side = "CENTER"
        sys_state.follow_state = follow_state
        if motors_enabled:
            motor.stop()
        return follow_state, pi_latched, "🔄 TARGET RESET: Waiting for Selection"

    persons = [d for d in (detections or []) if isinstance(d, dict) and d.get("class_name") == "person"]
    non_person_obstacles = [
        d for d in (detections or [])
        if isinstance(d, dict) and d.get("class_name") != "person" and d.get("type") != "negative"
    ]
    pothole_obstacles = [d for d in (detections or []) if isinstance(d, dict) and d.get("type") == "negative"]

    if sys_state and getattr(sys_state, "pending_select_point", None) is not None:
        nx, ny = sys_state.pending_select_point
        sys_state.pending_select_point = None
        px, py = nx * frame_w, ny * frame_h
        best_p = None
        best_dist = float('inf')
        for p in persons:
            x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
            if x1 <= px <= x2 and y1 <= py <= y2:
                best_p = p
                break
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            dist = math.hypot(cx - px, cy - py)
            if dist < best_dist and dist < (0.25 * frame_w):
                best_dist = dist
                best_p = p
        if best_p:
            x1, y1, x2, y2 = [int(v) for v in best_p["bbox"]]
            bbox = (x1, y1, x2 - x1, y2 - y1)
            tracker.start_calibration(frame, bbox)
            follow_state = "CALIBRATING"
            sys_state.follow_state = follow_state
            sys_state.target_bbox = bbox
            sys_state.calibration_progress = tracker.calibration_progress
            if motors_enabled:
                motor.stop()
            return follow_state, pi_latched, "🎯 CALIBRATING: Target Selected by Click"
        else:
            return follow_state, pi_latched, "⚠️ SELECT FAILED: No Person at Click Position"

    if sys_state and getattr(sys_state, "pending_calibrate_center", False):
        sys_state.pending_calibrate_center = False
        if persons:
            best_p = min(
                persons,
                key=lambda p: abs((p["bbox"][0] + p["bbox"][2]) / 2.0 - frame_w / 2.0)
            )
            x1, y1, x2, y2 = [int(v) for v in best_p["bbox"]]
            bbox = (x1, y1, x2 - x1, y2 - y1)
            tracker.start_calibration(frame, bbox)
            follow_state = "CALIBRATING"
            sys_state.follow_state = follow_state
            sys_state.target_bbox = bbox
            sys_state.calibration_progress = tracker.calibration_progress
            if motors_enabled:
                motor.stop()
            return follow_state, pi_latched, "🎯 CALIBRATING: Center Target Selected"
        else:
            return follow_state, pi_latched, "⚠️ CALIBRATE FAILED: No Person in View"

    # ── Respect Arduino Autonomous States ────────────────────────────────────
    if arduino_state in ("AVOIDING", "STATE_AVOIDING", "CLIFF_REVERSING", "STATE_CLIFF_REVERSING"):
        throttler.record_other(arduino_state)
        return follow_state, pi_latched, f"🔀 ARDUINO {arduino_state} (Yielding)"

    # ── Non-Target Obstacle Check for Pre-Tracking States ─────────────────────
    if follow_state in ("WAIT_CALIBRATION", "CALIBRATING", "LOCK_ON"):
        pre_obstacles = non_person_obstacles + pothole_obstacles
        obs_action, _ = _check_obstacle_ahead(pre_obstacles, frame_w, frame_h) if pre_obstacles else ("CLEAR", None)
        if obs_action == "STOP":
            if not pi_latched:
                print("🛑 PI_STOP: Non-target obstacle ahead (camera)")
                motor.pi_stop()
                pi_latched = True
                throttler.record_other("PI_STOP")
                return follow_state, pi_latched, "🛑 PI_STOP: Non-Target Obstacle"
            else:
                throttler.record_other("PI_STOPPED")
                return follow_state, pi_latched, "🛑 PI_STOPPED (Non-Target Obstacle)"

        if pi_latched:
            if obs_action == "CLEAR" and not us_blocked:
                print("✅ PI_GO: Path clear, releasing follow latch.")
                motor.pi_go()
                pi_latched = False
            else:
                throttler.record_other("PI_STOPPED")
                return follow_state, pi_latched, "🛑 PI_STOPPED (Follow Latch Active)"

    # ── State Machine ────────────────────────────────────────────────────────

    # 1. WAIT_CALIBRATION
    if follow_state == "WAIT_CALIBRATION":
        is_cal = getattr(tracker, "is_calibrated", True)
        is_trk = getattr(tracker, "is_tracking", False)
        if is_cal and is_trk:
            follow_state = "TRACKING"
            if sys_state:
                sys_state.follow_state = "TRACKING"
        else:
            if motors_enabled:
                motor.stop()
            throttler.record_other("STOP")
            return follow_state, pi_latched, "🎯 FOLLOW: Click Person or Calibrate"

    # 2. CALIBRATING
    if follow_state == "CALIBRATING":
        if motors_enabled:
            motor.stop()
        throttler.record_other("STOP")
        done = tracker.step_calibration(frame, persons) if hasattr(tracker, "step_calibration") else True
        cal_prog = getattr(tracker, "calibration_progress", 1.0)
        is_cal = getattr(tracker, "is_calibrating", False)
        is_locked = getattr(tracker, "is_calibrated", False)

        if not is_cal and not is_locked:
            # Person lost during calibration -> abort and wait for new selection
            follow_state = "WAIT_CALIBRATION"
            if sys_state:
                sys_state.follow_state = "WAIT_CALIBRATION"
                sys_state.target_bbox = None
                sys_state.calibration_progress = 0.0
            return follow_state, pi_latched, "⚠️ CALIBRATION ABORTED: Person Lost"

        if sys_state:
            sys_state.calibration_progress = cal_prog
            sys_state.target_bbox = getattr(tracker, "bbox", None)
            sys_state.last_seen_side = getattr(tracker, "last_seen_side", "CENTER")
        if done:
            follow_state = "TRACKING"
            if sys_state:
                sys_state.follow_state = "TRACKING"
            action_label = "🎯 TARGET LOCKED (Calibration Complete)"
        else:
            frames_count = getattr(tracker, "_total_calibration_frames", FOLLOW_CALIBRATION_FRAMES)
            action_label = f"🎯 CALIBRATING CLOTHES & FIGURE ({int(cal_prog * frames_count)}/{frames_count})"
        return follow_state, pi_latched, action_label

    # 3. LOCK_ON (Legacy auto-lock onto largest person)
    if follow_state == "LOCK_ON":
        if persons:
            best = max(
                persons,
                key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1])
            )
            x1, y1, x2, y2 = [int(v) for v in best["bbox"]]
            bbox = (x1, y1, x2 - x1, y2 - y1)
            tracker.lock_on(frame, bbox)
            follow_state = "TRACKING"
            if sys_state:
                sys_state.follow_state = "TRACKING"
                sys_state.target_bbox = bbox
            action_label = "🎯 TARGET LOCKED"
        else:
            if motors_enabled:
                motor.stop()
            throttler.record_other("STOP")
            action_label = "🔍 SCANNING FOR TARGET (Owner)"
        return follow_state, pi_latched, action_label

    # 4. TRACKING
    if follow_state == "TRACKING":
        if hasattr(tracker, "update_with_detections"):
            try:
                is_tracking, tgt_box, target_det, non_target_persons, sim_score = tracker.update_with_detections(
                    frame, persons, fresh_inference=fresh_inference
                )
            except TypeError:
                is_tracking, tgt_box, target_det, non_target_persons, sim_score = tracker.update_with_detections(
                    frame, persons
                )
        else:
            is_tracking, tgt_box = tracker.update(frame)
            target_det = None
            non_target_persons = persons
            sim_score = 1.0 if is_tracking else 0.0

        if sys_state:
            sys_state.target_bbox = tgt_box
            sys_state.target_similarity = sim_score
            sys_state.last_seen_side = getattr(tracker, "last_seen_side", "CENTER")
            sys_state.non_target_persons = non_target_persons

        if not is_tracking or tgt_box is None:
            print("🤷 Tracker lost target → RE_ACQUIRE.")
            motor.pi_stop()
            pi_latched = True
            throttler.record_other("PI_STOP")
            follow_state = "RE_ACQUIRE"
            if sys_state:
                sys_state.follow_state = "RE_ACQUIRE"
            last_side = getattr(tracker, "last_seen_side", "CENTER")
            return follow_state, pi_latched, f"⚠️ TARGET LOST: Re-acquiring (Last: {last_side})"

        # Non-target persons are treated as obstacles!
        all_obstacles = non_person_obstacles + pothole_obstacles + non_target_persons

        # Evaluate obstacle path ahead (Tier 1 Pi Camera Latch)
        obs_action, obs_slow = _check_obstacle_ahead(all_obstacles, frame_w, frame_h) if all_obstacles else ("CLEAR", None)
        if obs_action == "STOP":
            if not pi_latched:
                print("🛑 PI_STOP: Non-target obstacle ahead (camera)")
                motor.pi_stop()
                pi_latched = True
                throttler.record_other("PI_STOP")
                return follow_state, pi_latched, "🛑 PI_STOP: Non-Target Obstacle"
            else:
                throttler.record_other("PI_STOPPED")
                return follow_state, pi_latched, "🛑 PI_STOPPED (Non-Target Obstacle)"

        # Handle Existing Latch Release
        if pi_latched:
            if obs_action == "CLEAR" and not us_blocked:
                print("✅ PI_GO: Path clear, releasing follow latch.")
                motor.pi_go()
                pi_latched = False
            else:
                throttler.record_other("PI_STOPPED")
                return follow_state, pi_latched, "🛑 PI_STOPPED (Follow Latch Active)"

        # Respect Arduino autonomous states
        if arduino_state in ("AVOIDING", "STATE_AVOIDING", "CLIFF_REVERSING", "STATE_CLIFF_REVERSING"):
            throttler.record_other(arduino_state)
            return follow_state, pi_latched, f"🔀 ARDUINO {arduino_state} (Yielding)"

        if arduino_state in ("CLIFF_STOPPED", "STATE_CLIFF_STOPPED"):
            throttler.record_other(arduino_state)
            if obs_action != "STOP" and not us_blocked and is_valid_us(us.get("left")) and is_valid_us(us.get("right")):
                print("✅ PI_GO: Cliff drop-off cleared, releasing latch.")
                motor.pi_go()
                return follow_state, False, "✅ PI_GO: Cliff Cleared"
            return follow_state, pi_latched, "⚠️ CLIFF_STOPPED (Latched Drop-Off)"

        if arduino_state in ("PI_STOPPED", "STATE_PI_STOPPED"):
            throttler.record_other(arduino_state)
            if obs_action == "CLEAR" and not us_blocked:
                motor.pi_go()
            else:
                pi_latched = True
                return follow_state, pi_latched, "🛑 PI_STOPPED (Arduino Latched)"

        # Critical Ultrasonic auto-brake
        if us_blocked:
            throttler.record_other("STOP")
            motor.stop()
            return follow_state, pi_latched, "🛑 US AUTO-BRAKE (< 25cm)"

        # Steering & Follow Speed Modulation
        x, y, w, h = tgt_box
        center_x = x + w / 2.0
        bbox_h_frac = h / max(frame_h, 1)

        pid.update(center_x)
        angle = -int(pid.output)
        angle = max(-45, min(45, angle))

        # Active Obstacle Bypass While Following
        scores = compute_zone_scores(all_obstacles, frame_w, frame_h, sonar=us) if all_obstacles else {}
        decision = decide_avoidance(scores) if scores else "CLEAR"

        avoid_override = False
        us_l = us.get("left", 999.0)
        us_r = us.get("right", 999.0)

        if decision in ("LEFT", "RIGHT"):
            avoid_override = True
            angle = -35 if decision == "LEFT" else 35
            if throttler.should_send_avoid_hint(decision):
                motor.pi_avoid(decision)
                throttler.record_avoid_hint(decision)
        elif is_valid_us(us_l) and us_l < 42.0 and (not is_valid_us(us_r) or us_r > us_l + 10.0):
            avoid_override = True
            angle = max(25, angle)
        elif is_valid_us(us_r) and us_r < 42.0 and (not is_valid_us(us_l) or us_l > us_r + 10.0):
            avoid_override = True
            angle = min(-25, angle)

        prefix = "🔀 BYPASS+FOLLOW" if avoid_override else "🎯 FOLLOWING"

        if bbox_h_frac > FOLLOW_TOO_CLOSE:
            if motors_enabled:
                motor.stop()
            throttler.record_other("STOP")
            action_label = "🎯 TARGET TOO CLOSE (Holding)"
        elif bbox_h_frac > FOLLOW_IDEAL_MIN:
            spd = FOLLOW_SLOW_SPEED if obs_action == "SLOW" else FOLLOW_CRUISE_SPEED
            action_label = f"{prefix} (Cruise PWM {spd}, Steer {angle}°)"
            if motors_enabled:
                if abs(angle) < 5:
                    if throttler.should_send_drive(spd):
                        motor.drive(spd, throttler.drive_duration_ms)
                        throttler.record_drive(spd)
                else:
                    if throttler.should_send_turn(angle, spd):
                        motor.turn(angle, speed=spd)
                        throttler.record_turn(angle, spd)
            else:
                action_label = f"⏸️ BENCH SAFE: {prefix} CRUISE (Suppressed)"
        elif bbox_h_frac > FOLLOW_GETTING_FAR:
            spd = FOLLOW_SLOW_SPEED if obs_action == "SLOW" else FOLLOW_FAST_SPEED
            action_label = f"{prefix} (Fast PWM {spd}, Steer {angle}°)"
            if motors_enabled:
                if abs(angle) < 5:
                    if throttler.should_send_drive(spd):
                        motor.drive(spd, throttler.drive_duration_ms)
                        throttler.record_drive(spd)
                else:
                    if throttler.should_send_turn(angle, spd):
                        motor.turn(angle, speed=spd)
                        throttler.record_turn(angle, spd)
            else:
                action_label = f"⏸️ BENCH SAFE: {prefix} FAST (Suppressed)"
        else:
            spd = FOLLOW_SLOW_SPEED if obs_action == "SLOW" else FOLLOW_FAST_SPEED
            action_label = f"{prefix} (PWM {spd}, Steer {angle}°)"
            if motors_enabled:
                if abs(angle) < 5:
                    if throttler.should_send_drive(spd):
                        motor.drive(spd, throttler.drive_duration_ms)
                        throttler.record_drive(spd)
                else:
                    if throttler.should_send_turn(angle, spd):
                        motor.turn(angle, speed=spd)
                        throttler.record_turn(angle, spd)
            else:
                action_label = f"⏸️ BENCH SAFE: {prefix} FAR (Suppressed)"

        return follow_state, pi_latched, action_label

    # 5. RE_ACQUIRE
    elif follow_state == "RE_ACQUIRE":
        if persons and tracker.reacquire(frame, persons):
            print("🎯 Target re-identified! Resuming TRACKING.")
            motor.pi_go()
            pi_latched = False
            if hasattr(tracker, "reset_search"):
                tracker.reset_search()
            follow_state = "TRACKING"
            if sys_state:
                sys_state.follow_state = "TRACKING"
                sys_state.target_bbox = getattr(tracker, "bbox", None)
            action_label = "🎯 TARGET RE-IDENTIFIED"
            return follow_state, pi_latched, action_label

        all_obstacles = non_person_obstacles + pothole_obstacles + persons
        obs_action, _ = _check_obstacle_ahead(all_obstacles, frame_w, frame_h) if all_obstacles else ("CLEAR", None)

        # Stepped directional search turns toward last_seen_side
        now_mono = time.monotonic()
        side = getattr(tracker, "last_seen_side", "CENTER")
        should_step = tracker.should_step_search(now_mono) if hasattr(tracker, "should_step_search") else False
        search_steps = getattr(tracker, "search_steps_taken", 0)

        if side in ("LEFT", "RIGHT") and should_step:
            side_dist = us.get("left") if side == "LEFT" else us.get("right")
            side_clear = (not is_valid_us(side_dist)) or (float(side_dist) > 35.0)

            if side_clear and obs_action != "STOP" and not us_blocked:
                if pi_latched:
                    motor.pi_go()
                    pi_latched = False
                step_angle = -FOLLOW_SEARCH_STEP_DEG if side == "LEFT" else FOLLOW_SEARCH_STEP_DEG
                if motors_enabled:
                    motor.turn(step_angle, speed=FOLLOW_SEARCH_TURN_SPEED)
                    throttler.record_turn(step_angle, FOLLOW_SEARCH_TURN_SPEED)
                if hasattr(tracker, "record_search_step"):
                    tracker.record_search_step(now_mono)
                search_steps = getattr(tracker, "search_steps_taken", 0)
                action_label = f"🔍 SEARCHING {side} (Step {search_steps}/{FOLLOW_SEARCH_MAX_STEPS})"
            else:
                if motors_enabled:
                    motor.stop()
                action_label = f"🔍 SEARCH {side} BLOCKED (Holding)"
        else:
            if motors_enabled:
                motor.stop()
            if search_steps >= FOLLOW_SEARCH_MAX_STEPS:
                action_label = f"🔍 SEARCH HOLD (Scanned {side}, awaiting target)"
            elif persons:
                action_label = "🔍 RE-ACQUIRING: Comparing Candidates"
            else:
                action_label = f"🔍 RE-ACQUIRING: Scanning ({side})"

        return follow_state, pi_latched, action_label

    return follow_state, pi_latched, action_label


# =============================================================================
# Main Pipeline
# =============================================================================

def draw_traffic_signs(frame, detections):
    """Separate blue sign boxes: never mix signs into obstacle danger scoring."""
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    labels = {"stop": "STOP", "no_entry": "NO ENTRY", "turn_left": "LEFT", "turn_right": "RIGHT"}
    for detection in detections:
        try:
            from car.perception.traffic_signs import sign_geometry
            geometry = sign_geometry(detection, annotated.shape)
            x1, y1, x2, y2 = detection["bbox"]
            x1, x2 = [max(0, min(width - 1, int(x))) for x in (x1, x2)]
            y1, y2 = [max(0, min(height - 1, int(y))) for y in (y1, y2)]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 180, 40), 2)
            text = f"SIGN {labels[detection['class_name']]} {geometry['bearing_deg']:+.0f}deg"
            cv2.putText(annotated, text, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 180, 40), 1)
        except (ValueError, KeyError, TypeError):
            continue
    return annotated


def _pause_follow_for_sign(tracker, pid, state):
    """Drop stale frame tracking while keeping the selected owner's signature."""
    calibrated = tracker.is_calibrated
    tracker.stop()
    tracker.reset_search()
    setpoint = pid.SetPoint
    pid.clear()
    pid.SetPoint = setpoint
    # Unfinished samples must not span a stopped/turning camera interval.
    tracker.is_calibrating = False
    state.follow_state = "RE_ACQUIRE" if calibrated else "WAIT_CALIBRATION"
    state.target_bbox = None
    state.target_similarity = 0.0
    state.non_target_persons = []


class TrafficSignRuntime:
    """Automatic image-space sign actions; no metric distance or yaw claims.

    Lane turns use bounded pulses and current lane reacquisition. Autonomous
    turns are time calibrated only. Sonar and camera hazards preempt both.
    """
    def __init__(self, detector, clock=None, turn_ms_per_degree=10.0, turn_pwm=210):
        if not math.isfinite(turn_ms_per_degree) or not 1 <= turn_ms_per_degree <= 100:
            raise ValueError("turn_ms_per_degree must be finite and between 1 and 100")
        if isinstance(turn_pwm, bool) or not isinstance(turn_pwm, int) or not 0 <= turn_pwm <= 255:
            raise ValueError("traffic turn PWM must be an integer between 0 and 255")
        self.turn_pwm = turn_pwm
        self.detector = detector
        self.clock = time.monotonic if clock is None else clock
        self.policy = TrafficSignPolicy(clock=self.clock)
        self.turn_ms_per_degree = turn_ms_per_degree
        self.status = "Waiting for traffic signs"
        self.fault = None
        self.sign_detections = []
        self.geometry = None
        self.latest_sign = None
        self.direction = None
        self.turn_mode = None
        self.turn_started = 0.0
        self.turn_until = 0.0
        self.pulse_until = 0.0
        self.commanded_degrees = 0
        self.lane_confirmations = 0
        self.completed_status = None
        self.lane_departed = False
        self.last_pulse_stopped = 0.0

    def snapshot(self):
        return {"enabled": True, "status": self.status, "fault": self.fault,
                "latest_sign": self.latest_sign,
                "bearing_deg": self.geometry["bearing_deg"] if self.geometry else None,
                "height_frac": self.geometry["height_frac"] if self.geometry else None,
                "turn_state": "turning" if self.direction else "idle",
                **self.policy.snapshot()}

    def _interrupt(self, motor, reason):
        if self.direction:
            self.fault = "Turn interrupted: " + reason + "; reset required"
            self.policy.clear_pending()
        self.direction = None
        self.turn_until = self.pulse_until = 0.0
        motor.stop()

    def _complete(self, motor, message):
        motor.stop()
        self.direction = None
        self.turn_until = self.pulse_until = 0.0
        self.policy.clear_pending()
        self.completed_status = message
        self.status = message
        return message

    def step(self, frame, motor, *, camera_valid, obstacle_detections,
             obstacle_time, sonar, pi_latched, motors_enabled,
             reset=False, mode="autonomous", lane_result=None, sonar_time=None):
        from car.perception.traffic_signs import sign_geometry
        if reset:
            self._interrupt(motor, "operator reset")
            self.fault = None
            self.policy.reset()
            self.completed_status = None
        if mode not in ("autonomous", "lane", "follow"):
            self._interrupt(motor, "unsupported control mode")
            self.status = "Traffic signs: autonomous/lane/follow only - holding"
            return self.status
        if self.direction and mode != self.turn_mode:
            self._interrupt(motor, "control mode changed")
        detection_error = None
        if camera_valid:
            try:
                observations = self.detector.poll(frame)
                if observations is not None:
                    self.policy.observe(observations)
                    self.sign_detections = observations
            except Exception as error:
                logging.exception("Traffic-sign inference failed")
                detection_error = type(error).__name__
        now = self.clock()
        fresh = (self.policy.last_observation is not None and
                 now - self.policy.last_observation <= 0.75)
        reflex = motor.latest_state in ("AVOIDING", "STATE_AVOIDING", "CLIFF_REVERSING",
                                        "STATE_CLIFF_REVERSING")
        # Dwell only accrues over intervals for which this runtime commanded STOP,
        # not while Uno is actively reversing/avoiding or sensor state is unknown.
        self.policy.advance_stop(stopped=False if reflex else None)
        if not camera_valid or detection_error or not fresh:
            self.sign_detections = []
            self.geometry = None
            self._interrupt(motor, "sensing unavailable")
            self.policy.advance_stop(stopped=False)
            self.status = "Traffic signs: sensing unavailable - holding"
            return self.status

        candidates = []
        for detection in self.sign_detections:
            try:
                candidates.append((detection, sign_geometry(detection, frame.shape)))
            except ValueError:
                continue
        if candidates:
            detected, geometry = max(candidates, key=lambda pair: pair[1]["height_frac"])
            self.latest_sign = detected["class_name"]
            self.geometry = geometry
        else:
            self.latest_sign = None
            self.geometry = None

        # Obstacle checks precede ordinary sign behavior. No PI_GO is emitted here.
        action, _ = _check_obstacle_ahead(obstacle_detections, frame.shape[1], frame.shape[0])
        if action == "STOP":
            if not self.direction and not self.policy.hold_reason() and not self.fault:
                self.policy.advance_stop(stopped=False)
                self.status = "Traffic signs: yielding to existing obstacle decision"
                return None
            self._interrupt(motor, "camera hazard")
            motor.pi_stop()
            self.policy.advance_stop(stopped=not reflex)
            self.status = "Camera hazard - STOP"
            return self.status
        if obstacle_time is None or now - obstacle_time > 0.5 or sonar_time is None or now - sonar_time > 0.8 or not all(
                is_valid_us(sonar.get(side)) for side in ("left", "right")):
            self._interrupt(motor, "clearance unavailable")
            self.policy.advance_stop(stopped=False)
            self.status = "Traffic signs: waiting for fresh clearance"
            return self.status
        latched = pi_latched or motor.latest_state in ("PI_STOPPED", "STATE_PI_STOPPED",
                                                       "CLIFF_STOPPED", "STATE_CLIFF_STOPPED")
        hold = self.policy.hold_reason()
        if reflex:
            if self.direction:
                self._interrupt(motor, "Arduino reflex")
            self.policy.advance_stop(stopped=False)
            self.status = "Traffic signs: yielding to Arduino reflex"
            return self.status  # Suppress ordinary driving, never release safety.
        if hold:
            self._interrupt(motor, hold)
            self.policy.advance_stop(stopped=True)
            self.status = "STOP sign detected - stopped" if not self.policy.no_entry else "No Entry detected - stopped"
            return self.status
        if self.fault:
            motor.stop()
            self.policy.advance_stop(stopped=True)
            self.status = self.fault
            return self.status
        if latched:
            self._interrupt(motor, "safety latch") if self.direction else None
            self.policy.advance_stop(stopped=True)
            self.status = "Traffic signs: yielding to safety latch"
            return None  # Existing mode owns obstacle latch release, without a turn.
        self.policy.advance_stop(stopped=False)
        clearance = (action == "CLEAR" and
                     all(sonar[side] >= 42.0 for side in ("left", "right")))
        if not clearance or not motors_enabled:
            if self.direction:
                self._interrupt(motor, "blocked path or motors disabled")
                self.status = self.fault
                return self.status
            self.status = "Traffic signs: waiting for clearance" if motors_enabled else "Traffic signs: bench safe"
            return None

        if not self.direction:
            pending = self.policy.snapshot()["pending_turn"]
            matching = [geometry for detection, geometry in candidates
                        if detection["class_name"] == pending and geometry["actionable"]]
            if pending and matching:
                self.direction = self.policy.take_turn()
                self.turn_mode = mode
                self.turn_started = now
                self.turn_until = now + (6.0 if mode == "lane" else 90 * self.turn_ms_per_degree / 1000.0)
                self.commanded_degrees = self.lane_confirmations = 0
                self.lane_departed = False
                self.last_pulse_stopped = 0.0
                self.pulse_until = 0.0
                self.completed_status = None
            elif pending:
                side = "Left" if pending == "turn_left" else "Right"
                self.status = side + " sign detected - approaching"
                return None
            else:
                self.status = self.completed_status or "Waiting for traffic signs"
                return None

        side = "Left" if self.direction == "turn_left" else "Right"
        sign = -1 if self.direction == "turn_left" else 1
        if mode != "lane":
            if not self.commanded_degrees:
                if not motor.turn(sign * 90, speed=self.turn_pwm):
                    self._interrupt(motor, "serial turn write failed")
                    self.status = self.fault
                    return self.status
                self.commanded_degrees = 90
            elif now >= self.turn_until:
                return self._complete(motor, side + " timed turn complete - heading unverified")
            self.status = side + " sign detected - taking " + side.lower()
            return self.status

        locked = (lane_result is not None and lane_result.status == "DUAL_LOCKED" and
                  abs(lane_result.steer_angle_deg) <= 6 and
                  abs(lane_result.lateral_offset_pct) <= 10)
        if self.commanded_degrees and not locked:
            self.lane_departed = True
        if now >= self.turn_until:
            self._interrupt(motor, "lane reacquisition timed out")
            self.status = self.fault
            return self.status
        if self.pulse_until and now < self.pulse_until:
            self.status = side + " sign detected - taking " + side.lower()
            return self.status
        if self.pulse_until:
            # Explicit stop bounds every pulse even if firmware TURN_CAL differs.
            motor.stop()
            self.pulse_until = 0.0
            self.last_pulse_stopped = now
            self.status = side + " turn - checking current lane"
            return self.status
        if obstacle_time < self.last_pulse_stopped:
            motor.stop()
            self.status = side + " turn - waiting for fresh obstacle check"
            return self.status
        if self.commanded_degrees >= 72 and self.lane_departed:
            self.lane_confirmations = self.lane_confirmations + 1 if locked else 0
            if self.lane_confirmations >= 3:
                return self._complete(motor, side + " turn complete - lane reacquired")
            if self.lane_confirmations or self.commanded_degrees >= 96:
                motor.stop()
                self.status = side + " turn - reacquiring lane"
                return self.status
        if self.commanded_degrees >= 96:
            motor.stop()
            self.status = side + " turn - reacquiring lane"
            return self.status
        if not motor.turn(sign * 12, speed=self.turn_pwm):
            self._interrupt(motor, "serial turn write failed")
            self.status = self.fault
            return self.status
        self.commanded_degrees += 12
        self.pulse_until = now + 12 * self.turn_ms_per_degree / 1000.0
        self.status = side + " sign detected - taking " + side.lower()
        return self.status


def run_pipeline(
    mode: str = "autonomous",
    port: str = "auto",
    baud: int = 115200,
    web_port: int = 8080,
    bench_safe: bool = False,
    invert_motors: bool = False,
    awb_mode: str = "daylight",
    enable_stream: bool = True,
    confidence_thresh: float = 0.4,
    traffic_sign_model: Optional[str] = None,
    traffic_turn_ms_per_degree: float = 10.0,
    traffic_turn_pwm: int = 210,
    disable_traffic_signs: bool = False,
):
    print(f"🚀 Autonomous RC Car — mode: {mode.upper()}")
    if mode not in ("autonomous", "lane", "follow"):
        print(f"❌ Unknown mode '{mode}'. Use --mode autonomous, --mode lane, or --mode follow")
        sys.exit(1)

    selected_sign_model = resolve_sign_model(traffic_sign_model, disabled=disable_traffic_signs)
    if selected_sign_model is None and not disable_traffic_signs:
        raise FileNotFoundError(
            "Bundled traffic-sign weights are missing. Extract the complete trained package, "
            "supply --traffic-sign-model PATH, or explicitly use --disable-traffic-signs "
            "to run without sign recognition."
        )
    system_state = SystemState()
    if selected_sign_model is not None:
        print(f"  Traffic-sign weights: {selected_sign_model}")
        from car.perception.traffic_signs import TrafficSignDetector
        system_state.traffic_signs = TrafficSignRuntime(TrafficSignDetector(str(selected_sign_model)),
                                                       turn_ms_per_degree=traffic_turn_ms_per_degree,
                                                       turn_pwm=traffic_turn_pwm)
        system_state.traffic_sign_status = {"enabled": True, "status": "Loading bundled/local sign model",
                                            "model": str(selected_sign_model)}
    else:
        reason = ("Traffic signs explicitly disabled" if disable_traffic_signs else
                  "Bundled traffic-sign weights missing; use --traffic-sign-model PATH")
        print("  " + reason)
        system_state.traffic_sign_status = {"enabled": False, "status": reason}
    system_state.current_mode = mode
    system_state.control_mode = mode
    system_state.motors_enabled = not bench_safe
    system_state.invert_motors = invert_motors
    system_state.current_awb = awb_mode

    # ── 1. Detector ──────────────────────────────────────────────────────────
    print("  📦 Loading YOLOv10n detector & pothole pipeline...")
    try:
        detector = Detector(conf=confidence_thresh)
    except Exception as e:
        print(f"❌ Failed to load detector: {e}")
        sys.exit(1)

    # ── 2. Hardware Interfaces (Motor Controller & Sensors) ──────────────────
    motor = MotorController(port=port, baud=baud)
    motor.invert_motors = invert_motors
    system_state.motor = motor

    if not motor.connect():
        print("⚠️  MotorController: Operating in DRY-RUN / SIMULATED mode.")
    else:
        # Await READY or bootloader settling
        ready_deadline = time.monotonic() + 3.0
        while time.monotonic() < ready_deadline:
            line = motor.read_telemetry()
            if line and "READY" in line:
                print(f"  🤖 Arduino ready: {line}")
                break
            time.sleep(0.05)
        # Configure Arduino runtime defaults per PROTOCOL.md
        motor._send("TUNE SET TELEM_STREAM 1")
        motor.pi_go()
        motor.status()

    sensors = SensorManager(motor_controller=motor)
    nav = NavigationSystem(controller=motor, sensors=sensors)
    throttler = CommandThrottler()

    # ── 3. Follow-me Components ──────────────────────────────────────────────
    tracker = ObjectTracker()
    steering_pid = PID(P=0.3, I=0.01, D=0.05)
    steering_pid.SetPoint = CAMERA_WIDTH / 2.0
    steering_pid.setSampleTime(0.05)
    steering_pid.setWindup(50)
    follow_state = "LOCK_ON"

    # ── 3b. Chart-Paper Lane Following Components ────────────────────────────
    lane_detector = ChartLaneDetector(color_mode=system_state.lane_color_mode)

    # ── 4. Camera Setup ──────────────────────────────────────────────────────
    use_picam2 = False
    picam2     = None
    cap        = None

    try:
        from picamera2 import Picamera2
        picam2 = Picamera2()
        system_state.cam_instance = picam2

        # Configure Picamera2 in BGR888 mode
        cfg = None
        if Transform is not None:
            try:
                cfg = picam2.create_video_configuration(
                    main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "BGR888"},
                    transform=Transform(hflip=1, vflip=1)
                )
            except Exception:
                pass
        if cfg is None:
            cfg = picam2.create_preview_configuration(
                main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"}
            )
        picam2.configure(cfg)
        picam2.start()
        set_awb_mode(picam2, awb_mode)
        use_picam2 = True
        print(f"  📷 CSI Camera initialised via Picamera2 ({CAMERA_WIDTH}x{CAMERA_HEIGHT}).")
    except Exception:
        print("  📷 Picamera2 unavailable; falling back to cv2.VideoCapture.")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)
        else:
            print("  📷 No physical camera detected; generating synthetic frames.")

    # ── 5. HTTP Streaming Server ─────────────────────────────────────────────
    server_thread = None
    httpd = None
    if enable_stream:
        streaming_output = StreamingOutput()
        system_state.streaming_output = streaming_output
        try:
            httpd = StreamingServer(('', web_port), DiagnosticHandler, system_state)
            httpd.lane_detector = lane_detector
            server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            server_thread.start()
            print(f"  🌐 Diagnostic HUD live stream active on: http://0.0.0.0:{web_port}")
        except Exception as e:
            print(f"⚠️ Failed to start HTTP streaming server on port {web_port}: {e}")

    # ── 6. Signal Handling & Teardown ────────────────────────────────────────
    def sig_handler(sig, frame):
        system_state.running = False

    if threading.current_thread() is threading.main_thread():
        try:
            signal.signal(signal.SIGINT, sig_handler)
            signal.signal(signal.SIGTERM, sig_handler)
        except (ValueError, AttributeError):
            pass

    print("✅ System ready. Press Ctrl+C to terminate safely.\n")

    frame_count       = 0
    pi_latched        = False
    cached_detections = []
    fps               = 0.0
    t_prev            = time.time()
    last_ping_time    = time.monotonic()
    last_obstacle_inference = None

    try:
        while system_state.running:
            # Sync latch state from external triggers (e.g. Web UI Emergency Stop / Latch Release)
            pi_latched = system_state.pi_latched

            # ── A. Drain incoming Arduino serial telemetry ────────────────────
            # Drains pending lines to ensure distance and state stay current
            drained_lines = motor.drain_telemetry(max_lines=40)
            for line in drained_lines:
                if "HEARTBEAT_TIMEOUT" in line:
                    print("⚠️  Arduino watchdog fired — link was quiet.")

            # Periodic ping to keep ATmega328P watchdog happy if quiet
            now_mono = time.monotonic()
            if now_mono - last_ping_time > 1.5:
                if motor.is_connected():
                    motor.status()
                last_ping_time = now_mono

            # ── B. Frame Capture ──────────────────────────────────────────────
            camera_valid = True
            if use_picam2 and picam2:
                frame_raw = picam2.capture_array()
                if len(frame_raw.shape) == 3 and frame_raw.shape[2] == 3:
                    frame = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2BGR) if cfg.get("main", {}).get("format") != "BGR888" else frame_raw.copy()
                else:
                    frame = frame_raw.copy()
            elif cap and cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    camera_valid = False
                    frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
            else:
                camera_valid = False
                frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
                cv2.putText(frame, "SIMULATED CAMERA STREAM", (int(CAMERA_WIDTH*0.2), int(CAMERA_HEIGHT*0.5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            frame_count += 1

            # ── C. Ultrasonic Obstacle Check ─────────────────────────────────
            us = sensors.read_ultrasonic()
            us_blocked = (is_valid_us(us.get("left")) and us["left"] < EMERGENCY_STOP_DISTANCE_CM) or \
                         (is_valid_us(us.get("right")) and us["right"] < EMERGENCY_STOP_DISTANCE_CM)

            # ── D. Inference (every 3rd frame, starting immediately on frame 1) ──
            fresh_inference = False
            if frame_count % 3 == 1:
                raw_dets = detector.detect(frame)
                cached_detections = [d for d in raw_dets if d.get("confidence", 0) >= confidence_thresh]
                fresh_inference = True
                last_obstacle_inference = time.monotonic()

            # ── E. Decision Execution ────────────────────────────────────────
            with system_state.lock:
                current_ctrl_mode = system_state.control_mode

            current_lane_result = None
            effective_mode = ("lane" if current_ctrl_mode == "lane" or
                              (current_ctrl_mode == "autonomous" and mode == "lane") else current_ctrl_mode)
            if system_state.traffic_signs is not None and effective_mode == "lane":
                sign_runtime = system_state.traffic_signs
                recent_sign_boxes = (sign_runtime.sign_detections
                                     if sign_runtime.policy.last_observation is not None and
                                     time.monotonic() - sign_runtime.policy.last_observation <= 0.75 else [])
                current_lane_result = lane_detector.detect(
                    frame, exclude_boxes=cached_detections + recent_sign_boxes)
                system_state.lane_result = current_lane_result
            sign_action = None
            if system_state.traffic_signs is not None:
                with system_state.lock:
                    reset = system_state.traffic_reset_requested
                    system_state.traffic_reset_requested = False
                sign_action = system_state.traffic_signs.step(
                    frame, motor, camera_valid=camera_valid,
                    obstacle_detections=cached_detections,
                    obstacle_time=last_obstacle_inference, sonar=us,
                    pi_latched=pi_latched, motors_enabled=system_state.motors_enabled,
                    reset=reset, mode=effective_mode, lane_result=current_lane_result,
                    sonar_time=motor.latest_sonar_time,
                )
                with system_state.lock:
                    system_state.traffic_sign_status = system_state.traffic_signs.snapshot()

            if sign_action is not None:
                if effective_mode == "follow":
                    _pause_follow_for_sign(tracker, steering_pid, system_state)
                action_label = sign_action
                pi_latched = pi_latched or motor.latest_state == "PI_STOPPED"
            elif current_ctrl_mode == "manual":
                # Manual control mode: vehicle is controlled via Web UI / D-Pad
                # Safety auto-brake: if front US is blocked, stop immediately!
                if us_blocked:
                    motor.stop()
                    action_label = f"🛑 MANUAL AUTO-BRAKE (L:{us.get('left', 999):.0f}cm R:{us.get('right', 999):.0f}cm)"
                else:
                    with system_state.lock:
                        man_act = system_state.manual_action
                        man_spd = system_state.manual_speed
                    action_label = f"🎮 MANUAL REMOTE: {man_act} (PWM {man_spd})"
            elif current_ctrl_mode == "lane" or (current_ctrl_mode == "autonomous" and mode == "lane"):
                pi_latched, action_label, lane_res = _run_lane(
                    motor=motor,
                    sensors=sensors,
                    lane_detector=lane_detector,
                    detections=cached_detections,
                    us_blocked=us_blocked,
                    frame=frame,
                    pi_latched=pi_latched,
                    motors_enabled=system_state.motors_enabled,
                    throttler=throttler,
                    sys_state=system_state,
                    lane_result=current_lane_result,
                )
            elif current_ctrl_mode == "follow" or mode == "follow":
                follow_state, pi_latched, action_label = _run_follow(
                    motor=motor,
                    sensors=sensors,
                    tracker=tracker,
                    pid=steering_pid,
                    detections=cached_detections,
                    us_blocked=us_blocked,
                    frame=frame,
                    follow_state=system_state.follow_state,
                    pi_latched=pi_latched,
                    motors_enabled=system_state.motors_enabled,
                    throttler=throttler,
                    fresh_inference=fresh_inference,
                    sys_state=system_state,
                )
                system_state.follow_state = follow_state
            else:
                pi_latched, action_label = _run_autonomous(
                    motor=motor,
                    sensors=sensors,
                    detections=cached_detections,
                    us_blocked=us_blocked,
                    frame=frame,
                    pi_latched=pi_latched,
                    motors_enabled=system_state.motors_enabled,
                    throttler=throttler,
                    fresh_inference=fresh_inference,
                )

            # Sync state with shared container
            system_state.pi_latched = pi_latched
            system_state.active_action_label = action_label

            # ── F. FPS Calculation ───────────────────────────────────────────
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now
            system_state.fps = fps

            # ── G. HUD Rendering & Streaming ─────────────────────────────────
            if enable_stream and system_state.streaming_output is not None:
                hud_frame = draw_hud(
                    frame=frame,
                    detections=cached_detections,
                    action_label=action_label,
                    latest_sonar=motor.latest_sonar,
                    latest_state=motor.latest_state,
                    port_name=motor.port,
                    fps=fps,
                    awb_mode=system_state.current_awb,
                    mode=system_state.control_mode if system_state.control_mode in ("manual", "follow", "lane") else mode,
                    motors_enabled=system_state.motors_enabled,
                    invert_motors=system_state.invert_motors,
                    pi_latched=pi_latched,
                    target_bbox=system_state.target_bbox,
                    target_similarity=system_state.target_similarity,
                    calibration_progress=system_state.calibration_progress,
                    last_seen_side=system_state.last_seen_side,
                    follow_state=system_state.follow_state,
                    non_target_persons=system_state.non_target_persons,
                    lane_result=system_state.lane_result,
                )
                if system_state.traffic_signs is not None:
                    hud_frame = draw_traffic_signs(hud_frame, system_state.traffic_signs.sign_detections)
                ok, jpg = cv2.imencode(".jpg", hud_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    system_state.streaming_output.write(jpg.tobytes())

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n🛑 KeyboardInterrupt caught. Shutting down...")
    finally:
        print("🛑 Executing graceful hardware release...")
        system_state.running = False
        if system_state.streaming_output is not None:
            with system_state.streaming_output.condition:
                system_state.streaming_output.condition.notify_all()
        if motor:
            motor.pi_go()           # Release any latches on Arduino
            time.sleep(0.05)
            motor.stop()            # Force stop all motor outputs
            motor.disconnect()
        if sensors:
            sensors.close()
        if cap is not None:
            cap.release()
        if picam2 is not None:
            try:
                picam2.stop()
                picam2.close()
            except Exception:
                pass
        if httpd:
            try:
                httpd.shutdown()
                httpd.server_close()
            except Exception:
                pass
        print("👋 Shutdown complete. Hardware released safely.")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Autonomous RC Car Controller with Headless Live Stream & Telemetry HUD"
    )
    parser.add_argument(
        "--mode",
        choices=["autonomous", "lane", "follow"],
        default="autonomous",
        help="Operating mode: 'autonomous' (avoid obstacles), 'lane' (chart-paper lane following), or 'follow' (follow closest person).",
    )
    parser.add_argument(
        "--port",
        default="auto",
        help="Serial port to Arduino Uno ('auto', '/dev/ttyUSB0', '/dev/ttyACM0', etc.).",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=115200,
        help="Serial baud rate (default: 115200 per PROTOCOL.md).",
    )
    parser.add_argument(
        "--stream-port",
        "--web-port",
        type=int,
        dest="web_port",
        default=8080,
        help="Port for HTTP MJPEG live stream and web dashboard (default: 8080).",
    )
    parser.add_argument(
        "--bench-safe",
        "--safe",
        action="store_true",
        help="Run in Safe Bench-Test mode (perception & stream active, motor commands suppressed).",
    )
    parser.add_argument(
        "--invert-motors",
        action="store_true",
        help="Invert motor drive polarity (corrects reversed chassis TT motor wiring).",
    )
    parser.add_argument(
        "--awb",
        default="daylight",
        help="Camera white-balance mode: 'daylight', 'auto', 'indoor', etc. (default: daylight).",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable HTTP video streaming server.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="YOLO detection confidence threshold (default: 0.40).",
    )

    parser.add_argument("--traffic-sign-model", default=None,
                        help="Override bundled four-class weights with an existing local .pt/NCNN artifact.")
    parser.add_argument("--disable-traffic-signs", action="store_true",
                        help="Explicitly disable bundled traffic-sign recognition.")

    parser.add_argument("--traffic-turn-ms-per-degree", type=float, default=10.0,
                        help="Traffic turn calibration matching Uno TURN_CAL (default 10ms/degree; unverified yaw).")
    parser.add_argument("--traffic-turn-pwm", type=int, default=210,
                        help="Traffic turn PWM 0..255 (default210 per documented chassis pivot setting).")
    args = parser.parse_args()

    run_pipeline(
        mode=args.mode,
        port=args.port,
        baud=args.baud,
        web_port=args.web_port,
        bench_safe=args.bench_safe,
        invert_motors=args.invert_motors,
        awb_mode=args.awb,
        enable_stream=not args.no_stream,
        confidence_thresh=args.conf,
        traffic_sign_model=args.traffic_sign_model,
        traffic_turn_ms_per_degree=args.traffic_turn_ms_per_degree,
        traffic_turn_pwm=args.traffic_turn_pwm,
        disable_traffic_signs=args.disable_traffic_signs,
    )


if __name__ == "__main__":
    main()
