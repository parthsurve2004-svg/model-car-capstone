"""
motor_controller.py — Pi-Side Serial Bridge to Arduino Uno
===========================================================
Owned by integration-agent.
Implements the Pi-side half of the serial handshake defined in arduino/PROTOCOL.md.
Transport (USB-Serial or GPIO-UART) is behind a thin interface so it can be swapped
without touching callers. See config.py for SERIAL_PORT and SERIAL_BAUD.

Command methods mirror the three-tier priority stack in PROTOCOL.md:
  Tier 1: pi_stop(), pi_go(), pi_avoid()          ← Pi camera latch
  Tier 3: drive(), reverse(), turn(), stop(), ping() ← Normal driving

Telemetry is parsed passively in read_telemetry():
  [TELEM] US LEFT=x RIGHT=y  → updates latest_sonar dict
  [TELEM] STATE=...           → updates latest_state string
  [ACK] ...                   → returned as-is for callers that care about ACK
"""

import sys
import threading
import time
from typing import Optional, Dict, List, Any
try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None

from car.perception.config import SERIAL_PORT, SERIAL_BAUD


class MotorController:
    """Manages serial communications and sends commands to the Arduino Uno."""

    def __init__(self, port: str = SERIAL_PORT, baud: int = SERIAL_BAUD, timeout: float = 0.5):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self._serial: Optional[Any] = None if serial is None else None
        self.invert_motors: bool = False
        self._lock = threading.Lock()

        # Latest ultrasonic readings parsed from [TELEM] US LEFT=x RIGHT=y REAR=z
        # Architecture uses 2 front-facing sensors (LEFT / RIGHT) and optional rear sensor.
        self.latest_sonar: Dict[str, float] = {"left": 999.0, "right": 999.0, "rear": 999.0}

        # Latest Arduino state string ("IDLE", "DRIVING", "REVERSING", etc.)
        self.latest_state: str = "UNKNOWN"
        # Receipt time of a complete front-sonar pair, not physical ping time.
        self.latest_sonar_time: Optional[float] = None

    # ── Connection management ─────────────────────────────────────────────────

    def auto_connect(self, candidate_ports: Optional[List[str]] = None) -> bool:
        """Attempts connection across a list of candidate serial ports."""
        if serial is None:
            print("⚠️  MotorController: pyserial is not installed. Operating in dry-run mode.", file=sys.stderr)
            self.port = "SIMULATED"
            return False

        if candidate_ports is None:
            detected_ports = []
            if hasattr(serial, "tools") and hasattr(serial.tools, "list_ports"):
                try:
                    detected_ports = [p.device for p in serial.tools.list_ports.comports()]
                except Exception:
                    pass

            default_candidates = [
                "/dev/ttyUSB0",
                "/dev/ttyACM0",
                "/dev/ttyUSB1",
                "/dev/ttyACM1",
                "/dev/ttyAMA0",
            ]
            candidate_ports = detected_ports + [p for p in default_candidates if p not in detected_ports]

        for p in candidate_ports:
            try:
                s = serial.Serial(p, self.baud, timeout=self.timeout)
                time.sleep(2.0)
                s.reset_input_buffer()
                s.reset_output_buffer()
                self._serial = s
                self.port = p
                print(f"  🔌 Serial connected: {self.port} @ {self.baud} baud")
                return True
            except Exception:
                continue

        print("⚠️  MotorController: No physical Arduino detected. Operating in dry-run mode.")
        self.port = "SIMULATED"
        self._serial = None
        return False

    def connect(self) -> bool:
        """Opens the serial link and waits for Arduino bootloader to settle."""
        if self.port in ("auto", "AUTO", None):
            return self.auto_connect()

        if serial is None:
            print("⚠️  MotorController: pyserial is not installed. Operating in dry-run mode.", file=sys.stderr)
            self.port = "SIMULATED"
            return False

        try:
            self._serial = serial.Serial(self.port, self.baud, timeout=self.timeout)
            # ATmega328P resets when DTR toggles on serial open.
            # Allow 2 s for bootloader + setup() to complete before sending commands.
            time.sleep(2.0)
            print(f"  🔌 Serial connected: {self.port} @ {self.baud} baud")
            return True
        except Exception as e:
            print(f"⚠️  MotorController: Could not open {self.port}: {e}", file=sys.stderr)
            self._serial = None
            self.port = "SIMULATED"
            return False

    def disconnect(self):
        """Safely stops motors and closes serial link."""
        if self.is_connected():
            self.stop()
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None

    def is_connected(self) -> bool:
        return self._serial is not None and self._serial.is_open

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    # ── Low-level send ────────────────────────────────────────────────────────

    def _send(self, command: str) -> bool:
        """
        Sends an ASCII command terminated with newline.
        Returns True on success, False if disconnected or serial error.
        Callers that need ACK confirmation should call read_telemetry() after.
        """
        with self._lock:
            if not self.is_connected():
                return False
            try:
                line = (command.strip() + "\n").encode("utf-8")
                self._serial.write(line)
                self._serial.flush()
                return True
            except Exception as e:
                print(f"⚠️  Serial write error: {e}", file=sys.stderr)
                return False

    # ── Tier 1 — Pi Camera Latch Commands (highest priority) ─────────────────

    def pi_stop(self) -> bool:
        """
        Latched emergency stop from Pi camera pipeline.
        Freezes all motors instantly — even mid-avoidance.
        Only pi_go() releases the latch.
        Arduino responds: [ACK] PI_STOP  +  [TELEM] STATE=PI_STOPPED
        """
        self.latest_state = "PI_STOPPED"
        return self._send("PI_STOP")

    def pi_go(self) -> bool:
        """
        Releases the Pi camera latch.
        Arduino returns to STATE_IDLE.
        Arduino responds: [ACK] PI_GO
        """
        self.latest_state = "IDLE"
        return self._send("PI_GO")

    def pi_avoid(self, direction: str) -> bool:
        """
        Sends a one-shot directional hint for the Arduino's next Tier-2
        avoidance maneuver. If the hint is set when ultrasonics trigger,
        the Arduino turns in the hinted direction instead of deciding locally.

        Args:
            direction: "LEFT" or "RIGHT" (case-insensitive).
        """
        d = direction.strip().upper()
        if d not in ("LEFT", "RIGHT"):
            raise ValueError(f"pi_avoid: direction must be LEFT or RIGHT, got {direction!r}")
        if self.invert_motors:
            d = "RIGHT" if d == "LEFT" else "LEFT"
        return self._send(f"PI_AVOID {d}")

    # ── Tier 3 — Normal Driving Commands ─────────────────────────────────────

    def drive(self, speed: int, duration_ms: int) -> bool:
        """
        Drive forward at PWM speed (0–255) for duration_ms.
        A new DRIVE replaces any in-progress timed action (no queue).
        """
        speed = max(0, min(255, speed))
        cmd = f"REV {speed} {duration_ms}" if self.invert_motors else f"DRIVE {speed} {duration_ms}"
        return self._send(cmd)

    def reverse(self, speed: int, duration_ms: int) -> bool:
        """Drive reverse at PWM speed (0–255) for duration_ms."""
        speed = max(0, min(255, speed))
        cmd = f"DRIVE {speed} {duration_ms}" if self.invert_motors else f"REV {speed} {duration_ms}"
        return self._send(cmd)

    def turn(self, angle_deg: int, speed: int = 150) -> bool:
        """
        Skid-steer turn by angle_deg.
        Positive = right, negative = left. Range: -180 to +180.
        Duration is computed on the Arduino from TURN_MS_PER_DEGREE.
        """
        angle_deg = max(-180, min(180, angle_deg))
        speed = max(0, min(255, speed))
        eff_angle = -angle_deg if self.invert_motors else angle_deg
        return self._send(f"TURN {eff_angle} {speed}")

    def stop(self) -> bool:
        """
        Non-latched immediate stop → Arduino returns to IDLE.
        Does NOT lock out further commands (unlike pi_stop).
        """
        return self._send("STOP")

    def ping(self) -> bool:
        """
        Liveness check. Arduino responds with [ACK] PONG.
        Use to confirm the serial link is alive before starting a run.
        """
        return self._send("PING")

    def status(self) -> bool:
        """
        Queries current state and ultrasonic readings.
        Arduino responds with current status telemetry.
        """
        return self._send("STATUS")

    # ── Meta / Runtime Parameter Tuning (PROTOCOL.md) ─────────────────────────

    def tune_set(self, param: str, value: Any) -> bool:
        """
        Updates a runtime parameter on the Arduino without reflashing.
        See PROTOCOL.md for valid parameter names and ranges.
        """
        return self._send(f"TUNE SET {param} {value}")

    def tune_get(self) -> bool:
        """Requests dump of all active tunable parameters."""
        return self._send("TUNE GET")

    def tune_recal(self) -> bool:
        """Triggers fresh ground baseline calibration cycle for angled ultrasonic sensors."""
        return self._send("TUNE RECAL")

    def tune_save(self) -> bool:
        """Commits current parameters to ATmega328P EEPROM."""
        return self._send("TUNE SAVE")

    def tune_load(self) -> bool:
        """Reloads stored parameters from ATmega328P EEPROM."""
        return self._send("TUNE LOAD")

    def tune_reset(self) -> bool:
        """Resets parameters to compile-time factory defaults."""
        return self._send("TUNE RESET")

    # ── Telemetry Reader ──────────────────────────────────────────────────────

    def read_telemetry(self) -> Optional[str]:
        """
        Reads and parses one pending telemetry line from the Arduino.

        Side-effects:
          - Parses '[TELEM] US LEFT=x RIGHT=y' → updates self.latest_sonar
          - Parses '[TELEM] STATE=XXX ...'     → updates self.latest_state

        Returns:
          The raw line string (so callers can log / inspect ACKs), or None
          if no data is available or serial is disconnected.
        """
        with self._lock:
            if not self.is_connected():
                return None
            try:
                iw = getattr(self._serial, "in_waiting", 0)
                if callable(iw):
                    iw = iw()
                if iw == 0:
                    return None
                line = self._serial.readline().decode("utf-8", errors="replace").strip()
                if not line:
                    return None

                # ── Parse ultrasonic telemetry ────────────────────────────────────
                # Format: [TELEM] US LEFT=15.3 RIGHT=42.7 or [TELEM] CLIFF LEFT=...
                if "[TELEM] US" in line or "[TELEM] CLIFF" in line or ("LEFT=" in line and "RIGHT=" in line):
                    self._parse_us_telemetry(line)

                # ── Parse state telemetry ─────────────────────────────────────────
                # Format: [TELEM] STATE=DRIVING SPEED=180
                if "[TELEM] STATE=" in line or "STATE=" in line:
                    # Extract state token e.g. "STATE=DRIVING"
                    for token in line.split():
                        if token.startswith("STATE="):
                            self.latest_state = token.split("=", 1)[1]
                            break
                elif "[TELEM] AVOID" in line:
                    self.latest_state = "AVOIDING"
                elif "[TELEM] DONE" in line:
                    self.latest_state = "IDLE"
                elif "[ACK] BLOCKED (" in line:
                    try:
                        blocked_state = line.split("(", 1)[1].split(")", 1)[0].strip()
                        if blocked_state:
                            self.latest_state = blocked_state
                    except (IndexError, ValueError):
                        pass

                return line

            except Exception:
                return None

    def _parse_us_telemetry(self, line: str):
        """
        Parses '[TELEM] US LEFT=15.3 RIGHT=42.7 REAR=80.0' and updates latest_sonar.
        Silently skips malformed lines.
        """
        try:
            norm = (
                line.replace("US_LEFT=", "LEFT=")
                .replace("US_RIGHT=", "RIGHT=")
                .replace("US_REAR=", "REAR=")
                .replace("BACK=", "REAR=")
            )
            left_val = None
            right_val = None
            rear_val = None
            for token in norm.split():
                if token.startswith("LEFT="):
                    left_val = float(token.split("=", 1)[1])
                elif token.startswith("RIGHT="):
                    right_val = float(token.split("=", 1)[1])
                elif token.startswith("REAR="):
                    rear_val = float(token.split("=", 1)[1])
            if left_val is not None:
                self.latest_sonar["left"] = left_val
            if right_val is not None:
                self.latest_sonar["right"] = right_val
            if rear_val is not None:
                self.latest_sonar["rear"] = rear_val
            if left_val is not None and right_val is not None:
                self.latest_sonar_time = time.monotonic()
        except (ValueError, IndexError):
            pass  # Malformed telemetry — ignore silently

    def is_rear_clear(self, threshold_cm: float = 30.0) -> bool:
        """Returns True if rear sensor indicates safe clearance (> threshold_cm)."""
        r = self.latest_sonar.get("rear", 999.0)
        return r > threshold_cm

    def drain_telemetry(self, max_lines: int = 20):
        """
        Drains up to max_lines of pending telemetry, updating internal state.
        Useful after sending a command to flush ACKs before the next operation.

        Returns:
            List of all raw lines read.
        """
        lines = []
        for _ in range(max_lines):
            line = self.read_telemetry()
            if line is None:
                break
            lines.append(line)
        return lines

    def wait_for_ack(self, command_prefix: str, timeout_s: float = 2.0) -> Optional[str]:
        """
        Blocks until an [ACK] line matching command_prefix is received,
        or until timeout_s seconds elapse.

        Args:
            command_prefix: e.g. "DRIVE", "PI_STOP", "PONG"
        Returns:
            The matching ACK line, or None on timeout.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            line = self.read_telemetry()
            if line and line.startswith("[ACK]") and command_prefix in line:
                return line
            time.sleep(0.01)
        return None
