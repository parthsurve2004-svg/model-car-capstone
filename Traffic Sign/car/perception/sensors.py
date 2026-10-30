"""
sensors.py — Arduino Telemetry Parser & IMU Interface
======================================================
Owned by perception-agent.

Ultrasonic distances: read from Arduino serial telemetry via motor_controller.
  The 2× HC-SR04P sensors are polled by the Arduino at 100ms intervals and
  published as '[TELEM] US LEFT=x RIGHT=y'. motor_controller.py parses these
  lines and keeps latest_sonar up to date. SensorManager.read_ultrasonic()
  just returns that dict — no direct GPIO access needed from the Pi.

IMU (ICM20948): read over I2C. Provides yaw/heading for dead-reckoning in
  navigation.py. The register-level DMP fusion logic is a stub until the
  hardware IMU wiring is confirmed operational.
"""

import time
from typing import Dict, Optional
from car.perception.config import IMU_I2C_BUS, IMU_I2C_ADDRESS
from car.control.motor_controller import MotorController


class SensorManager:
    """Aggregates ultrasonic distance data (via Arduino telemetry) and IMU heading."""

    def __init__(self, motor_controller: Optional[MotorController] = None):
        self.motor = motor_controller
        self._imu_bus = None
        self._init_imu()

    # ── IMU Initialisation ────────────────────────────────────────────────────

    def _init_imu(self):
        """Attempts to open I2C bus for the ICM20948 IMU."""
        try:
            import smbus2
            self._imu_bus = smbus2.SMBus(IMU_I2C_BUS)
        except Exception:
            # Off-device or smbus2 not installed — fall through silently.
            self._imu_bus = None

    # ── Ultrasonic Distances (from Arduino telemetry) ─────────────────────────

    def read_ultrasonic(self) -> Dict[str, float]:
        """
        Returns the latest ultrasonic distances in cm as parsed from Arduino
        telemetry '[TELEM] US LEFT=x RIGHT=y'.

        Keys:
            "left"  — HC-SR04P front-left sensor distance (cm)
            "right" — HC-SR04P front-right sensor distance (cm)

        Returns 999.0 for each axis if no Arduino is connected (safe default
        that will NOT trigger avoidance thresholds spuriously).

        Note: There are exactly 2 sensors in this system (front-left and
        front-right). There is NO center sensor. Architecture §4 shows the
        two sensors providing a combined ~60° forward coverage cone.
        """
        if self.motor is not None:
            return dict(self.motor.latest_sonar)  # Copy to avoid mutation
        return {"left": 999.0, "right": 999.0, "rear": 999.0}

    def is_obstacle_ahead(self, threshold_cm: float = 20.0) -> bool:
        """
        Returns True if either front-facing sensor reports an obstacle
        closer than threshold_cm. Used by navigation.py for safe-stop checks.
        """
        d = self.read_ultrasonic()
        return d["left"] < threshold_cm or d["right"] < threshold_cm

    # ── IMU Heading (ICM20948 over I2C) ──────────────────────────────────────

    def read_imu_heading(self) -> Optional[float]:
        """
        Reads current yaw/heading in degrees from the ICM20948 IMU over I2C.

        Returns:
            Heading in degrees (-180.0 to +180.0), or 0.0 if the IMU is
            unavailable (running off-device or before I2C is confirmed).

        Status: STUB — register-level DMP/AHRS fusion not yet implemented.
        Will return 0.0 until the ICM20948 I2C wiring is confirmed and
        the DMP output registers are mapped.
        """
        if self._imu_bus is None:
            return 0.0

        # TODO (integration-agent): implement ICM20948 DMP read here.
        # Expected register sequence:
        #   1. Read DMP output (yaw) from ICM20948 FIFO or direct register
        #   2. Apply soft-iron calibration offsets
        #   3. Wrap to -180..+180 range
        return 0.0  # Unverified / stub — no hardware read yet

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def close(self):
        """Closes any open sensor interfaces."""
        if self._imu_bus is not None:
            try:
                self._imu_bus.close()
            except Exception:
                pass
            self._imu_bus = None
