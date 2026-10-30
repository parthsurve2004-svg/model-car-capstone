"""
navigation.py — IMU-Aided Dead-Reckoning Point-to-Point Navigation
===================================================================
Owned by integration-agent.

No wheel encoders on the chassis. Navigation relies on:
  - Calibrated PWM speed + timed duration for distance estimation.
  - ICM20948 IMU heading for turn accuracy (stub until IMU wiring confirmed).

Usage (from car/main.py in autonomous mode):
  nav = NavigationSystem(controller=motor, sensors=sensors)
  success = nav.navigate_step(target_distance_m=1.0, target_heading_deg=45)

Obstacle abort:
  Before driving, checks both front-facing ultrasonic sensors (left + right).
  If either sensor reports < 25cm, the step is aborted immediately.
  During a drive step, the same check fires every 50ms.

Speed baseline (unverified / simulated — must be calibrated on real chassis):
  At PWM 180, car covers approximately 0.35 m/s on flat ground.
"""

import time
from car.control.motor_controller import MotorController
from car.perception.sensors import SensorManager

# Distance calibration (unverified — calibrate on real chassis at demo PWM)
_SPEED_M_PER_S = 0.35    # Estimated metres/second at PWM 180 on flat ground
_DEFAULT_PWM   = 180

# Obstacle check threshold during navigation steps (slightly tighter than
# the 20cm Arduino Tier 2 threshold, so the Pi aborts before Arduino reacts)
_OBSTACLE_ABORT_CM = 25.0


class NavigationSystem:
    """Executes turn-then-drive waypoint steps with real-time obstacle abort."""

    def __init__(self, controller: MotorController, sensors: SensorManager):
        self.controller    = controller
        self.sensors       = sensors
        self.speed_m_per_s = _SPEED_M_PER_S
        self.default_pwm   = _DEFAULT_PWM

    def navigate_step(self, target_distance_m: float, target_heading_deg: float) -> bool:
        """
        Executes one waypoint step: turn to heading, then drive to distance.

        Args:
            target_distance_m:   Distance in metres to drive after turning.
            target_heading_deg:  Heading angle relative to current heading.
                                 Positive = right, negative = left (-180..+180).

        Returns:
            True  — step completed normally.
            False — step aborted due to obstacle detection.
        """

        # ── Pre-flight obstacle check ─────────────────────────────────────────
        if self.sensors.is_obstacle_ahead(threshold_cm=_OBSTACLE_ABORT_CM):
            print("🚨 navigate_step: Obstacle detected ahead — aborting before move.")
            self.controller.stop()
            return False

        # ── Step 1: Turn to target heading ────────────────────────────────────
        if abs(target_heading_deg) > 5.0:
            self.controller.turn(int(target_heading_deg), speed=160)
            # Wait for the Arduino to complete the turn
            # Duration estimate: TURN_MS_PER_DEGREE is 10ms on the Arduino (uncalibrated)
            turn_wait_s = abs(target_heading_deg) * 0.012 + 0.2  # small margin
            time.sleep(turn_wait_s)

        # ── Step 2: Drive forward for estimated duration ──────────────────────
        if target_distance_m <= 0:
            return True

        duration_ms = int((target_distance_m / self.speed_m_per_s) * 1000)
        self.controller.drive(self.default_pwm, duration_ms)

        # ── Monitor for obstacles while driving ───────────────────────────────
        start_t      = time.monotonic()
        drive_time_s = duration_ms / 1000.0

        while (time.monotonic() - start_t) < drive_time_s:
            if self.sensors.is_obstacle_ahead(threshold_cm=_OBSTACLE_ABORT_CM):
                print("🚨 navigate_step: Emergency obstacle abort during drive!")
                self.controller.pi_stop()
                return False
            time.sleep(0.05)

        return True

    def navigate_waypoints(self, waypoints: list) -> bool:
        """
        Executes a sequence of (distance_m, heading_deg) waypoints in order.

        Args:
            waypoints: List of (distance_m, heading_deg) tuples.

        Returns:
            True if all waypoints completed, False if any step was aborted.
        """
        for i, (distance_m, heading_deg) in enumerate(waypoints):
            print(f"  📍 Waypoint {i + 1}/{len(waypoints)}: "
                  f"{distance_m:.1f}m @ {heading_deg:+.0f}°")
            if not self.navigate_step(distance_m, heading_deg):
                print(f"  ❌ Aborted at waypoint {i + 1}.")
                return False
        print("  ✅ All waypoints complete.")
        return True
