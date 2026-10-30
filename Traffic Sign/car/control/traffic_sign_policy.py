"""Nonblocking four-sign policy. No hardware access or inferred junction geometry.

Observations must be fresh detector results, never reused cached boxes. Runtime
geometry and safety clearance determine when a confirmed turn becomes actionable.
"""
import time


class TrafficSignPolicy:
    def __init__(self, clock=time.monotonic, confirmations=3, max_gap=0.75,
                 stop_seconds=3.0, pending_seconds=10.0):
        self.clock = clock
        self.confirmations = confirmations
        self.max_gap = max_gap
        self.stop_seconds = stop_seconds
        self.pending_seconds = pending_seconds
        self.last_observation = None
        self.no_entry = False
        self.stop_remaining = 0.0
        self._dwell_tick = self.clock()
        self._was_stopped = False
        self.pending_turn = None
        self.pending_until = 0.0
        self._counts = {}
        self._consumed = set()
        self._absent = {}
        self._absent_since = {}

    def observe(self, detections):
        now = self.clock()
        if self.last_observation is None or now - self.last_observation > self.max_gap:
            self._counts.clear()
            self._absent.clear()
            self._absent_since.clear()
        self.last_observation = now
        present = {d.get("class_name") for d in detections}
        if {"turn_left", "turn_right"} <= present:
            self.pending_turn = None
        for sign in ("no_entry", "stop", "turn_left", "turn_right"):
            if sign not in present:
                self._counts[sign] = 0
                self._absent[sign] = self._absent.get(sign, 0) + 1
                self._absent_since.setdefault(sign, now)
                if self._absent[sign] >= 3 and now - self._absent_since[sign] >= 1.0:
                    self._consumed.discard(sign)
                continue
            self._absent.pop(sign, None)
            self._absent_since.pop(sign, None)
            self._counts[sign] = self._counts.get(sign, 0) + 1
            if self._counts[sign] < self.confirmations or sign in self._consumed:
                continue
            self._consumed.add(sign)
            if sign == "no_entry":
                self.no_entry = True
                self.pending_turn = None
            elif sign == "stop":
                self.stop_remaining = self.stop_seconds
                self._dwell_tick = now
                self._was_stopped = False
            elif not self.no_entry and not {"turn_left", "turn_right"} <= present:
                self.pending_turn = sign
                self.pending_until = now + self.pending_seconds

    def reset(self):
        """Explicitly release NO_ENTRY; never touches an obstacle stop latch."""
        self.no_entry = False
        self.pending_turn = None
        for sign in ("turn_left", "turn_right"):
            self._consumed.discard(sign)
            self._counts[sign] = 0

    def clear_pending(self):
        self.pending_turn = None

    def advance_stop(self, stopped=None):
        """Accumulate dwell only across intervals commanded stopped by runtime."""
        now = self.clock()
        if self._was_stopped and stopped is not False:
            self.stop_remaining = max(0.0, self.stop_remaining - (now - self._dwell_tick))
        self._dwell_tick = now
        if stopped is not None:
            self._was_stopped = stopped

    def hold_reason(self):
        if self.no_entry:
            return "NO_ENTRY (operator reset required)"
        if self.stop_remaining > 0:
            return "STOP sign dwell"
        return None

    def take_turn(self):
        now = self.clock()
        if self.hold_reason() or now >= self.pending_until:
            if now >= self.pending_until:
                self.pending_turn = None
            return None
        turn = self.pending_turn
        self.pending_turn = None
        return turn

    def snapshot(self):
        if self.clock() >= self.pending_until:
            self.pending_turn = None
        return {"hold": self.hold_reason(), "pending_turn": self.pending_turn,
                "no_entry": self.no_entry, "stop_remaining": round(self.stop_remaining, 2)}
