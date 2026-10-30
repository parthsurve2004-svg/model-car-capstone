from car.control.traffic_sign_policy import TrafficSignPolicy

class Clock:
    now = 0.0
    def __call__(self):
        return self.now


def observe(policy, clock, names, count=3):
    for _ in range(count):
        policy.observe([{"class_name": name} for name in names])
        clock.now += .25


def test_stop_confirms_three_fresh_samples_and_does_not_repeat():
    clock = Clock()
    p = TrafficSignPolicy(clock)
    observe(p, clock, ["stop"], 2)
    assert p.hold_reason() is None
    observe(p, clock, ["stop"], 1)
    assert p.hold_reason()
    p.advance_stop(stopped=True)
    for _ in range(20):
        observe(p, clock, ["stop"], 1)
        p.advance_stop(stopped=True)
    assert p.hold_reason() is None
    observe(p, clock, [], 5)
    observe(p, clock, ["stop"])
    assert p.hold_reason()


def test_confirmation_gap_and_empty_observation_reset():
    clock = Clock()
    p = TrafficSignPolicy(clock)
    observe(p, clock, ["no_entry"], 2)
    clock.now += 1
    observe(p, clock, ["no_entry"], 1)
    assert not p.no_entry
    observe(p, clock, [], 1)
    observe(p, clock, ["no_entry"], 2)
    assert not p.no_entry


def test_no_entry_latches_until_reset_and_visible_sign_does_not_rearm():
    clock = Clock()
    p = TrafficSignPolicy(clock)
    observe(p, clock, ["no_entry"])
    clock.now += 20
    assert p.hold_reason()
    p.reset()
    observe(p, clock, ["no_entry"])
    assert p.hold_reason() is None


def test_turn_requires_explicit_take_and_expires_or_conflicts():
    clock = Clock()
    p = TrafficSignPolicy(clock)
    observe(p, clock, ["turn_left"])
    assert p.snapshot()["pending_turn"] == "turn_left"
    observe(p, clock, ["turn_left", "turn_right"], 1)
    assert p.take_turn() is None
    p = TrafficSignPolicy(clock)
    observe(p, clock, ["turn_right"])
    clock.now += 11
    assert p.take_turn() is None
    p = TrafficSignPolicy(clock)
    observe(p, clock, ["turn_left"])
    assert p.take_turn() == "turn_left"
    assert p.take_turn() is None
