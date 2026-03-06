#!/usr/bin/env python3
"""
Excavator Trajectory Sender — ROS2 Humble
==========================================
State-machine trajectory controller that drives the MPM simulation through
the 4-phase dig cycle using sim_time for synchronisation.

Sync design
-----------
  - Subscribes to /excavator/sim_time  [std_msgs/Float64]
    Sim publishes its internal sim_time every frame (60 Hz).
    Trajectory targets are computed from sim_time — not wall clock — so
    trajectory and sim are always perfectly aligned even if the sim is slow,
    paused (Space), or restarted.

  - Subscribes to /excavator/joint_states  [sensor_msgs/JointState]
    Actual smoothed joint positions from the sim.
    A phase only advances when:
      (a) the interpolation fraction has reached 1.0  AND
      (b) actual joints are within PHASE_DONE_TOL of the phase target.
    This prevents skipping a phase if the sim is lagging behind.

Topics
------
  Subscribes: /excavator/sim_time     [std_msgs/Float64]
  Subscribes: /excavator/joint_states [sensor_msgs/JointState]
  Publishes:  /excavator/joint_commands [sensor_msgs/JointState]

Phases (total 10 s cycle)
--------------------------
  Phase 1  0–2 s    bucket  0° → +135°  (cosine),  shoulder 0°
  Phase 2  2–5 s    shoulder 0° → -90°  (linear),  bucket holds +135°
  Phase 3  5–8 s    both return to 0°   (cosine, same fraction)
  Phase 4  8–9 s    bucket  0° → -135°  (cosine),  shoulder 0°
  Phase 5  9–10 s   bucket -135° → 0°   (cosine),  shoulder 0°

Usage
-----
    # Terminal 1
    ros2 launch excavator_mpm excavator_full.launch.py

    # or start both manually
    ros2 launch excavator_mpm excavator.launch.py
    ros2 run excavator_mpm trajectory_sender
"""

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64


# ── Trajectory constants ──────────────────────────────────────────────────────
BUCKET_SCOOP   = math.radians(135)   # +135°: scoop position
BUCKET_DUMP    = -math.radians(135)  # -135°: dump position
SHOULDER_DIG   = -math.pi / 2        # -90°:  arm pointing down

PHASE_DONE_TOL = math.radians(3.0)   # 3° tolerance to consider a phase "reached"


# ── Phase definition ──────────────────────────────────────────────────────────

class Phase:
    def __init__(self, name, duration,
                 s_start, s_end,
                 b_start, b_end,
                 s_interp="cosine", b_interp="cosine"):
        self.name      = name
        self.duration  = duration
        self.s_start   = s_start;  self.s_end = s_end
        self.b_start   = b_start;  self.b_end = b_end
        self.s_interp  = s_interp
        self.b_interp  = b_interp

    def targets(self, frac: float) -> tuple[float, float]:
        """Return (shoulder, bucket) for fraction frac in [0, 1]."""
        return (
            _interp(self.s_start, self.s_end, frac, self.s_interp),
            _interp(self.b_start, self.b_end, frac, self.b_interp),
        )

    def reached(self, shoulder: float, bucket: float) -> bool:
        """True when actual joints are within tolerance of this phase's end."""
        return (abs(shoulder - self.s_end) < PHASE_DONE_TOL and
                abs(bucket   - self.b_end) < PHASE_DONE_TOL)


def _interp(a, b, t, mode):
    t = max(0.0, min(1.0, t))
    if mode == "linear":
        return a + (b - a) * t
    # cosine (default)
    return a + (b - a) * (0.5 - 0.5 * math.cos(math.pi * t))


PHASES = [
    Phase("1 — bucket scoop",    duration=2.0,
          s_start=0.0,          s_end=0.0,
          b_start=0.0,          b_end=BUCKET_SCOOP,
          s_interp="cosine",    b_interp="cosine"),

    Phase("2 — shoulder dig",    duration=3.0,
          s_start=0.0,          s_end=SHOULDER_DIG,
          b_start=BUCKET_SCOOP, b_end=BUCKET_SCOOP,
          s_interp="linear",    b_interp="cosine"),   # linear shoulder descent

    Phase("3 — return home",     duration=3.0,
          s_start=SHOULDER_DIG, s_end=0.0,
          b_start=BUCKET_SCOOP, b_end=0.0,
          s_interp="cosine",    b_interp="cosine"),   # both sync to 0°

    Phase("4 — bucket dump",     duration=1.0,
          s_start=0.0,          s_end=0.0,
          b_start=0.0,          b_end=BUCKET_DUMP,
          s_interp="cosine",    b_interp="cosine"),

    Phase("5 — reset",           duration=1.0,
          s_start=0.0,          s_end=0.0,
          b_start=BUCKET_DUMP,  b_end=0.0,
          s_interp="cosine",    b_interp="cosine"),
]


# ── ROS2 node ─────────────────────────────────────────────────────────────────

class TrajectorySender(Node):
    """
    Closed-loop trajectory state machine.
    Uses sim_time (not wall clock) for interpolation fractions.
    Waits for actual joints to reach phase target before advancing.
    """

    def __init__(self):
        super().__init__("excavator_trajectory_sender")

        self.declare_parameter("cycles", 0)   # 0 = infinite
        self._max_cycles = self.get_parameter("cycles").value

        # State machine
        self._phase_idx       = 0
        self._phase_sim_start = None   # sim_time when current phase began
        self._cycle_count     = 0
        self._done            = False

        # Latest actual joint positions (from sim feedback)
        self._actual_shoulder = 0.0
        self._actual_bucket   = 0.0

        # Publishers / Subscribers
        self._pub = self.create_publisher(
            JointState, "/excavator/joint_commands", 10)

        self.create_subscription(
            Float64, "/excavator/sim_time", self._on_sim_time, 10)

        self.create_subscription(
            JointState, "/excavator/joint_states", self._on_joint_states, 10)

        self.get_logger().info(
            f"Trajectory sender ready — waiting for /excavator/sim_time\n"
            f"  Phases: {len(PHASES)}   Cycle: "
            f"{'infinite' if self._max_cycles == 0 else str(self._max_cycles) + ' cycle(s)'}"
        )

    # ── Subscriber callbacks ───────────────────────────────────────────────────

    def _on_joint_states(self, msg: JointState):
        try:
            self._actual_shoulder = msg.position[msg.name.index("shoulder_joint")]
            self._actual_bucket   = msg.position[msg.name.index("bucket_joint")]
        except (ValueError, IndexError):
            pass

    def _on_sim_time(self, msg: Float64):
        if self._done:
            return

        sim_time = msg.data
        phase    = PHASES[self._phase_idx]

        # First message ever — start phase 1
        if self._phase_sim_start is None:
            self._phase_sim_start = sim_time
            self.get_logger().info(f"Phase {phase.name}  started")

        phase_elapsed = sim_time - self._phase_sim_start
        frac          = phase_elapsed / phase.duration   # may exceed 1.0

        shoulder_cmd, bucket_cmd = phase.targets(frac)
        self._publish(shoulder_cmd, bucket_cmd)

        # ── Terminal joint readout (throttled to ~5 Hz) ───────────────────────
        self.get_logger().info(
            f"[{phase.name}  {frac*100:5.1f}%]  "
            f"shoulder  cmd={math.degrees(shoulder_cmd):+7.2f}°  "
            f"act={math.degrees(self._actual_shoulder):+7.2f}°    "
            f"bucket    cmd={math.degrees(bucket_cmd):+7.2f}°  "
            f"act={math.degrees(self._actual_bucket):+7.2f}°",
            throttle_duration_sec=0.2,
        )

        # ── Phase completion check ────────────────────────────────────────────
        # Advance only when interpolation is done AND actual joints have arrived
        if frac >= 1.0 and phase.reached(self._actual_shoulder, self._actual_bucket):
            self._advance_phase(sim_time)

    def _advance_phase(self, sim_time: float):
        old_phase = PHASES[self._phase_idx]
        self._phase_idx += 1

        if self._phase_idx >= len(PHASES):
            # Completed one full cycle
            self._cycle_count += 1
            self.get_logger().info(
                f"Phase {old_phase.name}  done  →  cycle {self._cycle_count} complete")

            if self._max_cycles > 0 and self._cycle_count >= self._max_cycles:
                self.get_logger().info(
                    f"All {self._max_cycles} cycle(s) complete — sending home.")
                self._publish(0.0, 0.0)
                self._done = True
                return

            self._phase_idx = 0   # loop back to phase 1

        self._phase_sim_start = sim_time
        new_phase = PHASES[self._phase_idx]
        self.get_logger().info(
            f"Phase {old_phase.name}  done  →  Phase {new_phase.name}  started")

    def _publish(self, shoulder: float, bucket: float):
        msg             = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name         = ["shoulder_joint", "bucket_joint"]
        msg.position     = [shoulder, bucket]
        self._pub.publish(msg)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = TrajectorySender()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopped — sending home.")
    finally:
        node._publish(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
