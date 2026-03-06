#!/usr/bin/env python3
"""
Excavator MPM Simulation — ROS2 Humble
=======================================
Same Newton/Warp MPM physics as simulate_excavator.py, but joint targets come
from a ROS2 topic instead of a scripted trajectory.

Topics
------
  /excavator/joint_commands  [sensor_msgs/JointState]   <- subscribe
      .name     : ['shoulder_joint', 'bucket_joint']
      .position : [shoulder_rad, bucket_rad]

  /excavator/joint_states    [sensor_msgs/JointState]   -> publish @ 60 Hz
      (smoothed actual joint positions fed to MPM)

  /excavator/particles       [sensor_msgs/PointCloud2]  -> publish @ 10 Hz
      frame_id  : 'world'  (Z-up, matches Newton world frame)
      fields    : x, y, z  (float32)

Timeout
-------
  If joint_commands are received and then stop for 30 s, joints freeze at the
  last commanded position and a warning is shown in the GUI and console.
  If commands were never received, joints hold at 0 deg and wait indefinitely.

Usage
-----
    source /opt/ros/humble/setup.bash
    cd ~/ecorobotic/excavator
    ~/.pyenv/versions/ecorobotic-newton/bin/python simulate_excavator_ros2.py --viewer gl

RViz quick-start
----------------
    rviz2
    -> Panels > Displays > Add > By topic > /excavator/particles > PointCloud2
    -> Global Options > Fixed Frame : world
    -> PointCloud2 > Style : Points,  Size : 0.003 m

Send a command manually
-----------------------
    ros2 topic pub /excavator/joint_commands sensor_msgs/msg/JointState \
      '{name: [shoulder_joint, bucket_joint], position: [0.5, 1.0]}'
"""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM

# ── ROS2 imports (graceful fallback if not sourced) ───────────────────────────
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState, PointCloud2, PointField
    _ROS2_AVAILABLE = True
except ImportError:
    _ROS2_AVAILABLE = False
    print("[WARN] rclpy not found — running without ROS2 (joints hold at 0 deg).")

# ── Paths ─────────────────────────────────────────────────────────────────────
USD_PATH  = Path(__file__).parent / "arm.usd"
URDF_PATH = Path("/home/chinmay/ASUPHD/EcoRobotics/Spring/V1/urdf/Idea_3.urdf")
SCALE     = 10.0  # URDF only; USD uses S=1.0

# ── ROS2 constants ────────────────────────────────────────────────────────────
COMMAND_TIMEOUT  = 30.0  # seconds without a new command -> freeze joints
CLOUD_PUBLISH_HZ = 10    # PointCloud2 publish rate (10 Hz = ~4.4 MB/s for 363k pts)


# ── ROS2 node ─────────────────────────────────────────────────────────────────

class ExcavatorNode(Node):
    """
    Thin ROS2 node bridging joint commands and state between ROS2 and Newton.
    Runs in a daemon thread; float assignment is GIL-atomic in CPython so no
    explicit lock is needed for the two command values.
    """

    def __init__(self):
        super().__init__("excavator_mpm")

        # Command targets — written by ROS thread, read by sim thread
        self.cmd_shoulder: float = 0.0
        self.cmd_bucket:   float = 0.0
        self._last_cmd_wall: float | None = None  # wall-clock time of last msg

        self._sub = self.create_subscription(
            JointState,
            "/excavator/joint_commands",
            self._on_joint_cmd,
            10,
        )
        self._pub_joints = self.create_publisher(JointState,  "/excavator/joint_states", 10)
        self._pub_cloud  = self.create_publisher(PointCloud2, "/excavator/particles",    10)

        self.get_logger().info("ExcavatorNode ready — waiting for /excavator/joint_commands")

    # ── Subscriber ────────────────────────────────────────────────────────────

    def _on_joint_cmd(self, msg: JointState):
        try:
            idx_s = msg.name.index("shoulder_joint")
            idx_b = msg.name.index("bucket_joint")
        except ValueError:
            self.get_logger().warn(
                "joint_commands must contain 'shoulder_joint' and 'bucket_joint'",
                throttle_duration_sec=5.0,
            )
            return
        self.cmd_shoulder   = float(msg.position[idx_s])
        self.cmd_bucket     = float(msg.position[idx_b])
        self._last_cmd_wall = time.time()

    # ── Timeout helpers ───────────────────────────────────────────────────────

    @property
    def ever_received(self) -> bool:
        return self._last_cmd_wall is not None

    @property
    def timed_out(self) -> bool:
        """True only if we previously received commands and 30 s have elapsed."""
        if self._last_cmd_wall is None:
            return False
        return (time.time() - self._last_cmd_wall) > COMMAND_TIMEOUT

    @property
    def seconds_since_cmd(self) -> float:
        if self._last_cmd_wall is None:
            return 0.0
        return time.time() - self._last_cmd_wall

    # ── Publishers ────────────────────────────────────────────────────────────

    def publish_joint_states(self, shoulder: float, bucket: float):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name         = ["shoulder_joint", "bucket_joint"]
        msg.position     = [shoulder, bucket]
        self._pub_joints.publish(msg)

    def publish_pointcloud(self, particle_q_np: np.ndarray):
        """
        particle_q_np : (N, 3) float32 — particle world positions from Newton.
        Published as unordered PointCloud2 (height=1) in frame 'world' (Z-up).
        Point step = 12 bytes (3 x float32), no padding, is_dense=True.
        """
        pts = particle_q_np.astype(np.float32, copy=False)
        n   = len(pts)

        msg                 = PointCloud2()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.height          = 1
        msg.width           = n
        msg.fields          = [
            PointField(name="x", offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step   = 12
        msg.row_step     = 12 * n
        msg.data         = pts.tobytes()
        msg.is_dense     = True
        self._pub_cloud.publish(msg)


# ── Simulation class ───────────────────────────────────────────────────────────

class ExcavatorROS2Example:
    """
    Newton MPM excavator sim driven by ROS2 joint commands.
    No scripted trajectory — targets come from ExcavatorNode each frame.
    Exponential smoothing (alpha=0.08) still applied to protect MPM from
    velocity impulses caused by discrete command updates.
    """

    def __init__(self, viewer, ros_node: ExcavatorNode | None, usd_path: str | None = None):
        self.fps          = 60
        self.frame_dt     = 1.0 / self.fps
        self.sim_substeps = 1
        self.sim_dt       = self.frame_dt
        self.sim_time     = 0.0
        self.viewer       = viewer
        self.ros_node     = ros_node

        # ── Builder ───────────────────────────────────────────────────────────
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.default_shape_cfg.ke = 1.0e4
        builder.default_shape_cfg.kd = 1.0e2
        builder.default_shape_cfg.kf = 5.0e2
        builder.default_shape_cfg.mu = 0.6

        S = 1.0 if usd_path else SCALE

        # base_z=0.23m: at shoulder=-90 deg, bucket_joint Z = 0.005m (at sand surface)
        base_xform = wp.transform(wp.vec3(0.0, 0.0, 0.23 * S), wp.quat_identity())

        if usd_path:
            print(f"Loading USD: {usd_path}")
            builder.add_usd(
                usd_path, xform=base_xform,
                floating=False, collapse_fixed_joints=False,
                override_root_xform=True,
                ignore_paths=["/World/Idea_3/root_joint"],
                skip_mesh_approximation=True,
            )
        else:
            print(f"Loading URDF: {URDF_PATH}")
            builder.add_urdf(
                str(URDF_PATH), xform=base_xform,
                floating=False, collapse_fixed_joints=False, scale=SCALE,
            )

        # ── Joint DOFs ────────────────────────────────────────────────────────
        self._shoulder_dof = None
        self._bucket_dof   = None
        for joint_idx, label in enumerate(builder.joint_label):
            dof = builder.joint_qd_start[joint_idx]
            if "shoulder_joint" in label:
                builder.joint_limit_lower[dof] = -math.pi / 2       # -90 deg
                builder.joint_limit_upper[dof] =  math.pi / 2       # +90 deg
                self._shoulder_dof = dof
                print(f"  shoulder_joint -> DOF {dof}")
            elif "bucket_joint" in label:
                builder.joint_limit_lower[dof] = -math.radians(135) # -135 deg dump
                builder.joint_limit_upper[dof] =  math.radians(135) # +135 deg scoop
                self._bucket_dof = dof
                print(f"  bucket_joint   -> DOF {dof}")

        if self._shoulder_dof is None or self._bucket_dof is None:
            print("WARNING: joints not found. Labels:", builder.joint_label)

        # ── Bucket collision: only the collision mesh shape ───────────────────
        for body in range(builder.body_count):
            label    = (builder.body_label[body] or "").lower()
            is_bucket = "bucket" in label
            for shape in builder.body_shapes[body]:
                slabel  = (builder.shape_label[shape] if shape < len(builder.shape_label) else "") or ""
                has_col = "collision" in slabel.lower()
                has_vis = "visual"    in slabel.lower()
                active  = is_bucket and (has_col or (not has_vis and not has_col))
                if not active:
                    builder.shape_flags[shape] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES)

        # ── Initial pose: both joints at 0 deg ────────────────────────────────
        if self._shoulder_dof is not None:
            builder.joint_q[self._shoulder_dof] = 0.0
        if self._bucket_dof is not None:
            builder.joint_q[self._bucket_dof] = 0.0

        # ── Ground plane + Y containment walls ───────────────────────────────
        builder.add_ground_plane()
        builder.add_shape_plane(plane=(0.0, -1.0, 0.0,  0.5 * S), width=0.0, length=0.0)
        builder.add_shape_plane(plane=(0.0,  1.0, 0.0,  0.5 * S), width=0.0, length=0.0)

        # ── Particles ─────────────────────────────────────────────────────────
        SolverImplicitMPM.register_custom_attributes(builder)
        voxel_size = 0.008 * S  # ~363k particles, dia=2.7mm, ~45 FPS on RTX 4080 Laptop
        self._emit_particles(builder, voxel_size, S)

        # ── Finalize ──────────────────────────────────────────────────────────
        self.model = builder.finalize()
        self.model.mpm.hardening.fill_(0.0)
        print(f"Sand particles : {self.model.particle_count}")
        print(f"Bodies         : {self.model.body_count}   Shapes: {self.model.shape_count}")

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self._joint_q_np = self.model.joint_q.numpy().copy()

        # Smoothed joint values that chase ROS2 command targets each frame
        self._smooth_shoulder = 0.0
        self._smooth_bucket   = 0.0

        # ── MPM solver ────────────────────────────────────────────────────────
        mpm_options = SolverImplicitMPM.Config()
        mpm_options.voxel_size             = voxel_size
        mpm_options.tolerance              = 1.0e-5
        mpm_options.grid_type              = "sparse"
        mpm_options.grid_padding           = 0
        mpm_options.max_active_cell_count  = -1
        mpm_options.strain_basis           = "P0"
        mpm_options.max_iterations         = 250
        mpm_options.critical_fraction      = 0.0
        mpm_options.air_drag               = 1.0
        mpm_options.collider_velocity_mode = "finite_difference"

        self.mpm_solver = SolverImplicitMPM(self.model, mpm_options)
        self.mpm_solver.setup_collider(
            body_mass=wp.zeros_like(self.model.body_mass),
            body_q=self.state_0.body_q,
            collider_thicknesses=[0.0, 1.0 * voxel_size],
        )

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self._camera_framed = False

        self._frame_count = 0
        self._cloud_every = max(1, round(self.fps / CLOUD_PUBLISH_HZ))  # publish every 6 frames

    # ── Particle emission ──────────────────────────────────────────────────────

    def _emit_particles(self, builder, voxel_size, S):
        ppc     = 3.0
        density = 1000.0

        bed_lo = np.array([0.10 * S, -0.1 * S, 0.0])
        bed_hi = np.array([0.42 * S,  0.1 * S, 0.10 * S])

        bed_res = np.array(np.ceil(ppc * (bed_hi - bed_lo) / voxel_size), dtype=int)
        cell_sz = (bed_hi - bed_lo) / bed_res
        radius  = float(np.max(cell_sz) * 0.5)
        mass    = float(np.prod(cell_sz) * density)
        bed_lo[2] = radius  # raise floor so jitter stays above z=0

        builder.add_particle_grid(
            pos=wp.vec3(*bed_lo.tolist()), rot=wp.quat_identity(), vel=wp.vec3(0.0),
            dim_x=int(bed_res[0]) + 1,
            dim_y=int(bed_res[1]) + 1,
            dim_z=int(bed_res[2]) + 1,
            cell_x=float(cell_sz[0]),
            cell_y=float(cell_sz[1]),
            cell_z=float(cell_sz[2]),
            mass=mass, jitter=2.0 * radius, radius_mean=radius,
            custom_attributes={
                "mpm:friction":            0.68,
                "mpm:young_modulus":       1.0e15,
                "mpm:poisson_ratio":       0.30,
                "mpm:yield_pressure":      1.0e12,
                "mpm:tensile_yield_ratio": 0.0,
            },
        )

    # ── Simulation step ────────────────────────────────────────────────────────

    def step(self):
        self._frame_count += 1

        # Read ROS2 targets — freeze at last smooth value if timed out
        if self.ros_node is not None and not self.ros_node.timed_out:
            tgt_shoulder = self.ros_node.cmd_shoulder
            tgt_bucket   = self.ros_node.cmd_bucket
        else:
            tgt_shoulder = self._smooth_shoulder
            tgt_bucket   = self._smooth_bucket

        # Exponential smoothing: 8% per frame toward target, capped per-frame delta
        alpha     = 0.08
        max_delta = 2.0 * self.frame_dt
        for attr, tgt in (("_smooth_shoulder", tgt_shoulder),
                           ("_smooth_bucket",   tgt_bucket)):
            prev = getattr(self, attr)
            raw  = prev + alpha * (tgt - prev)
            setattr(self, attr, prev + float(np.clip(raw - prev, -max_delta, max_delta)))

        # Apply FK
        curr_q = self._joint_q_np.copy()
        if self._shoulder_dof is not None:
            curr_q[self._shoulder_dof] = self._smooth_shoulder
        if self._bucket_dof is not None:
            curr_q[self._bucket_dof] = self._smooth_bucket

        self.state_0.joint_q.assign(curr_q)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.state_1.body_q.assign(self.state_0.body_q)
        self.state_1.body_qd.assign(self.state_0.body_qd)

        self.mpm_solver.step(self.state_0, self.state_1, contacts=None, control=None, dt=self.sim_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0

        # NaN guard
        pq = self.state_0.particle_q.numpy()
        if np.any(np.isnan(pq)):
            nan_n = int(np.isnan(pq).any(axis=-1).sum())
            print(f"[NaN] t={self.sim_time:.3f}s  nan_particles={nan_n}/{self.model.particle_count}")

        # ROS2 publish — joints every frame, cloud at CLOUD_PUBLISH_HZ
        if self.ros_node is not None:
            self.ros_node.publish_joint_states(self._smooth_shoulder, self._smooth_bucket)
            if self._frame_count % self._cloud_every == 0:
                self.ros_node.publish_pointcloud(pq)

        self._joint_q_np = curr_q
        self.sim_time   += self.frame_dt

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if not self._camera_framed and hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(0.0, 0.0, 0.0), pitch=-25.0, yaw=135.0)
            if hasattr(self.viewer, "_frame_camera_on_model"):
                self.viewer._frame_camera_on_model()
            self._camera_framed = True
        self.viewer.end_frame()

    # ── GUI ───────────────────────────────────────────────────────────────────

    def gui(self, imgui):
        # ROS2 connection status
        if self.ros_node is None:
            imgui.text_disabled("ROS2: unavailable (rclpy not found)")
        elif self.ros_node.timed_out:
            imgui.text(f"ROS2: TIMED OUT ({self.ros_node.seconds_since_cmd:.0f}s) -- joints frozen")
        elif not self.ros_node.ever_received:
            imgui.text("ROS2: waiting for joint_commands ...")
        else:
            imgui.text(f"ROS2: active  (last cmd {self.ros_node.seconds_since_cmd:.1f}s ago)")

        imgui.separator()
        imgui.text(f"Shoulder : {math.degrees(self._smooth_shoulder):+.1f} deg")
        imgui.text(f"Bucket   : {math.degrees(self._smooth_bucket):+.1f} deg")
        imgui.separator()
        imgui.text_disabled(
            "Publish cmd example:\n"
            "ros2 topic pub /excavator/joint_commands\n"
            "  sensor_msgs/msg/JointState\n"
            "  '{name: [shoulder_joint, bucket_joint],\n"
            "    position: [-1.57, 2.35]}'"
        )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--usd", type=str,
        default=str(USD_PATH) if USD_PATH.exists() else None,
        metavar="PATH",
        help="USD file (default: arm.usd in same folder if present)",
    )
    viewer, args = newton.examples.init(parser)

    if wp.get_device().is_cpu:
        print("WARNING: GPU recommended for MPM.")

    # Start ROS2 spin in daemon thread so GL viewer can own the main thread
    ros_node   = None
    ros_thread = None

    if _ROS2_AVAILABLE:
        rclpy.init(args=None)
        ros_node = ExcavatorNode()

        def _spin():
            try:
                rclpy.spin(ros_node)
            except Exception:
                pass

        ros_thread = threading.Thread(target=_spin, daemon=True, name="rclpy-spin")
        ros_thread.start()
        print("[ROS2] Node started — spinning in background thread.")
    else:
        print("[ROS2] Not available — joints will hold at 0 deg.")

    try:
        example = ExcavatorROS2Example(viewer, ros_node, usd_path=args.usd)
        newton.examples.run(example, args)
    finally:
        if _ROS2_AVAILABLE and ros_node is not None:
            ros_node.destroy_node()
            rclpy.shutdown()
        print("[SIM] Done.")
