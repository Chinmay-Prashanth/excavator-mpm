"""
Excavator MPM Simulation
========================
Loads the Idea_3 excavator URDF (3-link arm: base → shoulder → bucket) and
simulates a scripted digging motion into a granular MPM sand bed.

Architecture (mirrors newton/examples/mpm/example_mpm_anymal.py):
  - Single model: URDF bodies + ground plane + MPM particles in one builder.
  - Kinematic arm: joints written directly each frame via eval_fk; no rigid-body
    solver needed.  mpm_solver.setup_collider(body_mass=zeros) tells the MPM to
    treat all bodies as kinematic (infinite effective mass).
  - Two alternating states (state_0 / state_1) — each substep writes particles
    into the unused state then swaps, preventing in-place aliasing.
  - _project_outside called after each substep to keep particles above ground.

Usage:
    cd ~/ecorobotic/excavator
    python simulate_excavator.py --viewer gl

    # With a USD exported from Isaac Sim:
    python simulate_excavator.py --usd /path/to/excavator.usd --viewer gl

GL viewer controls:
    WASD      — move camera
    Left-drag — orbit
    Scroll    — zoom
    Space     — pause / resume
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM

# ── Paths ──────────────────────────────────────────────────────────────────────
URDF_PATH = Path("/home/chinmay/ASUPHD/EcoRobotics/Spring/V1/urdf/Idea_3.urdf")

# ── Scale factor ───────────────────────────────────────────────────────────────
# All URDF dimensions are multiplied by SCALE.  10 → arm ~3.7 m long, which is
# easy to see in the default Newton GL camera.
SCALE = 10.0


class ExcavatorExample:
    """
    Kinematic excavator arm + MPM granular sand — anymal-style single model.

    Dig cycle (8 s, repeating):
      Phase 1 (0–2 s)  — lower arm into sand
      Phase 2 (2–4 s)  — curl bucket to scoop
      Phase 3 (4–6 s)  — lift arm with material
      Phase 4 (6–8 s)  — open bucket and reset
    """

    def __init__(self, viewer, usd_path: str | None = None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 1          # matches granular example: one implicit MPM step per frame
        self.sim_dt = self.frame_dt    # full frame_dt; finite_difference velocity = Δbody/frame_dt ✓
        self.sim_time = 0.0
        self.viewer = viewer

        # ── Single builder for EVERYTHING (arm + ground + particles) ──────────
        # This matches the anymal example: particles live in the same model as
        # the robot so that state_0 holds both body_q and particle_q.
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.default_shape_cfg.ke = 1.0e4
        builder.default_shape_cfg.kd = 1.0e2
        builder.default_shape_cfg.kf = 5.0e2
        builder.default_shape_cfg.mu = 0.6

        base_xform = wp.transform(
            wp.vec3(0.0, 0.0, 0.22 * SCALE), wp.quat_identity()
        )

        if usd_path:
            print(f"Loading USD: {usd_path}")
            builder.add_usd(
                usd_path, xform=base_xform,
                floating=False, collapse_fixed_joints=False,
            )
        else:
            print(f"Loading URDF: {URDF_PATH}  (scale={SCALE})")
            builder.add_urdf(
                str(URDF_PATH), xform=base_xform,
                floating=False, collapse_fixed_joints=False, scale=SCALE,
            )

        # ── Find joint DOFs (URDF locks them; unlock here) ────────────────────
        self._shoulder_dof = None
        self._bucket_dof = None
        for joint_idx, label in enumerate(builder.joint_label):
            dof = builder.joint_qd_start[joint_idx]
            if "shoulder_joint" in label:
                builder.joint_limit_lower[dof] = -math.pi * 0.6
                builder.joint_limit_upper[dof] =  math.pi * 0.6
                self._shoulder_dof = dof
                print(f"  shoulder_joint → DOF {dof}")
            elif "bucket_joint" in label:
                builder.joint_limit_lower[dof] = -math.pi * 0.7
                builder.joint_limit_upper[dof] =  math.pi * 0.6
                self._bucket_dof = dof
                print(f"  bucket_joint   → DOF {dof}")

        if self._shoulder_dof is None or self._bucket_dof is None:
            print("WARNING: joints not found. Labels:", builder.joint_label)

        # ── Disable MPM particle collision on non-bucket bodies ───────────────
        # Pattern from anymal example: only the "active" contact body collides
        # with particles.  Base and shoulder meshes span the entire sand bed at
        # their scaled size (>1 m) — registering them as MPM colliders causes
        # explosive contact forces.
        for body in range(builder.body_count):
            label = (builder.body_label[body] or "").lower()
            if "bucket" not in label:
                for shape in builder.body_shapes[body]:
                    builder.shape_flags[shape] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES)
                    print(f"  body[{body}] '{label}' shape[{shape}]: particle collision disabled")

        # ── Initial pose (arm elevated above sand bed) ─────────────────────────
        if self._shoulder_dof is not None:
            builder.joint_q[self._shoulder_dof] = 0.8
        if self._bucket_dof is not None:
            builder.joint_q[self._bucket_dof] = 0.2

        # ── Ground plane ──────────────────────────────────────────────────────
        builder.add_ground_plane()

        # ── MPM particle attributes — must be registered BEFORE add_particle_grid
        SolverImplicitMPM.register_custom_attributes(builder)

        # ── Sand particles (into the SAME builder) ────────────────────────────
        voxel_size = 0.010 * SCALE
        self._emit_particles(builder, voxel_size)

        # ── Finalize single model ──────────────────────────────────────────────
        self.model = builder.finalize()
        self.model.mpm.hardening.fill_(0.0)  # Chrono CRM: no strain hardening (cohesionless sand)
        print(f"Sand particles: {self.model.particle_count}")
        print(f"Bodies: {self.model.body_count}  Shapes: {self.model.shape_count}")

        # ── Two alternating states (granular-example pattern) ─────────────────
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # Initialise body transforms from initial joint_q
        newton.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, self.state_0
        )
        self._joint_q_np = self.model.joint_q.numpy().copy()

        # Manual slider control state (toggled via GUI checkbox)
        self._manual = False
        self._slider_shoulder = float(self._joint_q_np[self._shoulder_dof])
        self._slider_bucket   = float(self._joint_q_np[self._bucket_dof])

        # ── MPM solver ────────────────────────────────────────────────────────
        mpm_options = SolverImplicitMPM.Config()
        mpm_options.voxel_size            = voxel_size
        mpm_options.tolerance             = 1.0e-5
        mpm_options.grid_type             = "sparse"  # dynamic — follows particles, no fixed bounding box
        mpm_options.grid_padding          = 10
        mpm_options.max_active_cell_count = -1        # unlimited — let solver allocate as needed
        mpm_options.strain_basis          = "P0"
        mpm_options.max_iterations        = 250       # solver default — do not reduce for large particle counts
        mpm_options.critical_fraction     = 0.0
        mpm_options.air_drag              = 1.0
        mpm_options.collider_velocity_mode = "finite_difference"

        self.mpm_solver = SolverImplicitMPM(self.model, mpm_options)

        # Kinematic bodies (anymal pattern): body_mass=0 means MPM never applies
        # reaction forces to bodies; arm moves purely from eval_fk.
        # state_in.body_q is read each step automatically, so the MPM always
        # sees the current arm positions.
        # collider order: ground plane (-1) first, then bucket body.
        # collider_thicknesses inflates the SDF so thin bucket walls (<<voxel_size)
        # still catch particles — prevents penetration and lift-phase explosions.
        self.mpm_solver.setup_collider(
            body_mass=wp.zeros_like(self.model.body_mass),
            body_q=self.state_0.body_q,
            collider_thicknesses=[0.0, 0.75 * voxel_size],  # ground: none, bucket: 3/4 voxel (holds particles in floor)
        )

        # ── Viewer ────────────────────────────────────────────────────────────
        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self._camera_framed = False

    # ── Particle emission ─────────────────────────────────────────────────────

    def _emit_particles(self, builder: newton.ModelBuilder, voxel_size: float):
        """
        Sand bed in front of and below the initial bucket reach.

        Bed covers X = 0.10*S → 0.42*S, Y = ±0.15*S, Z = 0 → 0.10*S (scaled).

        IMPORTANT: bed_lo.z is raised by one particle radius after cell_size is
        computed.  This ensures the lowest jittered particles (jitter offset =
        −radius) land at z ≥ 0 and never penetrate the ground plane.
        """
        S = SCALE
        particles_per_cell = 3.0
        density = 1000.0  # kg/m³  granular example default

        bed_lo = np.array([0.10 * S, -0.15 * S, 0.0])
        bed_hi = np.array([0.42 * S,  0.15 * S, 0.10 * S])

        bed_res = np.array(
            np.ceil(particles_per_cell * (bed_hi - bed_lo) / voxel_size), dtype=int
        )
        cell_size = (bed_hi - bed_lo) / bed_res
        cell_vol  = float(np.prod(cell_size))
        radius    = float(np.max(cell_size) * 0.5)
        mass      = float(cell_vol * density)

        # Shift bottom of bed up by one radius so jitter (±radius) stays at z ≥ 0.
        bed_lo[2] = radius

        builder.add_particle_grid(
            pos=wp.vec3(*bed_lo.tolist()),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=int(bed_res[0]) + 1,
            dim_y=int(bed_res[1]) + 1,
            dim_z=int(bed_res[2]) + 1,
            cell_x=float(cell_size[0]),
            cell_y=float(cell_size[1]),
            cell_z=float(cell_size[2]),
            mass=mass,
            jitter=2.0 * radius,
            radius_mean=radius,
            custom_attributes={
                "mpm:friction":              0.68,   # granular example default
                "mpm:young_modulus":         1.0e15, # rigid limit — granular example default
                "mpm:poisson_ratio":         0.30,   # granular example default
                "mpm:yield_pressure":        1.0e12, # caps compressive yield — critical for stability
                "mpm:tensile_yield_ratio":   0.0,    # no tensile strength (cohesionless sand)
            },
        )

    # ── Scripted trajectory ───────────────────────────────────────────────────

    def _scripted_targets(self) -> tuple[float, float]:
        """
        Cosine-interpolated 4-phase dig cycle → (shoulder_rad, bucket_rad).

        Joint axis (0, −1, 0):
          positive angle → tip toward +Z (arm up / bucket open)
          negative angle → tip toward −Z (arm down / bucket curls)
        """
        CYCLE = 8.0
        t = self.sim_time % CYCLE

        def cos_lerp(a: float, b: float, tt: float) -> float:
            tt = max(0.0, min(1.0, tt))
            s = 0.5 - 0.5 * math.cos(math.pi * tt)
            return a + (b - a) * s

        if t < 2.0:        # lower arm into sand
            return cos_lerp(0.8, -0.35, t / 2.0), 0.2
        elif t < 4.0:      # curl bucket to scoop
            return -0.35,  cos_lerp(0.2, -1.0, (t - 2.0) / 2.0)
        elif t < 6.0:      # lift arm with material
            return cos_lerp(-0.35, 0.6, (t - 4.0) / 2.0), -1.0
        else:              # open bucket, return to start
            return cos_lerp(0.6, 0.8, (t - 6.0) / 2.0), cos_lerp(-1.0, 0.2, (t - 6.0) / 2.0)

    # ── Simulation step ───────────────────────────────────────────────────────

    def step(self):
        # Compute target joint angles for this frame
        if self._manual:
            # Slew-rate limiter: arm chases slider target at max 0.8 rad/s.
            # Prevents large per-frame jumps from creating explosive
            # finite_difference velocities on contact with particles.
            max_delta = 1.5 * self.frame_dt  # ~0.025 rad per frame @ 60 fps
            prev_sh = self._joint_q_np[self._shoulder_dof]
            prev_bk = self._joint_q_np[self._bucket_dof]
            shoulder = prev_sh + float(np.clip(self._slider_shoulder - prev_sh, -max_delta, max_delta))
            bucket   = prev_bk + float(np.clip(self._slider_bucket   - prev_bk, -max_delta, max_delta))
        else:
            shoulder, bucket = self._scripted_targets()
        curr_q = self._joint_q_np.copy()
        if self._shoulder_dof is not None:
            curr_q[self._shoulder_dof] = shoulder
        if self._bucket_dof is not None:
            curr_q[self._bucket_dof] = bucket
        # Update arm FK for this frame (1 substep: finite_difference velocity = Δbody/frame_dt = true velocity)
        self.state_0.joint_q.assign(curr_q)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.state_1.body_q.assign(self.state_0.body_q)
        self.state_1.body_qd.assign(self.state_0.body_qd)

        self.mpm_solver.step(self.state_0, self.state_1, contacts=None, control=None, dt=self.sim_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0

        # NaN diagnostic — print once when particles first go NaN
        pq = self.state_0.particle_q.numpy()
        if np.any(np.isnan(pq)):
            sh_deg = math.degrees(curr_q[self._shoulder_dof])
            bk_deg = math.degrees(curr_q[self._bucket_dof])
            nan_count = int(np.isnan(pq).any(axis=-1).sum())
            print(f"[NaN] t={self.sim_time:.3f}s  shoulder={sh_deg:.1f}°  bucket={bk_deg:.1f}°  "
                  f"nan_particles={nan_count}/{self.model.particle_count}")

        self._joint_q_np = curr_q
        self.sim_time += self.frame_dt

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        # log_state renders arm bodies AND particles (via show_particles=True)
        self.viewer.log_state(self.state_0)

        # Auto-frame camera on the first rendered frame.
        # pitch=-25° looks DOWN (positive pitch looks up in Z-up).
        if not self._camera_framed and hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(
                pos=wp.vec3(0.0, 0.0, 0.0),
                pitch=-25.0,
                yaw=135.0,
            )
            if hasattr(self.viewer, "_frame_camera_on_model"):
                self.viewer._frame_camera_on_model()
            self._camera_framed = True

        self.viewer.end_frame()

    # ── GUI (Newton auto-wires this to the side panel) ────────────────────────

    def gui(self, imgui):
        import math
        _changed, self._manual = imgui.checkbox("Manual Control", self._manual)
        imgui.separator()
        if self._manual:
            _c, self._slider_shoulder = imgui.slider_float(
                "Shoulder (deg)", math.degrees(self._slider_shoulder),
                -90.0, 90.0, "%.1f°"
            )
            self._slider_shoulder = math.radians(self._slider_shoulder)

            _c, self._slider_bucket = imgui.slider_float(
                "Bucket (deg)", math.degrees(self._slider_bucket),
                -180.0, 180.0, "%.1f°"
            )
            self._slider_bucket = math.radians(self._slider_bucket)
        else:
            imgui.text_disabled("Scripted dig cycle running...")

    def test_final(self):
        voxel_size = self.mpm_solver.voxel_size
        newton.examples.test_particle_state(
            self.state_0,
            "all particles remain above ground",
            lambda q, qd: q[2] > -voxel_size,
        )



# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--usd", type=str, default=None, metavar="PATH",
        help="USD file from Isaac Sim (overrides URDF)",
    )
    viewer, args = newton.examples.init(parser)

    if wp.get_device().is_cpu:
        print("WARNING: GPU recommended for MPM.")

    example = ExcavatorExample(viewer, usd_path=args.usd)
    newton.examples.run(example, args)
