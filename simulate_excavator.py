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
USD_PATH  = Path(__file__).parent / "arm.usd"

# ── Scale factor (URDF only) ───────────────────────────────────────────────────
# URDF meshes are in small units; ×10 → arm ~3.7 m, easy to see in GL viewer.
# USD (arm.usd) is already in real meters — no scaling applied there.
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

        # USD is in real meters (no scaling); URDF needs ×SCALE.
        S = 1.0 if usd_path else SCALE

        # base_z=0.23m: shoulder joint sits at Z=0.255m.
        # At shoulder=-90°: bucket_joint Z = 0.255 - 0.250 = 0.005m (just above ground, in sand).
        # At shoulder=-60°: bucket_joint Z = 0.255 - 0.217 = 0.038m (mid-sand).
        # Old value 0.40m was too high — bucket_joint Z at -90° was 0.175m, above the 10cm sand bed.
        base_xform = wp.transform(
            wp.vec3(0.0, 0.0, 0.23 * S), wp.quat_identity()
        )

        if usd_path:
            print(f"Loading USD: {usd_path}  (real-scale, S=1)")
            builder.add_usd(
                usd_path, xform=base_xform,
                floating=False, collapse_fixed_joints=False,
                override_root_xform=True,
                # The USD has two ArticulationRootAPI prims; ignore the fixed-
                # joint anchor so only one articulation is imported.
                ignore_paths=["/World/Idea_3/root_joint"],
                # Keep raw triangle meshes — CONVEX_MESH (type 10) is not
                # supported by the MPM SDF collider builder.
                skip_mesh_approximation=True,
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
                builder.joint_limit_lower[dof] = -math.pi / 2        # −90°
                builder.joint_limit_upper[dof] =  math.pi / 2        # +90°
                self._shoulder_dof = dof
                print(f"  shoulder_joint → DOF {dof}")
            elif "bucket_joint" in label:
                builder.joint_limit_lower[dof] = -math.radians(135)  # −135° dump
                builder.joint_limit_upper[dof] =  math.radians(135)  # +135° scoop
                self._bucket_dof = dof
                print(f"  bucket_joint   → DOF {dof}")

        if self._shoulder_dof is None or self._bucket_dof is None:
            print("WARNING: joints not found. Labels:", builder.joint_label)

        # ── Configure MPM particle collision per shape ────────────────────────
        # Only the bucket collision mesh participates in particle contact.
        # - Base/shoulder: too large, causes explosive contact forces.
        # - Bucket visual shape: identical to collision mesh — having both
        #   enabled doubles the SDF force and can eject particles from bucket.
        for body in range(builder.body_count):
            label = (builder.body_label[body] or "").lower()
            is_bucket_body = "bucket" in label
            for shape in builder.body_shapes[body]:
                shape_label = (builder.shape_label[shape] if shape < len(builder.shape_label) else "") or ""
                has_collision_tag = "collision" in shape_label.lower()
                has_visual_tag    = "visual"    in shape_label.lower()
                # USD: each body has both visual + collision shapes — only enable
                # the collision shape on the bucket.
                # URDF: shapes have no visual/collision tags — enable all bucket shapes.
                is_active = is_bucket_body and (has_collision_tag or (not has_visual_tag and not has_collision_tag))
                if not is_active:
                    builder.shape_flags[shape] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES)
                    print(f"  body[{body}] shape[{shape}]: particle collision disabled")
                else:
                    print(f"  body[{body}] shape[{shape}]: particle collision ENABLED")

        # ── Initial pose: both joints at 0° — trajectory starts from here ────
        if self._shoulder_dof is not None:
            builder.joint_q[self._shoulder_dof] = 0.0
        if self._bucket_dof is not None:
            builder.joint_q[self._bucket_dof] = 0.0

        # ── Ground plane + side walls ─────────────────────────────────────────
        builder.add_ground_plane()
        # Invisible containment walls at y = ±0.5*S (infinite planes, no visual)
        # plane=(a,b,c,d) → normal (a,b,c), position = -(d/|n|)*n̂
        # Wall at y = +0.5*S, normal pointing -Y
        builder.add_shape_plane(plane=(0.0, -1.0, 0.0,  0.5 * S), width=0.0, length=0.0)
        # Wall at y = -0.5*S, normal pointing +Y
        builder.add_shape_plane(plane=(0.0,  1.0, 0.0,  0.5 * S), width=0.0, length=0.0)

        # ── MPM particle attributes — must be registered BEFORE add_particle_grid
        SolverImplicitMPM.register_custom_attributes(builder)

        # ── Sand particles (into the SAME builder) ────────────────────────────
        # USD (S=1): vox=0.008m → ~363k particles, dia=2.7mm, ~45 FPS on RTX 4080 Laptop
        # URDF (S=10): same ratio → 0.10m voxels
        voxel_size = 0.008 * S
        self._emit_particles(builder, voxel_size, S)

        # ── Finalize single model ──────────────────────────────────────────────
        self.model = builder.finalize()
        self.model.mpm.hardening.fill_(0.0)  # no strain hardening
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
        # Smoothed joint values — exponential filter chases slider target
        self._smooth_shoulder = self._slider_shoulder
        self._smooth_bucket   = self._slider_bucket

        # ── MPM solver ────────────────────────────────────────────────────────
        mpm_options = SolverImplicitMPM.Config()
        mpm_options.voxel_size            = voxel_size
        mpm_options.tolerance             = 1.0e-5
        mpm_options.grid_type             = "sparse"  # dynamic — follows particles, no fixed bounding box
        mpm_options.grid_padding          = 0   # granular example default; 10 inflates FEM matrix → OOM
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
            collider_thicknesses=[0.0, 1.0 * voxel_size],  # static (ground+walls): none; bucket: 1× voxel
        )

        # ── Viewer ────────────────────────────────────────────────────────────
        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self._camera_framed = False

    # ── Particle emission ─────────────────────────────────────────────────────

    def _emit_particles(self, builder: newton.ModelBuilder, voxel_size: float, S: float):
        """
        Sand bed in front of and below the initial bucket reach.

        Bed covers X = 0.10*S → 0.42*S, Y = ±0.10*S, Z = 0 → 0.10*S.
        S=SCALE for URDF (×10), S=1 for USD (real metres).
        Invisible side walls at Y = ±0.50*S contain all particles.

        IMPORTANT: bed_lo.z is raised by one particle radius after cell_size is
        computed.  This ensures the lowest jittered particles (jitter offset =
        −radius) land at z ≥ 0 and never penetrate the ground plane.
        """
        particles_per_cell = 3.0
        density = 1000.0  # kg/m³  granular example default

        bed_lo = np.array([0.10 * S, -0.1 * S, 0.0])
        bed_hi = np.array([0.42 * S,  0.1 * S, 0.10 * S])   # 10 cm real height — covers bucket reach from -60° to -90°

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
                "mpm:tensile_yield_ratio":   0.0,    # cohesionless sand
            },
        )

    # ── Scripted trajectory ───────────────────────────────────────────────────

    def _scripted_targets(self) -> tuple[float, float]:
        """
        3-phase dig cycle → (shoulder_rad, bucket_rad).

        Phase 1 (0–2 s)  — bucket pre-curls to −135° (scoop), shoulder holds at 0°.
        Phase 2 (2–5 s)  — shoulder descends LINEARLY to −90° into sand; bucket holds.
        Phase 3 (5–8 s)  — both joints cosine-return to 0° simultaneously (same fraction).

        Joint axis (0, −1, 0):
          negative angle → tip toward −Z (arm down / bucket curls to scoop)
          positive angle → tip toward +Z (arm up / bucket opens)

        At base_z=0.23m:  shoulder_joint world Z = 0.255m
          shoulder=−90° → bucket_joint Z = 0.255 − 0.250 = 0.005m  [at sand surface]
        """
        BUCKET_SCOOP = math.radians(135)    # +135°: scoop / dig position
        BUCKET_DUMP  = -math.radians(135)   # −135°: dump / drop position
        SHOULDER_DIG = -math.pi / 2          # −90°: arm pointing straight down

        CYCLE = 10.0
        t = self.sim_time % CYCLE

        def cos_lerp(a: float, b: float, tt: float) -> float:
            tt = max(0.0, min(1.0, tt))
            return a + (b - a) * (0.5 - 0.5 * math.cos(math.pi * tt))

        def lin_lerp(a: float, b: float, tt: float) -> float:
            tt = max(0.0, min(1.0, tt))
            return a + (b - a) * tt

        if t < 2.0:        # Phase 1: bucket curls to +135° (scoop); shoulder at 0°
            return 0.0, cos_lerp(0.0, BUCKET_SCOOP, t / 2.0)
        elif t < 5.0:      # Phase 2: shoulder descends linearly to −90°; bucket holds
            return lin_lerp(0.0, SHOULDER_DIG, (t - 2.0) / 3.0), BUCKET_SCOOP
        elif t < 8.0:      # Phase 3: both return to 0° simultaneously (same fraction)
            f = (t - 5.0) / 3.0
            return cos_lerp(SHOULDER_DIG, 0.0, f), cos_lerp(BUCKET_SCOOP, 0.0, f)
        elif t < 9.0:      # Phase 4: bucket dumps to −135° to drop particles
            return 0.0, cos_lerp(0.0, BUCKET_DUMP, (t - 8.0) / 1.0)
        else:              # Phase 5: bucket returns to 0° — ready for next cycle
            return 0.0, cos_lerp(BUCKET_DUMP, 0.0, (t - 9.0) / 1.0)

    # ── Simulation step ───────────────────────────────────────────────────────

    def step(self):
        # Compute target joint angles for this frame
        if self._manual:
            # Exponential smoothing: each frame advance 8% toward slider target.
            # Gives natural deceleration as arm approaches target (feels smooth).
            # Hard cap keeps per-frame delta safe for MPM finite_difference velocity.
            alpha = 0.08
            max_delta = 2.0 * self.frame_dt
            for attr, tgt in (("_smooth_shoulder", self._slider_shoulder),
                               ("_smooth_bucket",   self._slider_bucket)):
                prev = getattr(self, attr)
                raw  = prev + alpha * (tgt - prev)
                setattr(self, attr, prev + float(np.clip(raw - prev, -max_delta, max_delta)))
            shoulder = self._smooth_shoulder
            bucket   = self._smooth_bucket
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
        "--usd", type=str,
        default=str(USD_PATH) if USD_PATH.exists() else None,
        metavar="PATH",
        help="USD file (default: arm.usd in same folder if present)",
    )
    viewer, args = newton.examples.init(parser)

    if wp.get_device().is_cpu:
        print("WARNING: GPU recommended for MPM.")

    example = ExcavatorExample(viewer, usd_path=args.usd)
    newton.examples.run(example, args)
