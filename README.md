# Excavator MPM Simulation

A kinematic excavator arm simulation using NVIDIA Newton's Material Point Method (MPM) solver
for granular soil dynamics, with full ROS 2 Humble integration for joint control.

The simulation models ~363k sand particles (2.7mm grain size, `voxel_size=0.008m`) and runs
at ~45 FPS on an RTX 4080 Laptop. Joints are driven by a closed-loop trajectory sender that
syncs to the simulator's internal clock — the controller never drifts even if the sim is paused
or slowed down.

---

## What it does

The excavator performs a repeating 10-second dig cycle:

| Phase | Duration | Motion |
|-------|----------|--------|
| 1 — bucket scoop  | 2 s | Bucket rotates 0° → +135° (cosine easing) |
| 2 — shoulder dig  | 3 s | Arm drives 0° → -90° linearly while bucket holds +135° |
| 3 — return home   | 3 s | Both joints return to 0° together (cosine, synchronized) |
| 4 — bucket dump   | 1 s | Bucket tips 0° → -135° to drop collected material |
| 5 — reset         | 1 s | Bucket returns 0° to start next cycle |

---

## Hardware requirements

- NVIDIA GPU with CUDA (tested on RTX 4080 Laptop, 12 GB)
- Ubuntu 22.04 (ROS 2 Humble requires this)
- ~8 GB RAM minimum

---

## Software prerequisites

You need the following installed before starting:

- **ROS 2 Humble** — https://docs.ros.org/en/humble/Installation.html
- **pyenv** — https://github.com/pyenv/pyenv#installation
- **CUDA toolkit** compatible with your driver (CUDA 12.x recommended)
- **git**, **colcon**, standard build tools

---

## Step 1 — Create a Python 3.10 virtual environment

ROS 2 Humble's `rclpy` C extensions are compiled against Python 3.10. Newton/Warp also need
to run in the same interpreter. Create a dedicated pyenv environment:

```bash
# Install Python 3.10.16 (if not already installed)
pyenv install 3.10.16

# Create a named virtualenv
pyenv virtualenv 3.10.16 ecorobotic-newton-ros2

# Activate it
pyenv activate ecorobotic-newton-ros2
```

---

## Step 2 — Install Newton and Warp

Newton is NVIDIA's physics engine built on top of Warp. Clone and install:

```bash
# Install Warp first (GPU kernel compiler Newton depends on)
pip install warp-lang

# Clone Newton (feature/newton branch of IsaacLab)
git clone -b feature/newton https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab

# Install Newton itself
cd ~/IsaacLab
pip install -e source/extensions/omni.isaac.newton
```

Alternatively, if you have Newton installed as a standalone package:

```bash
pip install newton
```

Verify the install:

```bash
python -c "import newton; print('Newton OK')"
python -c "import warp; print('Warp OK')"
```

---

## Step 3 — Install remaining Python dependencies

```bash
pip install pyglet usd-core "imgui-bundle==1.92.5" numpy
```

> **Note on imgui-bundle version**: Version 1.92.600+ has a breaking API change in
> `color_edit3`. Pin to `==1.92.5` to avoid a runtime crash in Newton's viewer.

---

## Step 4 — Install ROS 2 Python packages into the environment

The pyenv environment needs to find `rclpy` and the message packages. Add them to your
PYTHONPATH — ROS 2 Humble ships Python packages in two separate locations:

```bash
export PYTHONPATH="$HOME/.pyenv/versions/ecorobotic-newton-ros2/lib/python3.10/site-packages:\
/opt/ros/humble/local/lib/python3.10/dist-packages:\
/opt/ros/humble/lib/python3.10/site-packages:$PYTHONPATH"
```

Add this to your `~/.bashrc` (or `~/.zshrc`) so it persists across terminals:

```bash
echo 'export PYTHONPATH="$HOME/.pyenv/versions/ecorobotic-newton-ros2/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages:$PYTHONPATH"' >> ~/.bashrc
source ~/.bashrc
```

Verify:

```bash
python -c "import rclpy; print('rclpy OK')"
```

---

## Step 5 — Get the USD arm model

The excavator arm is described by `arm.usd`, which contains relative references to mesh
files. It **must stay in its original directory** — do not copy it elsewhere or the meshes
will fail to load.

If you cloned this repo into `~/ecorobotic/excavator/`, the default USD path will work
automatically. If you put it somewhere else, pass `--usd /path/to/arm.usd` when launching.

---

## Step 6 — Build the ROS 2 package

```bash
source /opt/ros/humble/setup.bash

cd ~/ros2_ws   # or wherever your colcon workspace is
# Copy the package in if you haven't already:
# cp -r ~/ecorobotic/excavator/excavator_mpm src/

colcon build --packages-select excavator_mpm
source install/setup.bash
```

---

## Running the simulation

### Option A — Full launch (sim + trajectory controller together)

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 launch excavator_mpm excavator_full.launch.py
```

Run a fixed number of dig cycles then stop:

```bash
ros2 launch excavator_mpm excavator_full.launch.py cycles:=3
```

Run headless (no OpenGL window):

```bash
ros2 launch excavator_mpm excavator_full.launch.py viewer:=none cycles:=1
```

### Option B — Simulation only

```bash
ros2 launch excavator_mpm excavator.launch.py
```

### Option C — Send commands manually

Start the sim, then in a second terminal send joint commands directly:

```bash
# Terminal 1
ros2 launch excavator_mpm excavator.launch.py

# Terminal 2 — shoulder to -45°, bucket to +90°
ros2 topic pub /excavator/joint_commands sensor_msgs/JointState \
  '{name: ["shoulder_joint","bucket_joint"], position: [-0.785, 1.571]}'
```

### Option D — Standalone (no ROS 2)

The scripted standalone version requires only Newton/Warp:

```bash
pyenv activate ecorobotic-newton-ros2
python simulate_excavator.py
```

---

## ROS 2 topics

| Topic | Type | Direction | Description |
|-------|------|-----------|-------------|
| `/excavator/joint_commands` | `sensor_msgs/JointState` | Subscribe | Commanded joint positions (rad) |
| `/excavator/joint_states`   | `sensor_msgs/JointState` | Publish   | Actual joint positions from sim (60 Hz) |
| `/excavator/sim_time`       | `std_msgs/Float64`       | Publish   | Simulator internal time (60 Hz) |
| `/excavator/particles`      | `sensor_msgs/PointCloud2`| Publish   | Sand particle positions (10 Hz) |

---

## Terminal output

While running you will see joint status at ~5 Hz:

```
[1 — bucket scoop  67.1%]  shoulder  cmd=  +0.00°  act=  +0.00°    bucket    cmd=+116.82°  act=+112.44°
[2 — shoulder dig  43.2%]  shoulder  cmd= -39.00°  act= -37.81°    bucket    cmd=+135.00°  act=+134.12°
```

Phase transitions are logged explicitly:

```
[INFO] Phase 1 — bucket scoop  done  →  Phase 2 — shoulder dig  started
```

---

## Keyboard controls (sim window)

| Key | Action |
|-----|--------|
| Space | Pause / resume |
| R | Reset simulation |
| Q / Esc | Quit |

---

## Repository structure

```
excavator/
├── simulate_excavator.py          # Standalone scripted sim (no ROS 2)
├── simulate_excavator_ros2.py     # ROS 2 sim node source
├── arm.usd                        # USD robot model (keep here — relative mesh refs)
└── excavator_mpm/                 # ROS 2 colcon package
    ├── package.xml
    ├── setup.py
    ├── excavator_mpm/
    │   ├── simulate_excavator_ros2.py   # Sim node (publishes topics, reads commands)
    │   └── trajectory_sender.py         # Closed-loop 5-phase dig controller
    └── launch/
        ├── excavator.launch.py          # Sim only
        └── excavator_full.launch.py     # Sim + trajectory sender
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'warp'`**
You are running with the wrong Python. Make sure the launch file points to your pyenv
Python and that PYTHONPATH includes the pyenv site-packages. Check:
```bash
which python  # should be the pyenv python
python -c "import warp"
```

**`ModuleNotFoundError: No module named 'rclpy'`**
PYTHONPATH is missing the ROS 2 dist-packages path. Re-run Step 4 above.

**`imgui ... color_edit3 ... unexpected keyword argument`**
imgui-bundle version is too new. Downgrade:
```bash
pip install "imgui-bundle==1.92.5"
```

**USD composition errors / missing meshes**
The `arm.usd` file was moved or copied. It must load from its original directory because it
uses relative paths to mesh `.obj` files. Pass the correct path:
```bash
ros2 launch excavator_mpm excavator_full.launch.py usd:=/absolute/path/to/arm.usd
```

**Sim starts but joints do not move**
The trajectory sender waits for `/excavator/sim_time` before sending any commands. If the
sim takes more than ~30 seconds to start (CUDA JIT compilation on first run), the sender
may time out. Re-run after the sim has fully initialised — subsequent runs are fast because
shaders are cached.

---

## License

MIT
