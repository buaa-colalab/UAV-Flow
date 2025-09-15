## UAV-Flow Evaluation (on UnrealCV/Gym-UnrealCV)

This repository evaluates UAV-Flow in the UnrealCV-based simulation. Below are the minimal steps to configure the environment, run batch evaluation, and compute metrics.

### 1) Setup (same as the simulation environment)
1. Dependencies (recommend using conda/venv)
   - Python ≥ 3.8
   - UnrealCV, Gym, NumPy, Matplotlib, OpenCV, SciPy
2. Install gym-unrealcv (simulation interface)
   ```bash
   git clone https://github.com/UnrealCV/gym-unrealcv.git
   cd gym-unrealcv
   pip install -e .
   # This installs OpenAI Gym, UnrealCV, numpy, matplotlib, etc.
   ```
3. Prepare UE binary (Unreal Engine environments)
   - Download the UE binary packages per the simulation docs (UE4/UE5 example scenes or collections)
   - Unzip and place them under a folder you choose (see next step for path)
4. Set UnrealEnv path (so gym-unrealcv can locate binaries)
   - Default lookup path is under user home: `.unrealcv/UnrealEnv`
   - You can override it by exporting `UnrealEnv` to your binary root path
   - Examples:
     - Windows (PowerShell):
       ```powershell
       setx UnrealEnv "C:\\Users\\<username>\\.unrealcv\\UnrealEnv"
       ```
     - Linux (bash):
       ```bash
       export UnrealEnv=/home/<username>/.unrealcv/UnrealEnv
       ```
     - macOS (zsh/bash):
       ```bash
       export UnrealEnv=/Users/<username>/.unrealcv/UnrealEnv
       ```
5. Permissions (Linux/macOS)
   ```bash
   chmod +x -R "$UnrealEnv"
   ```
6. Quick verification (optional)
   - Use an example agent from the simulation repo to ensure the environment launches, e.g.:
   ```bash
   python ./example/random_agent_multi.py -e UnrealTrack-track_train-ContinuousColor-v0
   ```

### 2) Run evaluation (batch)
Start your inference server locally first (must expose POST /predict).

Then run batch evaluation and trajectory visualization:
```bash
python batch_run_act_all.py \
  --env_id UnrealTrack-DowntownWest-ContinuousColor-v0 \
  --time_dilation 10 \
  --seed 0 \
  --json_folder ./test_jsons \
  --images_dir ./results/UnrealTrack-DowntownWest-ContinuousColor-v0/openvla \
  --server_port 5007 \
  --max_steps 100 \
  --instruction_type instruction \
  --log_level INFO \
  --tick_x 100 --tick_y 100 --tick_z 100 \
  --min_span_x 400 --min_span_y 400 --min_span_z 400
```
Outputs per task (in images_dir):
- `<task>.json` trajectory
- `<task>_2d.png` 2D trajectory plot
- `<task>_3d.png` 3D trajectory plot

### 3) Compute metrics
Make sure the following paths exist (relative paths by default):
- `./results/UnrealTrack-DowntownWest-ContinuousColor-v0/<model>/`
- `./test_jsons`
- `./classified_instr.json`

Run metrics:
```bash
python metric.py
```
Logs will be saved to `./metric.txt`.
