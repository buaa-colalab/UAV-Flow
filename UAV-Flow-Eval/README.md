# UAV-Flow-Eval

UAV-Flow-Eval is a simulation environment for UAV evaluation tasks, built on top of [UnrealZoo Gym](https://github.com/UnrealZoo/unrealzoo-gym).


### 1. Set Up the Environment
Create a conda environment, then install this repository in editable mode:
```bash
conda create -n unrealcv python=3.11
conda activate unrealcv
pip install -e .
```

### 2. Download the Simulation Environment
We tested on Windows using the packaged UnrealZoo environment:
[Collection_WinNoEditor_0424_25.zip](https://modelscope.cn/datasets/UnrealZoo/UnrealZoo-UE4/file/view/master/Collection_WinNoEditor_0424_25.zip?id=77779&status=2). Download and extract the archive to a local directory.

### 3. Configure the Environment Path
We use the DowntownWest as the test campus environment.
You need to update the configuration file:

/gym_unrealcv/envs/setting/Track/DowntownWest.json

Change the env_bin_win field to the actual path of your extracted simulation environment.

### 4. Run Evaluation

Set up your model on the server side and expose it through a specific port.
Then run:

```bash
python batch_run_act_all.py
```

You can modify arguments either in the batch_run_act_all.py or directly via the command line.
For example, to change the inference server port to 5006:

```bash
python batch_run_act_all.py --server_port 5006
```
After inference finishes, compute the NDTW metric by running:
```bash
python metric.py
```