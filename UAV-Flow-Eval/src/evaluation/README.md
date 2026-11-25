# Single-model evaluation

Run one model over the reference test set in one fresh pass. Follow the
[quick start](../../README.md#3-closed-loop-evaluation), or run
`uav-flow evaluate --help` for all options. `python -m uav_flow evaluate` uses
the same environment's interpreter.

Choose an existing model API with `--server-url` or let the runner manage a compatible model server with `--checkpoint` and an explicit `--model-gpu`. A full simulation also requires an explicit `--graphics-adapter` and a fresh output directory. Vulkan adapter numbers must be verified against physical GPU UUIDs.

## Controller and closed-loop worker

`uav-flow evaluate` invokes `run_single_model.py`, which manages one complete
evaluation: validate the task manifest and fresh output directory, connect to or
launch the model API, prepare isolated UE files under the output root, supervise the worker,
collect trajectories, score results and clean up owned resources.

It launches `python -m evaluation.closed_loop_runner` from the current environment's installed
package as a child process, with a working directory under the output root. The worker
iterates over the tasks, resets model state, sends observations for inference,
executes the returned waypoints and saves trajectories and frames. The controller
passes the worker's failure back to the public command.

This process separation lets the simulator run under its configured ordinary
account using the same Python interpreter while the controller enforces the timeout and
owns resource cleanup. The worker uses the same model API throughout the single pass.

`--dry-run` prints the task manifest without starting a service. `--smoke` checks only model reset/predict without starting UE. `--score-only` re-scores an existing run using the expected task filenames. Reports are excluded from the denominator; missing tasks remain failures/errors.

The model reset must succeed before inference for each task. Execution failure produces a nonzero controller exit code. Cleanup applies only to resources created by this runner.

The evaluation protocol uses up to 400 steps for Surround with early success at mean radial error <30 cm and reference-direction net angle >=342 degrees; other categories use 100 steps. Existing non-Surround tail trimming requires an explicit protocol review before changes.

The simulator and checkers are shared from `src/`; evaluation selects its own
instance-level lifecycle profile and orbit policy. Both workflows use the invoking
interpreter's installed package without copying Python code or task data.
External UE assets and model weights are supplied separately.

Run the isolated regression suite from the repository root:

```bash
pytest tests/evaluation -m "not simulator"
```
