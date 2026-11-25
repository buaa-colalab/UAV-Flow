This document provides instructions for reproducing our **experimental results** with SimplerEnv.  

# 🚀 Eval SimplerEnv
The evaluation pipeline is adapted from https://github.com/InternRobotics/InternVLA-M1/examples/SimplerEnv

Steps:
1) Download the checkpoint:`results/Checkpoints/1_need/0723_v6_vla_dino_32/checkpoints/steps_10000_pytorch_model.pt`

We also provide a parallel evaluation script:

```bash
check_pt=0723_v6_vla_dino_32/checkpoints/steps_10000_pytorch_model.pt
bash examples/SimplerEnv/eval_scripts/star_bridge.sh ${check_pt}
```

Before running star_bridge.sh, set the following three paths:
- star_vla_python: Python interpreter for the StarVLA environment.
- sim_python: Python interpreter for the SimplerEnv environment.
- SimplerEnv_PATH: Local path to the SimplerEnv project.
Alternatively, edit these variables directly at the top of `star_bridge.sh`.



## 📊 Experimental Results

### 1. WidowX Robot

| Task                              | Success Rate (%) |
| --------------------------------- | ---------------- |
| Put Spoon on Towel                | 87.5             |
| Put Carrot on Plate               | 67.9             |
| Stack Green Block on Yellow Block | 31.3             |
| Put Eggplant in Yellow Basket     | 100.0            |
| **Average**                       | **71.7**         |

---

### 2. Google Robot (Visual Matching)

| Task                                     | Success Rate (%) |
| ---------------------------------------- | ---------------- |
| Pick Coke Can                            | 95.3             |
| Move Near                                | 90.0             |
| Open/Close Drawer                        | 75.5             |
| Open Top Drawer and Place Apple          | 62.0             |
| **Average**                              | **80.7**         |

---

### 3. Google Robot (Variant Aggregation)

| Task                                     | Success Rate (%) |
| ---------------------------------------- | ---------------- |
| Pick Coke Can                            | 86.1             |
| Move Near                                | 82.0             |
| Open/Close Drawer                        | 72.0             |
| Open Top Drawer and Place Apple          | 64.0             |
| **Average**                              | **76.0**         |

---



