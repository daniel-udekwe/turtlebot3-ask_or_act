# Ask or Act: Calibrated Confidence for Risk-Aware Navigation Under Uncertain Human Intent

Code for the paper *"Ask or Act: A Calibrated-Confidence Framework for Risk-Aware
Navigation Under Uncertain Human Intent"* (D. Udekwe and Q. Guo).

A robot driving along a path meets a pedestrian whose intent (crossing or not
crossing) is unknown. It predicts the probability that the pedestrian will cross,
calibrates that probability, and at a decision deadline yields, proceeds, or asks
the pedestrian when the calibrated probability is uncertain. The paper shows that
calibration measured over all samples can hide miscalibration at the moments that
drive decisions, and proposes **decision-aware calibration**, fitted only on the
decision window.

The final results in the paper were produced from the git tag **`v1-final-eval`**.

## Requirements

- Ubuntu 22.04, ROS 2 Humble, Gazebo 11 (Classic), TurtleBot3 packages
- Python 3.10 with numpy 1.21.5, pandas 1.3.5, scikit-learn 0.23.2, scipy 1.8.0, matplotlib
  (the Ubuntu 22.04 `apt` versions)

The offline experiments (predictor, calibration, policy evaluation, statistics) need
only Python. ROS 2 and Gazebo are needed only for the Gazebo episodes and validation.

The saved model (`predictor/intent_model.pkl`) must be loaded with scikit-learn 0.23.2.

## Repository structure

| File | Purpose |
|---|---|
| `ped_traj.py` | Deterministic pedestrian trajectories (shared by all components) |
| `generate_scenarios.py` | Scenario generator: 8 families, 6 operating conditions, ground-truth intent |
| `build_dataset.py` | Predictor training, calibration and test data (no Gazebo needed) |
| `intent_features.py` | Feature extraction (shared by training and the robot) |
| `calibration.py` | Temperature scaling and isotonic calibration |
| `train_predictor.py` | Trains the MLP and logistic regression; fits all calibrators, including decision-aware |
| `policy.py` | The decision policies (Always act, Always cautious, Always ask, Oracle, threshold) |
| `simulate_policies.py` | Offline closed-loop simulator |
| `sweep_response.py` | Response-reliability sweep (answer probability, delay, accuracy) |
| `final_eval.py` | Final evaluation: main comparison, ask-band, deadline and predictor-seed experiments |
| `final_stats.py` | Pre-specified hypothesis tests and result tables |
| `make_figures.py` | All paper figures and LaTeX tables |
| `scripted_pedestrians.py` | ROS 2 node that moves pedestrians in Gazebo and answers questions |
| `line_controller.py` | ROS 2 robot controller implementing all policies |
| `run_episodes.py` | Runs Gazebo episodes for any policy |
| `gazebo_validation.py`, `compare_validation.py` | Gazebo validation against the offline simulator |
| `benchmark_onboard.py` | Onboard computation benchmark (run on the robot's Raspberry Pi) |
| `scene_2d.py`, `stage_scene.py`, `rviz_scene.py` | Scene illustrations (2D, Gazebo, RViz) |
| `ANALYSIS_PLAN.md` | The pre-specified analysis plan, with deviations log |

## Reproducing the results

Offline experiments (Python only):

```bash
python3 build_dataset.py                 # predictor data
python3 train_predictor.py               # train and calibrate the predictors
python3 final_eval.py                    # final evaluation on seeds 7000+ (about 1-2 h on 18 cores)
python3 sweep_response.py --full --seed-start 7000 --seeds 100 --out sim/final/response
python3 final_stats.py                   # hypothesis tests -> sim/final/stats/RESULTS.md
python3 make_figures.py                  # figures and tables -> paper/
```

Gazebo validation (needs ROS 2, Gazebo and the TurtleBot3 packages):

```bash
python3 gazebo_validation.py             # 320 Gazebo episodes, resumable
python3 compare_validation.py            # agreement with the offline simulator
```

The Gazebo experiments use the cafe world from
[HuNavSim](https://github.com/robotics-upo/hunav_sim), with a `gazebo_ros_state`
plugin added (`worlds/cafe_scripted.world`, created by `run_episodes.py`).
HuNavSim's own pedestrian agents are not used; pedestrians are scripted.

Onboard benchmark (on the robot):

```bash
python3 benchmark_onboard.py
```

## Citation

If you use this code, please cite the paper (citation details will be added on publication).

## License

MIT License; see [LICENSE](LICENSE).
