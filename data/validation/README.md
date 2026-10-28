# Stage 1 validation datasets

Raw per-trial data behind the two Stage 1 figures in `figures/`. Both experiments use
the stock robosuite **Lift** task (one cube, no clutter, no observation or execution
noise) with a Franka Panda under OSC_POSE control, and the scripted top grasp in
`src/robot_uncertainty/sim/primitives.py`. They validate that primitive; they are not
the project's main experiments.

| File | Rows | Produced by | Figure |
|---|---|---|---|
| `yaw_switch.csv` | 165 | `scripts/diagnose_yaw_switch.py` | `figures/yaw_switch_45deg_selection_vs_lift.png` |
| `grasp_alignment.csv` | 100 | `scripts/compare_grasp_alignment.py` | `figures/finger_gap_fixed_vs_aligned.png` |

## How a grasp is judged

Phases: open gripper; move 10 cm above the cube while turning the fingers onto a face
pair; descend to the cube centre; close; lift 10 cm; hold. `success` is judged from the
true simulator state after the hold: the cube has risen at least 5 cm, is within 2 cm
of the grip site, and the fingers have not closed on nothing.

Common columns: `seed`; `success` (bool); `face_axis`, the cube body axis (0 or 1) the
fingers closed along; `wrist_deg`, the commanded wrist yaw turn; `gap_mm`, the finger
gap after the hold; `lift_cm`, cube height gained; `steps`, 20 Hz control steps;
`wall_s`, wall-clock seconds (machine-dependent, excluded from reproducibility checks).
Angles are in degrees, yaws are headings in the world xy-plane, and a cube's yaw is the
heading of a side-face normal (body x axis).

## Experiment A: 45-degree orientation switching (`yaw_switch.csv`)

**Question.** The grasp closes on whichever of the cube's four equivalent face
orientations needs the least wrist rotation, so the choice flips where the cube's yaw
is 45 degrees from the gripper's. Does that discontinuity in *selection* produce a
discontinuity in *physical grasp performance*?

**Design.** Seeds 0-2 (`make_lift_env(seed, hard_reset=False)`). After reset and a
0.25 s settle, the resting cube is rotated in place to a true yaw of `delta_deg` in
{40, 44, 45, 46, 50} from the gripper's starting yaw. The grasp is then planned from an
observed yaw of true + `eps_deg`, with eps in {0, ±0.01, ±0.1, ±0.5, ±1, ±2}. 3 seeds x
5 deltas x 11 eps = 165 trials, one per condition (the simulation is deterministic).

**Held constant.** Within each (seed, delta), the full MuJoCo state at the start of the
grasp (qpos, qvel, actuator state, solver warm-start) is fingerprinted and identical for
every eps, so only the observed yaw differs. The cube's position, size, the robot's
start configuration and all controller settings are fixed by the seed.

**Recorded.** `delta_deg`, `eps_deg`; `true_rel_deg` and `observed_rel_deg` (true and
observed cube yaw relative to the gripper's start yaw); `face_width_mm` (width of the
gripped face pair); `yaw_err_close_deg` and `yaw_err_final_deg` (finger-to-face angle as
the grip closes and after the hold); `timeouts` (phases that ran out of steps); and the
common columns.

**Result.** The selection switches exactly where the observed relative yaw crosses 45
degrees (wrist +45 to -45, face pair 0 to 1). All 165 grasps succeeded with no
timeouts; lift 9.01-9.10 cm and 80-83 steps throughout. The alignment error at close
equals |eps| (up to 2.0 degrees), and closing removes about 70% of it (at most 0.72
degrees after the hold). Across the switch, lift changes by at most 0.02 cm and steps by
at most 1, less than the variation between seeds. The finger gap follows the gripped
face pair's width (0.15-0.22 mm under it), which differs between pairs by up to 1.75 mm
because Lift's cube is not exactly square. Conclusion: a selection discontinuity, not a
physical one.

**Limitations.** Three seeds; Lift's single small cube rather than the proposal's
4 x 4 x 8 cm blocks; no clutter, so both face pairs are always reachable (a neighbour
beside one pair could make the choice matter); no execution noise; the exact tie at
eps = 0 is resolved by floating-point rounding (it went to +45 on all seeds).

## Experiment B: fixed vs yaw-aligned controller (`grasp_alignment.csv`)

**Purpose.** Measure what aligning the fingers with the cube's observed yaw changes,
against a controller that always closes its fingers along world x.

**Baseline.** `controller = fixed` is a **reconstruction** of the original controller,
not an exact replay of it. It uses the original fixed target (grip-site orientation
`TOP_DOWN`, yaw 0) but runs on the current controller code, which clips rotation
commands by their magnitude; the original clipped each component separately. A one-off
check against the original implementation (not kept in the repository) on the same 50
seeds gave per-seed finger gaps differing by a median of 0.02 mm (at most 0.55 mm) and
step counts by -1 to +5. Treat the baseline as equivalent in target, close but not
identical in trajectory.

**Why identical seeds.** Each seed fixes the cube's size, position and yaw and the
robot's initial joint noise. Each controller grasps from its own fresh environment
built with the same seed (`make_lift_env(seed)`, robosuite's default hard reset), so
each pair sees the same scene and differences are attributable to the controller.

**Design.** Seeds 0-49, two controllers each: 100 trials.

**Recorded.** `controller` (`fixed` or `aligned`); `cube_width_x_mm`,
`cube_width_y_mm`, `cube_yaw_deg` (true yaw at the start); `target_yaw_deg`; and the
common columns. The dimensions and yaw allow the span of a rigid, unrotated cube across
the fixed fingers to be computed: width_x |cos yaw| + width_y |sin yaw|.

**Findings.** Both controllers: 50/50 verified grasps. Finger gap: fixed 40.3-61.2 mm
(median 47.1), aligned 39.9-43.8 mm (median 41.8), against recorded face widths of
40.0-43.9 mm, so every aligned grasp closes on two faces. Steps: fixed 91-98 (mean 94.0),
aligned 75-82 (mean 78.2), because the aligned wrist turns at most 45 degrees instead of
about 90. Lift is similar (8.65-9.18 vs 8.63-9.11 cm). The fixed controller shows three
regimes: up to about 6 degrees of misalignment the closing fingers rotate the cube
square (gap = face width); from about 8 to 32 degrees they rotate it part way (gap 1.5-7.4
mm below the rigid-cube span); from 33 degrees the cube stays gripped across its corners
(gap within 1.1 mm of the span).

**Limitations.** Reconstructed baseline (above); Lift's cube and no clutter or noise;
the misalignment regimes are specific to this gripper, cube and friction.

## Reproducing

Environment used: Python 3.12.3, MuJoCo 3.9.0, robosuite 1.5.2, numpy 1.26.4 (held
below 2 by robosuite's `mink` dependency), Matplotlib 3.11.2; Linux (WSL2), x86_64,
AMD Ryzen 5 5500.

The exact versions of every installed package are pinned in
`requirements-lock.txt` (a `pip freeze` of that environment, without the project itself).

```bash
# Exact tested environment:
pip install -r data/validation/requirements-lock.txt
pip install -e . --no-deps
# Or the project's own (unpinned) dependency ranges:
pip install -e ".[sim,dev,analysis]"

# Figures from the committed CSVs (no MuJoCo needed):
python scripts/plot_grasp_figures.py

# Rerun both experiments (about 5 minutes) into results/regenerated/ and compare
# with these files; the committed data is never replaced:
python scripts/plot_grasp_figures.py --regenerate

# Or run one experiment; output goes to results/ unless --out is given, and an
# existing file is only replaced with --overwrite:
python scripts/diagnose_yaw_switch.py --seeds 3 --out results/yaw_switch.csv
python scripts/compare_grasp_alignment.py --seeds 50 --out results/grasp_alignment.csv
```

Every column except `wall_s` is reproduced exactly on this setup. MuJoCo is
deterministic for the same build on the same CPU type; bitwise results on other
hardware or MuJoCo versions may differ slightly.
