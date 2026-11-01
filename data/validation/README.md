# Stage 1 validation datasets

Raw data behind the Stage 1 figures in `figures/`. Experiments A and B use the stock
robosuite **Lift** task (one cube, no clutter, no observation or execution noise) with a
Franka Panda under OSC_POSE control, and the scripted top grasp in
`src/robot_uncertainty/sim/primitives.py`. Experiment C uses the same grasp on the
project's own tabletop scene (`src/robot_uncertainty/sim/tabletop.py`, M1.1). They
validate that primitive; they are not the project's main experiments.

| File | Rows | Produced by | Figure |
|---|---|---|---|
| `yaw_switch.csv` | 165 | `scripts/diagnose_yaw_switch.py` | `figures/yaw_switch_45deg_selection_vs_lift.png` |
| `grasp_alignment.csv` | 100 | `scripts/compare_grasp_alignment.py` | `figures/finger_gap_fixed_vs_aligned.png` |
| `palm_clearance.csv` | 2560 | `scripts/audit_palm_clearance.py` | `figures/palm_clearance_centre_vs_raised_grasp.png` |
| `pick_place_m1_2.csv` | 204 | `scripts/run_pick_place.py` | none (M1.2 regression reference) |
| `contact_params.csv` | 24 | `scripts/diagnose_contact_params.py` | none (M1.3a contact diagnostic) |

`tests/sim/data/m1_2_reference_qpos.npz` (from `scripts/capture_m1_2_qpos.py`) holds the
matching final MuJoCo states.

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

## Experiment C: palm clearance, centre vs raised grasp point (`palm_clearance.csv`)

**Question.** The tabletop target is a 4 x 4 x 8 cm block. The Panda's palm collision
mesh sits 31 mm behind the grip site, so gripping the block at its centre would put the
palm 9 mm below the top face. Does the raised grasp point (`top_grasp_height`: +15 mm,
for 6 mm of nominal palm clearance) keep the palm off the block in the real contact
geometry, while the fingers still grip properly?

**Design.** 16 configurations: the layouts of scene seeds 0-11, and the four extreme
footprint centres of the workspace, (±7, -17 or +3) cm from the table centre, at 45 deg
yaw. Each is grasped twice from the same settled state: `centre` (`grasp_height` 0, the
original behaviour) and `raised`. 32 grasps, one row per control step (2560 rows).

**Measurement** (`src/robot_uncertainty/sim/contacts.py`, the method of the regression
test `test_raised_grasp_keeps_palm_clearance_and_grips_with_both_pads`). After every
step, `clearance_mm` is the signed palm-block distance from MuJoCo's collider (palm
collision mesh vs block box; negative is penetration), computed on a copy of the model
with a 5 cm detection margin on the palm, or 50 cm when the palm is further away.
`palm_contact` and the force columns (normal force, N) come from the live simulation's
contact list. Where both report the palm, their depths agreed to within 0.15 mm in the
two grasps checked.
`mj_geomDistance` is not used: it returns spurious zeros for this pair.

**Columns.** `config`, `scene_seed`, the block's start pose (`x_cm`, `y_cm` from the
table centre, `yaw_deg`); `strategy`, `grasp_height_mm`, `success` (per grasp); `step`
(from 1), `phase`; `clearance_mm`, `palm_contact`, `palm_force_n`, `pad1_force_n`,
`pad2_force_n`. There is no wall-clock column, so a rerun must match exactly.

**Results.** All 32 grasps lifted the block. Raised: no palm contact at any step;
minimum clearance 6.1-8.9 mm, always during the close phase, 1.1-2.4 mm below the
clearance at the end of the descent (whether the block rises or the hand sinks as the
fingers close was not separated).
The least clearance, 6.1 mm, is at the corner (-7, +3) cm. Centre: palm contact on
37-40 steps in every configuration, as one unbroken stretch from late in the descent to
the end of the hold; penetration up to 3.2-4.5 mm. During the descent the palm pushes
the block into the table with up to 53 N (median 23 N), more than either pad's grip of
about 20 N, and it still presses on the block's top while the block is lifted and held
(median 9 N lifting, 8 N holding).

**Limitations.** One block size, the default friction and mass, no noise; the 6 mm
nominal clearance includes no allowance for vertical execution noise, which the M1
noise model should be checked against. Contact forces are MuJoCo's soft-contact
estimates, useful for comparison rather than as physical values.

## M1.2 regression references (`pick_place_m1_2.csv`, `m1_2_reference_qpos.npz`)

Frozen before any M1.3a change, from commit `c3f88e2` with no uncommitted changes to
tracked code (the `.npz` records the commit and an empty `dirty` field).

- `pick_place_m1_2.csv`: `python scripts/run_pick_place.py --first-seed 1000 --seeds 200
  --corners --out data/validation/pick_place_m1_2.csv`: validation seeds 1000-1199 and
  the four workspace corners at 45 degrees, one environment, in that order. 204/204
  placements succeeded. Columns are those of `run_pick_place.py` at that commit.
- `tests/sim/data/m1_2_reference_qpos.npz`: the complete final `env.sim.data.qpos` (arm,
  fingers and block, 16 values) after `pick_and_place` for seeds 1000-1009, each reset,
  settled and run as in `run_pick_place.py`, plus the MuJoCo (3.9.0), robosuite (1.5.2)
  and NumPy (1.26.4) versions.

`tests/sim/test_m1_2_regression.py` checks that the final qpos is **bitwise** equal, and
that `run_pick_place.py` reproduces every original column exactly except `wall_s`
(`utils.datasets.diff_rows` with `allow_extra_columns`, so columns the script has gained
since, `y`, `failure_stage` and `flags`, do not relax the comparison of the old ones).
Simulated time and step counts are compared exactly, floats at their recorded precision.
**These references are specific to this environment**: the same MuJoCo build, robosuite
and NumPy versions on the same kind of CPU (x86_64, AMD Ryzen 5 5500). The qpos test skips
on other versions; on other hardware the CSV may differ in the last recorded digits.

## M1.3a provisional outcome definitions (additions to proposal Section 3.1)

These extend the outcome label of the proposal for scenes with more than one block. They
are provisional, implemented in `src/robot_uncertainty/sim/outcome.py` and tested in
`tests/test_outcome.py` (synthetic records) and `tests/sim/test_distractor.py`
(simulated blocks). The proposal itself is unchanged.

**Records.** For every block, pick_and_place records the true state at the start of the
attempt (after the scene has settled), its position and orientation after every 20 Hz
control step, and the state after the final 0.25 s settle. Each block's outcome is a pure
function of the start record, the final record and the recorded poses.

**Displacement.** Final displacement is the planar distance between the start and final
centres; maximum displacement is the largest planar distance from the start centre over
every recorded pose, d_max = max_t ||p_xy(t) - p_xy(0)||, not the path length. A
**non-target** block with d_max > `disturbance_threshold` = 3 cm fails the attempt
(`displaced`), even if it later comes back near its start. The target is exempt: moving
it is the task. Poses are sampled once per control step (20 Hz), so a peak between two
steps is not seen.

**Toppling** is a final-state classification, not a tilt threshold. A block rests flat
on a face (named by its signed body axis, `-z` for the bottom of a block standing as
built) when that face's outward normal is within `face_flat_tilt` = 5 degrees of straight
down, all four of its corners are within 1 mm of the table top (`max_support_gap`), the
block touches the table and it is at rest (at most 5 mm/s and 0.05 rad/s). A block has
**toppled** when it ends resting flat on a face other than the one it started on: lying on
a side, or upside down. The start face must be the start of the attempt: a test that
lays a block down must take the start record from the upright block first. A toppled
block, of any role, fails the attempt.

**Initial state (precondition).** Every block must start resting flat on a known face,
by the same rule and the same tolerances as at the end. A start without one (moving
faster than 5 mm/s, not touching the table, or not flat) is invalid input: it is
neither a success nor a manipulation failure, and it gets no label.
- `object_outcome` raises `ValueError` naming the block if its start record has no
  resting face, and `outcome` raises if a supplied `ObjectOutcome` has
  `resting_face_start = None`.
- Before anything moves, `pick_and_place` checks `TabletopEnv.check_initial_state()`
  (at rest, upright within 2 degrees, no interpenetration) and the resting face of every
  block's start record, and raises `ValueError` if either fails. The check only reads the
  state; it never settles, repairs or resamples the scene, so the caller must settle
  first.
- A failure that arises during the attempt (a drop, a timeout, a block knocked over)
  is still a labelled outcome. Only the starting evidence is rejected.
- The pure classifier still accepts any known start face, such as a block lying on its
  side. The upright start is the environment's own rule, applied by
  `check_initial_state`.

**Diagnostics** (recorded, never decide the outcome by themselves): `tilted` (the
original vertical axis more than `significant_tilt` = 5 degrees from vertical at the
end, not toppled); `leaning` (no flat face, and touching something other than the table
or floor that can hold it up: another block or the robot); `unresolved_pose` (no flat
face, not leaning); `unsettled` (not at rest after the normal settle; no settle time was
added); `com_outside_base` (the centre of mass, projected straight down, outside the
projection of the block's original bottom face; direction-aware: for the 5 x 5 x 11 cm
block that happens past 24.4 degrees about an edge but 32.7 degrees about a corner).
Tilt is measured from the original vertical axis, so a block upside down has tilt 180.

**Label.** y = 1 requires the existing M1.2 success (verified grasp, no timeout, no drop,
released, retract completed, target validly placed), no block off the table, no
non-target block displaced and no block toppled. `failure_stage` is the earliest of
`grasp`, `transport`, `release`, `placement`, `scene`. The flags are non-exclusive; contact
flags (`collision_object`, `collision_environment`) are recorded without failing the
attempt. `PickPlaceResult.success` and `failure_reason` keep their M1.2 meaning.

**Contacts.** Contact pairs are read from MuJoCo's contact list after each control step,
with each pair's peak total normal force. A contact lasting less than one control step
(50 ms, 25 physics steps) can fall between samples and be missed: in 4 of 50 sampled
tall-distractor scenes a target-distractor impact during the carry toppled the
distractor with no contact recorded (the physics substeps show 49-57 steps of contact).
The physical outcome (displaced, toppled) is still labelled correctly.

## M1.3a contact parameters (`contact_params.csv`)

`scripts/diagnose_contact_params.py --out data/validation/contact_params.csv` records, for
every geom pair that touched during a run, the contact's parameters as MuJoCo formed them
(`condim`, the five friction values, `solref`, `solreffriction`, `solimp`, `mu`,
`includemargin`), both geoms' own `friction`, `solmix`, `priority`, `condim` and
`solref`, and whether everything matches `contact_params.expected_contact_params`. Runs:
`legacy` (scene seed 0, legacy pick-and-place), `distractor` (the same target with the
5 x 5 x 6 cm distractor touching it, on the face pair the fingers do not close on), and
both again with the target's friction set to (0.5, 0.005, 0.0001), restored afterwards.

**Model-wide settings, measured:** `npair` 0, `nexclude` 0, global contact override off,
elliptic friction cone, `impratio` 20. So every contact uses ordinary geom mixing; all 24
pairs matched the rules exactly.

| Contact | condim | Effective friction | solref | mu | Rule that applied |
|---|---|---|---|---|---|
| finger pad - target | 3 | 2, 0.05, 0.0001 | 0.015, 0.75 | 0.447 | max of pad (2, 0.05, 0.0001) and target; solref the solmix-weighted mean of 0.01/0.5 and 0.02/1 |
| finger mesh - target | 4 | 1, 0.005, 0.0001 | 0.02, 1 | 0.224 | max; condim the larger (mesh 4) |
| target - table | 3 | 1, 0.005, 0.0001 | 0.02, 1 | 0.224 | identical geoms |
| distractor - table | 3 | 1, 0.005, 0.0001 | 0.02, 1 | 0.224 | identical geoms |
| target - distractor | 3 | 1, 0.005, 0.0001 | 0.02, 1 | 0.224 | identical geoms (9 control steps of contact, as the target is lifted) |

All priorities are 0 and all `solmix` 1; `solimp` is (0.9, 0.95, 0.001, 0.5, 2) everywhere.
`mu` is measured as friction / sqrt(impratio), and only once a contact is an active
constraint.

**Friction sensitivity.** Halving the target's sliding friction to 0.5 changed **no**
effective friction: every geom the target touches has sliding friction of at least 1,
and the element-wise maximum takes it. Pad-target stays at 2, target-table and
target-distractor at 1, and every pair was in contact on the same number of steps. Randomising only
the object's friction would therefore have no physical effect in this scene; it needs a
different design (contact pairs, priorities, or changing both geoms), to be decided before
friction randomisation is implemented. The rules themselves (priority, solmix weighting,
direct-form solref, explicit pairs, exclusions, the global override) are checked against
MuJoCo on a synthetic model in `tests/sim/test_contact_params.py`.

## M1.3a distractor diagnostics (not committed; `results/m1_3a/`)

`python scripts/run_distractor_trials.py` writes one CSV per experiment to `results/m1_3a/`
(git-ignored) and prints the summary. Explicit, noise-free scenes; one distractor.

- **Both grasp axes, isolated** (`axes`, seeds 0-49 and the four corners, both candidates):
  108/108 succeeded with y = 1, and the commanded face axis equalled the gripped one every
  time. The candidate the legacy rule would not pick needs 45-89 degrees of wrist turn
  and 190 control steps (median) against 174.
- **Far distractor** (`far`, 20 scenes, tall distractor 16.7-30.4 cm from the hand footprint
  swept from the target to the goal, on the half of the table away from the robot): 20/20
  y = 1, no contact with the distractor, displacement 0.0 mm, still on its `-z` face.
- **Grasp-axis contrast** (`beside`, 10 scenes, distractor 4 cm face to face along
  candidate A's closing axis). Tall: candidate A failed 10/10 at the grasp; the palm landed
  on the distractor's top (peak 79-123 N), the descent timed out, the distractor was pushed
  2.7-9.5 mm (1.8-7.9 mm at the end) and tilted up to 6.9 degrees, then settled upright.
  Candidate B succeeded 10/10 with no contact. Short: both candidates succeeded 10/10;
  the fingers brushed the distractor once (5.5 N, candidate A).
- **Carry path** (`carry`, 5 scenes, tall distractor halfway along a path of at least 30 cm):
  the carried target (bottom 10 cm above the table) hit the 11 cm distractor in all 5.
  Two placements succeeded by the M1.2 check but had the distractor knocked onto its side
  75-78 mm away (y = 0, stage `scene`); two ended with the target lowered onto the fallen
  distractor (timeout in `lower`, stage `transport`); one grazed it (8.9 mm at most, 4.9 mm
  at the end, y = 1).
- **Throughput** (`throughput`, the existing random sampler with the tall distractor, seeds
  0-49): 0.78 attempts per second including reset and settle (pick_and_place alone 0.80 per
  second, median 1.26 s); M1.2 is 0.77-0.78. y = 1 in 30/50: 10 grasp failures (palm on the
  distractor), 9 scene failures (distractor toppled during the carry) and 1 transport
  failure. The sampler's 2 cm enclosing-circle clearance was designed for M1.1 and does
  not keep the hand or the carry path clear of a tall distractor.

No block was `unsettled` after the normal 0.25 s settle in any of these runs.

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
python scripts/plot_grasp_figures.py   # experiments A and B
python scripts/plot_palm_clearance.py  # experiment C

# Rerun the experiments (A and B about 5 minutes, C about 1 minute) into
# results/regenerated/ and compare with these files; the committed data is never replaced:
python scripts/plot_grasp_figures.py --regenerate
python scripts/plot_palm_clearance.py --regenerate

# Or run one experiment; output goes to results/ unless --out is given, and an
# existing file is only replaced with --overwrite:
python scripts/diagnose_yaw_switch.py --seeds 3 --out results/yaw_switch.csv
python scripts/compare_grasp_alignment.py --seeds 50 --out results/grasp_alignment.csv
python scripts/audit_palm_clearance.py --out results/palm_clearance.csv

# M1.2 references (must be captured from the reference commit) and M1.3a diagnostics:
python scripts/run_pick_place.py --first-seed 1000 --seeds 200 --corners --out results/pick_place.csv
python scripts/capture_m1_2_qpos.py --out results/m1_2_reference_qpos.npz
python scripts/diagnose_contact_params.py --out results/contact_params.csv
python scripts/run_distractor_trials.py            # about 6 minutes
pytest -m "not slow"                               # skips the four longest checks
```

Every column except `wall_s` is reproduced exactly on this setup. MuJoCo is
deterministic for the same build on the same CPU type; bitwise results on other
hardware or MuJoCo versions may differ slightly.
