"""XgoDuck robot configs for the shared task stack.

Same joint, body, site, and foot-collision names as the upstream tasks.
Actuator is the HLS1910 BAM model. Speed comes from voltage and back-EMF
(max_velocity=100 disables the firmware goal-slew).

Walk uses a ±22° HOME pose and ±1.5° serial gear play.
All XgoDuck task models use the same ±1.5° servo play and HLS1910 drive.
Trunk heights keep the soles clear of the floor.
"""

from copy import deepcopy
from dataclasses import replace
import math
import os
from pathlib import Path

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from mjlab_microduck.actuator import BacklashEncoderBamActuatorCfg, FrictionDRBamActuatorCfg


_ROBOT_DIR: Path = Path(os.path.dirname(__file__)) / "xgoduck"
_HLS1910_JSON: Path = _ROBOT_DIR / "params" / "1910_m6.json"

XGODUCK_WALK_XML: Path = _ROBOT_DIR / "robot_walk.xml"
XGODUCK_WALK_BACKLASH_XML: Path = _ROBOT_DIR / "robot_walk_backlash.xml"
XGODUCK_ALLCOLLISIONS_XML: Path = _ROBOT_DIR / "robot_allcollisions.xml"
XGODUCK_SITSTAND_BACKLASH_XML: Path = _ROBOT_DIR / "robot_allcollisions_backlash.xml"

assert XGODUCK_WALK_XML.exists(), f"XML not found: {XGODUCK_WALK_XML}"
assert XGODUCK_WALK_BACKLASH_XML.exists(), f"XML not found: {XGODUCK_WALK_BACKLASH_XML}"
assert XGODUCK_ALLCOLLISIONS_XML.exists(), f"XML not found: {XGODUCK_ALLCOLLISIONS_XML}"
assert _HLS1910_JSON.exists(), f"HLS1910 BAM params not found: {_HLS1910_JSON}"

# Trunk z at XGODUCK STAND (hip_pitch/ankle ±22°) so soles sit 1 mm above the floor.
# CAD-zero (all joints 0) is 0.129394 m; the HOME leg bend needs ~10 mm more.
XGODUCK_STAND_Z = 0.139158
# Same kinematic sole-clearance height at the sitstand SIT keyframe
# (knee ±1.35, hip_pitch ±0.4079, ankle/hip_roll 0; neck stays HOME).
XGODUCK_SIT_Z = 0.085868

_DEG22 = math.radians(22)
XGODUCK_HOME_FRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, XGODUCK_STAND_Z),
    joint_pos={
        r".*hip_yaw.*": 0.0,
        r".*left_hip_roll.*": -0.0873,
        r".*right_hip_roll.*": 0.0873,
        r".*left_hip_pitch.*": -_DEG22,
        r".*right_hip_pitch.*": _DEG22,
        r".*left_knee.*": -0.0049,
        r".*right_knee.*": 0.0049,
        r".*left_ankle.*": _DEG22,
        r".*right_ankle.*": -_DEG22,
        r".*neck_pitch.*": 0.3491,
        r".*head_pitch.*": 0.3491,
        r".*head_yaw.*": 0.0,
        r".*head_roll.*": 0.0,
    },
    joint_vel={".*": 0.0},
)

# Keep Walk HOME independent of tricks and roller configuration.
# At 22 degrees, nominal soles clear the floor by 1 mm.
XGODUCK_WALK_STAND_Z = XGODUCK_STAND_Z
XGODUCK_WALK_HOME_FRAME = deepcopy(XGODUCK_HOME_FRAME)
XGODUCK_WALK_HOME_FRAME.pos = (0.0, 0.0, XGODUCK_WALK_STAND_Z)
# Passive hinges must not match the broad servo HOME patterns.
XGODUCK_WALK_HOME_FRAME.joint_pos = {
    r"^passive_.*_backlash$": 0.0,
    **{r"^(?!passive_)" + pattern.lstrip("^"): value
       for pattern, value in XGODUCK_HOME_FRAME.joint_pos.items()},
}

# ── Posa in piedi reale per il solo rialzo (hd1910-standup) ──────────────────
# Anche ±26,2°, caviglie ±18°, resto uguale a HOME.
# z calcolata con stand_z.py: suola a 1 mm dal suolo (HOME ±22° dà 0.139158).
XGODUCK_STANDUP_STAND_Z = 0.141762
_standup_pose = dict(XGODUCK_HOME_FRAME.joint_pos)
_standup_pose[r".*left_hip_pitch.*"] = -math.radians(26.2)
_standup_pose[r".*right_hip_pitch.*"] = math.radians(26.2)
_standup_pose[r".*left_ankle.*"] = math.radians(18.0)
_standup_pose[r".*right_ankle.*"] = -math.radians(18.0)
XGODUCK_STANDUP_HOME_FRAME = deepcopy(XGODUCK_WALK_HOME_FRAME)
XGODUCK_STANDUP_HOME_FRAME.pos = (0.0, 0.0, XGODUCK_STANDUP_STAND_Z)
XGODUCK_STANDUP_HOME_FRAME.joint_pos = {
    r"^passive_.*_backlash$": 0.0,
    **{r"^(?!passive_)" + p.lstrip("^"): v for p, v in _standup_pose.items()},
}


# Named `*_collision` geoms. Bit 0 = world/floor, bit 1 = head↔base.
# Walk: feet only. Standup: every link has a mesh and hits the floor.
# Feet and other links: contype=1 conaffinity=0 → floor only, no robot-robot.
# trunk_collision (base) and head_collision (head_roll / jaw_soft):
#   contype=3 conaffinity=2 → floor and each other, not the other links.
XGODUCK_COLLISION = CollisionCfg(
    geom_names_expr=[".*_collision"],
    contype={
        r"^(trunk_collision|head_collision)$": 3,
        ".*_collision": 1,
    },
    conaffinity={
        r"^(trunk_collision|head_collision)$": 2,
        ".*_collision": 0,
    },
    condim={r"^(left|right)_foot_collision$": 3, ".*_collision": 1},
    priority={r"^(left|right)_foot_collision$": 1},
    friction={r"^(left|right)_foot_collision$": (1.0,)},
)


# Get-up tasks: preserve every link's ground contact, and allow exactly
# head-trunk plus the head/left-foot/right-foot clique. Bit 0 is floor,
# bit 1 is head-trunk, bit 2 is shared by the head and the two soles.
XGODUCK_GETUP_COLLISION = replace(
    XGODUCK_COLLISION,
    contype={
        r"^head_collision$": 7,
        r"^trunk_collision$": 3,
        r"^(left|right)_foot_collision$": 5,
        ".*_collision": 1,
    },
    conaffinity={
        r"^head_collision$": 6,
        r"^trunk_collision$": 2,
        r"^(left|right)_foot_collision$": 4,
        ".*_collision": 0,
    },
)

# HLS1910 identification (sts3215 firmware). max_velocity=100 disables the
# firmware goal-slew so speed comes from voltage + back-EMF.
# kp_fw=5 matches the real Feetech P. BAM converts XML <position> to motor;
# XML kp is only a viewer companion.
actuators = BacklashEncoderBamActuatorCfg(
    json_path=str(_HLS1910_JSON),
    target_names_expr=(r"^(?!passive_).*",),
    kp_fw=5.0,
    vin_range=(7.4, 8.0),
    vin_drop_gain_range=(0.0, 0.2),
    vin_min=7.0,  # sag floor below sampled vin (7.4–8.0) so load can still drop V
    # mjlab applies delay through this buffer, not BAM JSON command_delay.
    # Every XgoDuck task uses 0–6 physics steps at 5 ms = 0–30 ms.
    delay_min_lag=0,
    delay_max_lag=6,
)


def _disable_unnamed_geoms(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """CollisionCfg looks up geoms by name, so unnamed collision meshes leak."""
    for geom in spec.geoms:
        if not geom.name:
            geom.contype = 0
            geom.conaffinity = 0
    return spec


def get_walk_spec() -> mujoco.MjSpec:
    return _disable_unnamed_geoms(mujoco.MjSpec.from_file(str(XGODUCK_WALK_BACKLASH_XML)))


def get_standup_spec() -> mujoco.MjSpec:
    return _disable_unnamed_geoms(mujoco.MjSpec.from_file(str(XGODUCK_SITSTAND_BACKLASH_XML)))


# Shared backlash-servo defaults: 0–6 physics steps for every XgoDuck task.
_walk_actuators = BacklashEncoderBamActuatorCfg(
    json_path=str(_HLS1910_JSON),
    target_names_expr=(r"^(?!passive_).*",),
    kp_fw=5.0,
    vin_range=(7.4, 8.0),
    vin_drop_gain_range=(0.0, 0.2),
    vin_min=7.0,
    delay_min_lag=0,
    delay_max_lag=6,
)

XGODUCK_WALK_ROBOT_CFG = EntityCfg(
    spec_fn=get_walk_spec,
    init_state=XGODUCK_WALK_HOME_FRAME,
    collisions=(XGODUCK_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        # Walk: 0–6 physics steps at 5 ms = 0–30 ms.
        actuators=(replace(_walk_actuators, delay_min_lag=0, delay_max_lag=6),),
        soft_joint_pos_limit_factor=0.9,
    ),
)

XGODUCK_STANDUP_ROBOT_CFG = EntityCfg(
    spec_fn=get_standup_spec,
    init_state=deepcopy(XGODUCK_WALK_HOME_FRAME),
    collisions=(XGODUCK_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)


def get_sitstand_spec() -> mujoco.MjSpec:
    return _disable_unnamed_geoms(mujoco.MjSpec.from_file(str(XGODUCK_SITSTAND_BACKLASH_XML)))


# SitStand needs full-body contacts with the same ±1.5° gear play as Walk.
_sitstand_home = deepcopy(XGODUCK_HOME_FRAME)
_sitstand_home.joint_pos = {
    r"^passive_.*_backlash$": 0.0,
    **{r"^(?!passive_)" + pattern.lstrip("^"): value
       for pattern, value in XGODUCK_HOME_FRAME.joint_pos.items()},
}
XGODUCK_SITSTAND_ROBOT_CFG = deepcopy(XGODUCK_STANDUP_ROBOT_CFG)
XGODUCK_SITSTAND_ROBOT_CFG.spec_fn = get_sitstand_spec
XGODUCK_SITSTAND_ROBOT_CFG.init_state = _sitstand_home
XGODUCK_SITSTAND_ROBOT_CFG.articulation.actuators = (deepcopy(_walk_actuators),)


def get_roulade_spec() -> mujoco.MjSpec:
    """Keep the legacy Roulade trunk CoM independent of the Walk model tuning."""
    spec = get_standup_spec()
    trunk = spec.body("trunk_base")
    ipos = list(trunk.ipos)
    ipos[0] = -0.01327575578150381
    trunk.ipos = ipos
    return spec


XGODUCK_ROULADE_ROBOT_CFG = deepcopy(XGODUCK_STANDUP_ROBOT_CFG)
XGODUCK_ROULADE_ROBOT_CFG.spec_fn = get_roulade_spec
XGODUCK_ROULADE_ROBOT_CFG.articulation.actuators = (deepcopy(_walk_actuators),)


# ── hd1910-standup: pila di schede sopra il tronco (solo rialzo) ─────────────
# Arduino UNO Q + millefori con IMU + Bus Servo Adapter, avvitati sui distanziali originali.
# Sistema del tronco: X avanti, Y sinistra, Z su. Metri e kg.
# La pila sta sulla SCHIENA (lato -X, opposto al collo) e sporge all'indietro.
# Dimensioni misurate; posizione dall'editor (faccia interna sulle teste dei distanziali, x = -47 mm).
STACK_SIZE = (0.035, 0.055, 0.069)   # lungo X (sporgenza), Y (larghezza), Z (altezza)
STACK_INNER_X = -0.036               # faccia appoggiata ai distanziali
STACK_CENTER_Z = -0.030              # centro in altezza
STACK_MASS = 0.055                   # spostata dal tronco: massa totale invariata


def _find_body(spec, name):
    finder = getattr(spec, "body", None) or getattr(spec, "find_body")
    body = finder(name)
    if body is None:
        raise RuntimeError(f"corpo {name!r} non trovato nello spec")
    return body


def add_electronics_stack(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    trunk = _find_body(spec, "trunk_base")
    m_old = float(trunk.mass)
    if m_old <= STACK_MASS:
        raise RuntimeError(f"massa del tronco nello spec = {m_old}: serve un <inertial> esplicito")
    scale = (m_old - STACK_MASS) / m_old
    trunk.mass = m_old - STACK_MASS
    trunk.inertia = [v * scale for v in trunk.inertia]
    if any(abs(v) > 0 for v in trunk.fullinertia):
        trunk.fullinertia = [v * scale for v in trunk.fullinertia]

    L, W, H = STACK_SIZE
    body = trunk.add_body(name="electronics_stack", pos=[STACK_INNER_X - L / 2, 0.0, STACK_CENTER_Z])
    body.mass = STACK_MASS
    body.ipos = [0.0, 0.0, 0.0]
    body.inertia = [
        STACK_MASS / 12 * (W * W + H * H),
        STACK_MASS / 12 * (L * L + H * H),
        STACK_MASS / 12 * (L * L + W * W),
    ]
    if hasattr(body, "explicitinertial"):
        body.explicitinertial = True
    body.add_geom(name="stack_collision", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[L / 2, W / 2, H / 2], group=3, rgba=[0.9, 0.3, 0.1, 0.5])
    # copia solo visiva (gruppo 2, senza collisione) per vederla nel viewer
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[L / 2, W / 2, H / 2],
                  group=2, contype=0, conaffinity=0, rgba=[0.1, 0.6, 0.2, 0.8])
    return spec


def get_getup_spec_with_stack() -> mujoco.MjSpec:
    return add_electronics_stack(get_standup_spec())
