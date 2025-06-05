import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import check_file_path, read_file
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import SceneEntityCfg

from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.devices import Se2Keyboard  # 키보드 디바이스 추가
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg, SubTerrainBaseCfg
from isaaclab.sim import VisualMaterialCfg

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip
from terrains import MazeTerrainCfg
##
# Custom observation terms
##


def constant_commands(env: ManagerBasedEnv) -> torch.Tensor:
    """The generated command from the command generator."""
    return torch.tensor([[1, 0, 0]], device=env.device).repeat(env.num_envs, 1)  # vx, vy, wz


def keyboard_commands(env: ManagerBasedEnv) -> torch.Tensor:
    """키보드로부터 명령을 받아옵니다."""
    if not hasattr(env, "keyboard"):
        env.keyboard = Se2Keyboard(
            v_x_sensitivity=1.0, v_y_sensitivity=1.0, omega_z_sensitivity=3.0
        )
        env.keyboard.reset()
    
    command = env.keyboard.advance()
    return torch.tensor(command, device=env.device, dtype=torch.float32).unsqueeze(0).repeat(env.num_envs, 1)

##
# Scene definition
##

LOCAL_GO2_USD_PATH = r"C:/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/Unitree/Go2/go2.usd"

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Example scene configuration."""

    # add terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # plane, generator
        terrain_generator=TerrainGeneratorCfg(
            size=(6*1, 6*1),
            border_width=0.2,
            sub_terrains={
                "maze": MazeTerrainCfg(
                    rows=6,
                    cols=6,
                    cell_size=1,
                )
            }
        ),
        # visual_material=VisualMaterialCfg(color=(1.0, 0.0, 0.7, 1.0)),
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # add robot
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot", 
        spawn=UNITREE_GO2_CFG.spawn.replace(usd_path=LOCAL_GO2_USD_PATH)
    )

    # sensors
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.35, 0.0, 0.02), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )

    top_view = CameraCfg(
        prim_path="/World/camera_top_view",
        update_period=0.1,
        height=720,
        width=720,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(3.0, 3.0, 10.0), rot=(0.0, -1, 1.0, 0.0), convention="ros"),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    light = AssetBaseCfg(
        prim_path="/World/light2",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2000.0),
    )


##
# MDP settings
##

@configclass
class ActionsCfg:
    """MDP를 위한 액션 명세"""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=[".*"], 
        scale=0.5, 
        use_default_offset=True
    )

@configclass
class ObservationsCfg:
    """MDP를 위한 관측 명세"""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=constant_commands)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()



@configclass
class EventCfg:
    """이벤트 설정"""

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


##
# Environment configuration
##


@configclass
class Go2EnvCfg(ManagerBasedEnvCfg):
    scene: MySceneCfg = MySceneCfg(
        num_envs=16,
        env_spacing=0.5 * max(6, 6) + 1.0
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.scene.num_envs = 16  # 나중에 main()에서 값을 넣어줘도 됨
        self.sim.device = "cuda"
        # self.scene.terrain.terrain_type = "plane"

        self.decimation = 4  
        self.sim.dt = 0.005  
        self.sim.physics_material = self.scene.terrain.physics_material

        # self.scene.camera = None
        # self.scene.height_scanner = None
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt