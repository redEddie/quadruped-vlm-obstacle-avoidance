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

from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.devices import Se2Keyboard  # 키보드 디바이스 추가

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip

from go2_cfg import Go2EnvCfg, constant_commands, keyboard_commands


from agent_module import start_input_listener, llm_command_callback
from env_state import set_policy_index, get_policy_index

##
# Environment configuration
##

@configclass
class LanguageEnvCfg(Go2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # self.scene.camera = None
        # self.scene.height_scanner = None
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
            self.scene.height_scanner.debug_vis = True
        self.scene.height_scanner.debug_vis = False

        # LLM-based velocity 콜백으로 교체
        start_input_listener()
        self.observations.policy.velocity_commands.func = llm_command_callback

        self.policy_walk = self.load_policy_walk()
        self.policy_climb = self.load_policy_climb()


    def load_policy_walk(self):
        # 레벨 정책 로드
        policy_path = "pretrained/policy-walk.pt"
        # 정책 파일 존재 확인
        if not check_file_path(policy_path):
            raise FileNotFoundError(f"정책 파일 '{policy_path}'이(가) 존재하지 않습니다.")
        file_bytes = read_file(policy_path)
        # 정책 jit 로드
        policy = torch.jit.load(file_bytes).to(self.sim.device).eval()
        return policy

    def load_policy_climb(self):
        # 레벨 정책 로드
        policy_path = "pretrained/policy-climb.pt"
        # 정책 파일 존재 확인
        if not check_file_path(policy_path):
            raise FileNotFoundError(f"정책 파일 '{policy_path}'이(가) 존재하지 않습니다.")
        file_bytes = read_file(policy_path)
        # 정책 jit 로드
        policy = torch.jit.load(file_bytes).to(self.sim.device).eval()
        return policy


    def compute_action(self, obs: torch.Tensor) -> torch.Tensor:
        # agent_module 에서 결정된 policy_index(0: walk, 1: crawl) 사용
        policy_index = get_policy_index()
        if policy_index == 1:
            # crawl policy
            return self.policy_climb(obs)
        else:
            # walk policy (default)
            return self.policy_walk(obs)