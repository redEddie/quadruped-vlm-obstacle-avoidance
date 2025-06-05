import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="평지 환경에서 Go2 로봇을 위한 튜토리얼")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=16, help="생성할 환경의 수")
parser.add_argument("--draw", action="store_true", default=False, help="포인트 클라우드를 시각화할지 여부")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")

parser.add_argument("--rows",      type=int,   default=3,    help="미로 셀 행 개수")
parser.add_argument("--cols",      type=int,   default=3,    help="미로 셀 열 개수")
parser.add_argument("--cell_size", type=float, default=2,   help="각 셀 크기(m)")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# 입력된 인자를 마음대로 조작; cli명령어 치기 귀찮아
args_cli.num_envs = 1
args_cli.enable_cameras = True  # 카메라를 기본적으로 활성화

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import os
import pprint
import inspect
import io

import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from datetime import datetime

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
from isaaclab.devices import Se2Keyboard  # 키보드 디바이스 추가
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils.dict import print_dict
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from env_state import get_policy_index, set_env  
from terrains import MazeTerrainCfg
##
# Pre-defined configs
##
from llm_cfg import LanguageEnvCfg
env_cfg = LanguageEnvCfg()

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_dir = os.path.join("output", timestamp)
os.makedirs(save_dir, exist_ok=True)
top_view_dir = save_dir

def illegal_contact(env: ManagerBasedEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history

    base_force_vec = net_contact_forces[:, :, 0, :]
    base_force_mag = torch.norm(base_force_vec, dim=-1) 
    
    # check if any contact force exceeds the threshold
    return torch.any(base_force_mag > threshold, dim=1)

def capture(env: ManagerBasedEnv, sim_time: float):
    camera = env.scene[SceneEntityCfg("top_view").name]
    rgb_data = camera.data.output["rgb"]
    save_path = os.path.join(top_view_dir, f"{sim_time:.2f}.png")
    image_np = rgb_data.cpu().numpy()
    if image_np.ndim > 3:
        image_np = np.squeeze(image_np, axis=0)
    pil = Image.fromarray(image_np.astype(np.uint8))
    pil.save(save_path)

##
# Scene definition
##


def main():
    """메인 함수"""
    # 기본 환경 설정
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    maze_cfg = env_cfg.scene.terrain.terrain_generator
    maze_cfg.sub_terrains["maze"].rows = args_cli.rows
    maze_cfg.sub_terrains["maze"].cols = args_cli.cols
    maze_cfg.sub_terrains["maze"].cell_size = args_cli.cell_size
    maze_cfg.size = (args_cli.rows * args_cli.cell_size, args_cli.cols * args_cli.cell_size)

    env = ManagerBasedEnv(cfg=env_cfg)
    set_env(env)

    log_root_path = os.path.join("logs", "go2_llm")
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Log directory: {log_root_path}")
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root_path, datetime_str)
    os.makedirs(log_dir, exist_ok=True)

    # video 기록 기능은 나중에 추가
    # if args_cli.video:
    #     video_kwargs = {
    #         "video_folder": os.path.join(log_dir, "videos", "play"),
    #         "step_trigger": lambda step: step == 0,
    #         "video_length": args_cli.video_length,
    #         "disable_logger": True,
    #     }
    #     print("[INFO] Recording videos during training.")
    #     print_dict(video_kwargs, nesting=4)
    #     env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # 물리 시뮬레이션
    count = 0
    obs, _ = env.reset()
    
    # 시간 추적을 위한 변수 추가
    sim_time = 0.0
    sim_dt = env.sim.get_physics_dt()  # 시뮬레이션 시간 간격 가져오기
    
    while simulation_app.is_running():
        with torch.inference_mode():
            # 리셋 (1000 * 0.01초 = 10초)
            # if count % 1000 == 0:
            #     # if obs["base_pos"][0, 2] <0.05:
            #     obs, _ = env.reset()
            #     count = 0
            #     print("-" * 80)
            #     print("[INFO]: 환경 리셋 중...")

            done = illegal_contact(env, threshold=10.0, sensor_cfg=SceneEntityCfg("contact_forces"))
            if done:
                obs, _ = env.reset()
                count = 0
                print("-" * 80)
                print("[INFO]: 환경 리셋 중...")
            
            if sim_time % 1.0 < sim_dt:
                print(f"Time: {sim_time:.2f}s")
            
            if sim_time % 0.1 < sim_dt:
                capture(env, sim_time)

            # 액션 추론
            policy_index = get_policy_index()
            if policy_index == 1:
                action = env_cfg.policy_climb(obs["policy"])
            else:
                policy_obs = obs["policy"] # without height_scan
                obs_walk = policy_obs[:, :48] # [B, 48]
                action = env_cfg.policy_walk(obs_walk)
            # 환경 스텝
            obs, _ = env.step(action)
            # 카운터 업데이트
            count += 1
            # 시간 업데이트
            sim_time += sim_dt

    # 환경 종료
    env.close()


if __name__ == "__main__":
    # 메인 함수 실행
    main()
    # 시뮬레이션 앱 종료
    simulation_app.close()
    