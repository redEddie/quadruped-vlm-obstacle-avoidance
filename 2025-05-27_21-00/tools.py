# tools.py

import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
from openai import OpenAI
import base64
import io
from datetime import datetime
import numpy as np


import json

from dotenv import load_dotenv
load_dotenv()

from typing import Annotated

from langchain_google_community import GoogleSearchAPIWrapper
from langchain.tools import tool
from langgraph.types import Command, interrupt
from langchain_core.tools import InjectedToolCallId
from langchain_core.messages import ToolMessage

import env_state
from isaaclab.managers import SceneEntityCfg


search = GoogleSearchAPIWrapper()

@tool
def google_search_tool(query: str) -> str:
    """Search Google for the given query and return raw results as a string."""
    return search.run(query)


@tool
def human_assistance(tool_call_id: Annotated[str, InjectedToolCallId]) -> str:
    """ 
    사람이 판단을 내려야 하는 상황에서 확인 또는 수정을 요청합니다.

    이 툴은 LLM의 실행 흐름을 일시적으로 중단하고,  
    사용자에게 **"이 내용이 맞습니까?"**라고 질문하여  
    **예/아니오(또는 수정된 입력)** 형태의 응답을 기다립니다.

    용도:
        - 모델의 판단에 불확실성이 있을 때
        - 중요한 결정 또는 사용자 승인이 필요한 경우
        - 사람이 직접 판단해야 할 선택지를 제시할 때

    반환:
        - Command: 사용자의 응답을 받아 대화 흐름을 업데이트합니다.
    """
    human_response = interrupt({"question": "Is this correct?"})
    if human_response.get("correct", "").lower().startswith("y"):
        response = "Correct!"
    else:
        response = f"Made a correction: {human_response}"

    state_update = {"messages": [ToolMessage(response, tool_call_id=tool_call_id)]}
    return Command(update=state_update)

@tool
def set_velocity(tool_call_id: Annotated[str, InjectedToolCallId], vx: float, vy: float, wz: float):
    """
    로봇의 선속도(vx, vy)와 회전속도(wz)를 설정하여 전진, 후진, 좌우 이동 및 회전을 수행합니다.

    매개변수:
        vx (float): X축 방향 선속도 (m/s)
            - 양수: 전진
            - 음수: 후진

        vy (float): Y축 방향 선속도 (m/s)
            - 양수: 왼쪽으로 평행 이동(strafe left)
            - 음수: 오른쪽으로 평행 이동(strafe right)

        wz (float): Z축(위쪽) 기준의 회전 속도(라디안/초)
            - 양수: 반시계 방향 회전 → **좌회전**
            - 음수: 시계 방향 회전 → **우회전**
            - **최대 절댓값: 0.8 rad/s** (과도한 회전은 제동이 어려움)

    지원 동작:
        - 직선 이동 (전진, 후진, 좌우)
        - 제자리 회전 (vx=vy=0, wz≠0)
        - 곡선 회전 (vx≠0, wz≠0) → **전진하면서 좌회전/우회전**
            예:
                - vx > 0, wz > 0 → **전진하면서 좌회전**
                - vx > 0, wz < 0 → **전진하면서 우회전**

    동작 방식:
        1. 안전을 위해 vx, vy는 [-1.0, 1.0], wz는 [-0.8, 0.8] 범위로 자동 클리핑됩니다.
        2. 멈추는 명령 (vx=vy=wz=0)인 경우:
            a. 이전 속도의 50%로 줄여 천천히 감속
            b. 약 0.2초간 감속된 속도로 유지
            c. 그 후 완전히 정지
        3. 이는 로봇이 급격하게 멈추지 않도록 하여 자세를 안정적으로 유지하게 합니다.

    예시:
        # 전진
        set_velocity(1.0, 0.0, 0.0)

        # 후진
        set_velocity(-0.5, 0.0, 0.0)

        # 왼쪽으로 이동
        set_velocity(0.0, 0.5, 0.0)

        # 제자리 좌회전
        set_velocity(0.0, 0.0, 0.8)

        # 전진하면서 좌회전
        set_velocity(0.5, 0.0, 0.5)

        # 전진하면서 우회전
        set_velocity(0.5, 0.0, -0.5)

        # 부드럽게 멈춤
        set_velocity(0.0, 0.0, 0.0)
    """
    velocity = {"vx": vx, "vy": vy, "wz": wz}
    return Command(update={
        "velocity": velocity,
        "messages": [ToolMessage(json.dumps({"velocity": velocity}), tool_call_id=tool_call_id)]
    })

@tool
def proprioception(tool_call_id: Annotated[str, InjectedToolCallId]):
    """
    로봇의 현재 전역 위치, 자세, 속도, 이동 거리 등의 정보를 조회하여  
    목표 위치 도달 여부나 움직임의 정확도를 평가합니다.

    사용 시점:
        - 로봇이 움직이기 **전 또는 후에 호출**하여,
          실제로 얼마나 이동했는지, 목표 위치에 도달했는지를 확인할 때 사용합니다.

    반환값:
        dict: {
            "position": {"x": float, "y": float, "z": float},  # 로봇의 전역 위치
            "orientation": {"x": float, "y": float, "z": float, "w": float},  # 쿼터니언 형태의 자세 정보
            "base_velocity": [vx, vy, vz],  # 선속도 (m/s)
            "base_angular_velocity": [wx, wy, wz],  # 각속도 (rad/s)
            "projected_gravity": [gx, gy, gz]  # 로봇에 작용하는 중력 벡터
        }
    """
    env = env_state.get_env()
    if env is None:
        raise RuntimeError("env가 아직 설정되지 않았습니다. llm_command_callback에서 env_state.set_env를 호출했는지 확인하세요.")

    scene = env.scene
    asset_cfg = SceneEntityCfg("robot")
    asset = scene[asset_cfg.name]

    # pose 정보
    body_pose_w = asset.data.body_state_w[:, asset_cfg.body_ids, :7]
    pos = body_pose_w[..., :3] - scene.env_origins.unsqueeze(1)

    position = {
        "x": float(pos[0, 0, 0]),
        "y": float(pos[0, 0, 1]),
        "z": float(pos[0, 0, 2])
    }
    orientation = {
        "x": float(body_pose_w[0, 0, 3]),
        "y": float(body_pose_w[0, 0, 4]),
        "z": float(body_pose_w[0, 0, 5]),
        "w": float(body_pose_w[0, 0, 6])
    }

    # Tensor → 리스트 변환
    base_velocity = asset.data.root_lin_vel_b[0].cpu().tolist()
    base_angular_velocity = asset.data.root_ang_vel_b[0].cpu().tolist()
    projected_gravity = asset.data.projected_gravity_b[0].cpu().tolist()

    # 디버그 출력
    print("position: ", position)
    print("orientation: ", orientation)
    print("base_velocity: ", base_velocity)
    print("base_angular_velocity: ", base_angular_velocity)
    print("projected_gravity: ", projected_gravity)

    # 반환할 메시지
    payload = {
        "position": position,
        "orientation": orientation,
        "base_velocity": base_velocity,
        "base_angular_velocity": base_angular_velocity,
        "projected_gravity": projected_gravity
    }

    return Command(update={
        **payload,
        "messages": [ToolMessage(json.dumps({"proprioception": payload}), tool_call_id=tool_call_id)]
    })

@tool
def switch_policy(tool_call_id: Annotated[str, InjectedToolCallId]):
    """
    로봇의 제어 정책을 "걷기(walking)"와 "오르기(climbing)" 사이에서 전환합니다.  
    지형의 높이에 따라 적절한 정책을 선택하여 로봇의 이동 방식을 유연하게 조절합니다.

    정책 동작 설명:
        - 초기 정책 (index=0): **"걷기(walking)"** — 평지에서 효율적이고 부드러운 보행을 수행합니다.
        - 전환 시 정책 (index=1): **"오르기(climbing)"** — 바닥의 높이 정보를 활용하여, 계단이나 울퉁불퉁한 지형에 적응하는 보행 패턴을 사용합니다.
        - 평탄한 지형에서는 걷기 정책이 적합하고, 장애물이나 높낮이가 있는 지형에서는 오르기 정책을 사용하는 것이 안전합니다.

    반환값:
        - `Command`: 현재 설정된 정책 인덱스를 업데이트합니다. (`0`: 걷기, `1`: 오르기)
    """
    policy_index = get_policy_index()
    policy_index = (policy_index + 1) % 2
    set_policy_index(policy_index)
    return Command(update={
        "messages": [ToolMessage(json.dumps({"policy_index": policy_index}), tool_call_id=tool_call_id)]
    })


@tool
def process_rgb(tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """
    로봇 전방의 카메라에서 RGB 이미지를 캡처하여 GPT-4 Vision 모델로 분석을 수행합니다.

    이 툴은 로봇이 주변 환경을 명확하게 인식하지 못하거나, 경로가 차단된 상황에서 사용됩니다.
    정확한 이미지를 확보하기 위해, **이미지를 촬영하기 전 반드시 로봇은 완전히 정지해야 합니다.**

    이미지 분석 결과는 **장면에 대한 텍스트 설명**으로 반환되며, 이후 다음 행동 결정(예: 방향 전환, 장애물 인식 등)에 활용됩니다.

    일반적인 사용 사례:
    - 명확한 주행 경로가 보이지 않을 때 주변을 다시 평가할 때
    - 천천히 회전한 후 주변을 관찰하여 진로를 탐색할 때
    - 시각적 장면을 의미적으로 해석하여 언어 기반 제어를 지원할 때

    ⚠️ 주의: 이미지 촬영 중에는 로봇이 절대 움직이지 않아야 합니다.
    """
    env = env_state.get_env()
    camera = env.scene[SceneEntityCfg("camera").name]
    image_tensor = camera.data.output["rgb"]
    image_np = image_tensor.cpu().numpy()
    if image_np.ndim > 3:
        image_np = np.squeeze(image_np, axis=0)

    # 이미지 저장 경로 생성
    save_dir = os.path.join("output", "rgb")
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(save_dir, f"{timestamp}.png")

    # numpy 배열을 PIL 이미지로 변환 및 저장 (uint8 보장)
    pil = Image.fromarray(image_np.astype(np.uint8))
    pil.save(save_path)
    print(f"[이미지 저장]: {save_path}")

    # 2) PNG 포맷으로 인코딩 → base64 data URL
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    data_url = f"data:image/png;base64,{b64}"

    client = OpenAI()
    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "what's in this image?"},
                    {
                        "type": "input_image",
                        "image_url": data_url,
                        "detail": "low"
                    }
                ]
            }
        ]
    )
    
    print(f"[LLM 입력 이미지]: {save_path}")
    
    return Command(update={
        "messages": [
            ToolMessage(response.output_text, tool_call_id=tool_call_id)
        ]
    })


@tool
def process_depth(tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """
    당신은 로봇 주변의 깊이 정보를 해석하여 안전한 이동을 도와야 합니다.

    이 툴은 로봇이 주변 환경을 명확하게 인식하지 못하거나, 경로가 차단된 상황에서 사용됩니다.
    정확한 이미지를 확보하기 위해, **이미지를 촬영하기 전 반드시 로봇은 완전히 정지해야 합니다.**

    이 도구는 로봇 전방의 **깊이 이미지**를 처리합니다. 이 이미지는 색상으로 거리 정보를 시각화한 것이며, 다음과 같은 기준을 따릅니다:
    - **파란색**: 가까운 거리 (예: 0~2m)
    - **빨간색**: 먼 거리 (예: 8~10m)
    ※ 이 시스템은 '파란색 = 가까움', '빨간색 = 멀어짐'으로 해석해야 합니다.
    ※ 깊이 값은 **0m부터 최대 10m까지의 범위**를 갖습니다. 이 범위를 기준으로 물체까지의 거리를 추정할 수 있습니다.

    ⚠️ 주의: 파란색이라고 해서 항상 통과 가능한 공간은 아닙니다.
    - 예: 로봇 바로 앞에 있는 **가까운 벽**은 깊이 값이 작기 때문에 이미지 전체가 파란색으로 나타날 수 있으며, 이는 **열린 공간으로 착각할 위험이 있습니다.**

    해석 기준:
    - 이미지 상단에서 하단으로 **빨간색 → 파란색**으로 점진적으로 변화하면, 바닥이 멀어지며 펼쳐진 형태일 가능성이 높습니다.
    - 이미지 전체가 **균일한 파란색**이면, 전방에 **매우 가까운 장애물(예: 벽)**이 있을 수 있습니다.
    - **급격한 색상 변화**가 있는 경우, **복도를 가로막는 장애물**로 해석될 수 있습니다.
    - **부드럽고 일관된 색상 전환**은 **넓은 벽면이나 열린 구조물**일 가능성이 높습니다.

    행동 지침:
    - 이미지가 대부분 파란색으로 균일하면, **로봇이 벽 앞에 너무 가까이 접근해 있는 상황**일 수 있습니다.
    - 이 경우, **0.5m 정도 후진**하고 **다른 방향으로 회전하여 경로를 재탐색**하세요.

    당신의 역할:
    - 깊이 이미지 상의 색상 패턴과 깊이 범위를 바탕으로 **로봇 전방의 입체 구조를 설명**하세요.
    - **장애물로 추정되는 영역을 식별**하세요.
    - 색상과 거리 패턴을 종합해 **로봇의 안전한 다음 행동**을 제안하세요.
    """

    # 1) 깊이 데이터 읽어오기
    env = env_state.get_env()
    cam = env.scene[SceneEntityCfg("camera").name]
    image_np = cam.data.output["distance_to_image_plane"].cpu().numpy()
    depth_2d = np.squeeze(image_np)  # shape: (H, W)
    print("depth shape:", depth_2d.shape,
          "min/max:", depth_2d.min(), depth_2d.max())
    
    # 2) 0…255 uint8로 정규화
    z_min, z_max = 0, 8
    depth_clean = np.nan_to_num(depth_2d, nan=z_max, posinf=z_max, neginf=z_min)
    depth_clipped = np.clip(depth_clean, z_min, z_max)
    depth_uint8 = ((depth_clipped - z_min) / (z_max - z_min) * 255).astype(np.uint8)

    # 3) 컬러맵 적용 (JET)
    depth_color_bgr = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)

    # 4) BGR → RGB로 변환 후 PIL 이미지로
    depth_color_rgb = cv2.cvtColor(depth_color_bgr, cv2.COLOR_BGR2RGB)
    pil_color = Image.fromarray(depth_color_rgb)

    # 5) 저장
    save_dir = os.path.join("output", "depth_color")
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(save_dir, f"{timestamp}.png")
    pil_color.save(save_path)
    print(f"[컬러맵 이미지 저장]: {save_path}")

    # 6) VLM 전송을 위한 base64 인코딩
    buf = io.BytesIO()
    pil_color.save(buf, format="PNG")
    data_url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    SYSTEM_PROMPT = """
    당신은 로봇 주변의 깊이 정보를 해석하여 안전한 이동을 도와야 합니다.

    이 툴은 로봇이 주변 환경을 명확하게 인식하지 못하거나, 경로가 차단된 상황에서 사용됩니다.
    정확한 이미지를 확보하기 위해, **이미지를 촬영하기 전 반드시 로봇은 완전히 정지해야 합니다.**

    이 도구는 로봇 전방의 **깊이 이미지**를 처리합니다. 이 이미지는 색상으로 거리 정보를 시각화한 것이며, 다음과 같은 기준을 따릅니다:
    - **파란색**: 가까운 거리 (예: 0~2m)
    - **빨간색**: 먼 거리 (예: 8~10m)
    ※ 이 시스템은 '파란색 = 가까움', '빨간색 = 멀어짐'으로 해석해야 합니다.
    ※ 깊이 값은 **0m부터 최대 10m까지의 범위**를 갖습니다. 이 범위를 기준으로 물체까지의 거리를 추정할 수 있습니다.

    ⚠️ 주의: 파란색이라고 해서 항상 통과 가능한 공간은 아닙니다.
    - 예: 로봇 바로 앞에 있는 **가까운 벽**은 깊이 값이 작기 때문에 이미지 전체가 파란색으로 나타날 수 있으며, 이는 **열린 공간으로 착각할 위험이 있습니다.**

    해석 기준:
    - 이미지 상단에서 하단으로 **빨간색 → 파란색**으로 점진적으로 변화하면, 바닥이 멀어지며 펼쳐진 형태일 가능성이 높습니다.
    - 이미지 전체가 **균일한 파란색**이면, 전방에 **매우 가까운 장애물(예: 벽)**이 있을 수 있습니다.
    - **급격한 색상 변화**가 있는 경우, **복도를 가로막는 장애물**로 해석될 수 있습니다.
    - **부드럽고 일관된 색상 전환**은 **넓은 벽면이나 열린 구조물**일 가능성이 높습니다.

    행동 지침:
    - 이미지가 대부분 파란색으로 균일하면, **로봇이 벽 앞에 너무 가까이 접근해 있는 상황**일 수 있습니다.
    - 이 경우, **0.5m 정도 후진**하고 **다른 방향으로 회전하여 경로를 재탐색**하세요.

    당신의 역할:
    - 깊이 이미지 상의 색상 패턴과 깊이 범위를 바탕으로 **로봇 전방의 입체 구조를 설명**하세요.
    - **장애물로 추정되는 영역을 식별**하세요.
    - 색상과 거리 패턴을 종합해 **로봇의 안전한 다음 행동**을 제안하세요.
    """


    client = OpenAI()
    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "what's in this depth image?"},
                    {
                        "type": "input_image",
                        "image_url": data_url,
                        "detail": "low"
                    }
                ]
            }
        ]
    )
    
    print(f"[LLM 입력 이미지]: {save_path}")
    
    return Command(update={
        "messages": [
            ToolMessage(f"minimum depth: {depth_2d.min():.2f}." + response.output_text, tool_call_id=tool_call_id)
        ]
    })