# agent_module.py
import threading
import queue
import json
import torch
import time

from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_google_community import GoogleSearchAPIWrapper
from langchain.tools import tool
from langgraph.types import Command, interrupt
from langchain_core.tools import InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.runnables import RunnableConfig

from isaaclab.managers import SceneEntityCfg

from tools import *
import env_state


# --- 1) State 타입 정의 (messages 필드만 있으면 충분) ---
class State(TypedDict):
    messages: Annotated[list, add_messages]

# --- 2) 툴 정의 ---
@tool
def time_travel(tool_call_id: Annotated[str, InjectedToolCallId], step_index: int=None):
    """ 
    LangGraph의 상태(state) 히스토리에서 원하는 시점(step_index)으로 돌아갑니다.
    step_index가 None이면 전체 히스토리를 출력하고,
    값이 주어지면 해당 시점으로 이동합니다.
    """
    history = list(graph.get_state_history(config))
    if not history:
        return "No state history available."
    if step_index is None:
        for state in history:
            print("Num Messages: ", len(state.values["messages"]), "Next: ", state.next)
            print("-" * 80)
        return f"Available steps: 0 ~ {len(history) - 1}. Use time_travel(step_index) to replay."
    if step_index < 0 or step_index >= len(history):
        return f"Invalid step_index: {step_index}. Available steps: 0 ~ {len(history) - 1}."
    to_replay = history[step_index]
    # 반드시 해당 시점의 messages를 넣고, ToolMessage도 추가.
    replay_message = f"Replayed to step {step_index}."
    new_messages = to_replay.values["messages"] + [ToolMessage(replay_message, tool_call_id=tool_call_id)]
    return Command(update={"messages": new_messages})



@tool
def asynchronous_timer(
    timer: float,
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """
    🟢 **중단 없이 실행되는 동작을 위한 비차단형 타이머**입니다.

    이 타이머는 로봇이 비교적 긴 시간 동안 움직여야 할 때 사용되며,  
    `blocking_timer`와 달리 LLM의 판단 흐름을 멈추지 않고 유지할 수 있어  
    더 유연하고 부드러운 행동 계획이 가능합니다.

    주요 사용 사례:
    - 0.5초 이상 전진 또는 회전
    - 넓은 공간에서의 탐색 또는 주행
    - 중단 없이 진행되는 연속 동작 제어

    동작 방식:
    - 지정한 시간이 지나면 `asynchronous_timer_expired()` 함수가 자동으로 호출되어  
      LLM이 다음 행동을 생성할 수 있도록 흐름을 이어줍니다.

    참고 사항:
    - **예측 가능한 경로에서 LLM 호출 횟수를 줄이고자 할 때** 이상적입니다.
    - 특별히 정밀한 제어가 필요한 경우가 아니라면, 기본적으로 이 타이머를 사용하는 것을 권장합니다.
    """
    env = env_state.get_env()
    dt = env.sim.get_physics_dt()
    steps = int(timer / dt)

    def _wait_and_expire(step_count: int, call_id: str):
        start = env_state.get_timer_steps()
        target = start + step_count

        milestones = {1, 2, 5, 10, 20}
        while env_state.get_timer_steps() < target:
            time.sleep(dt)
            remain = target - env_state.get_timer_steps()
            if remain in milestones:
                print(f"[async 타이머 진행] asynchronous_timer가 {remain} steps 남았습니다.")
                milestones.remove(remain)

        # 만료 콜백 큐에 직접 삽입
        llm_command_queue.put("asynchronous_timer_expired()")

    # 백그라운드 스레드에서 대기 후 만료 처리
    threading.Thread(
        target=_wait_and_expire,
        args=(steps, tool_call_id),
        daemon=True
    ).start()

    return Command(update={
        "messages": [
            ToolMessage(f"asynchronous_timer started: {timer}s ({steps} steps)", tool_call_id=tool_call_id)
        ]
    })

@tool
def asynchronous_timer_expired(
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """
    비동기 타이머가 만료되었을 때 자동으로 호출되는 이벤트입니다.

    이 이벤트는 예약된 지연 시간이 끝났음을 시스템에 알리는 역할을 하며,  
    이를 통해 LLM은 이후 동작이나 판단을 생성할 수 있게 됩니다.

    사용 목적:
    - `asynchronous_timer`를 통해 설정된 대기 시간이 끝났음을 알림
    - 대기 후 이어지는 행동(예: 다음 이동, 정책 전환 등)을 실행 가능하게 함
    """
    llm_command_queue.put(
        "⏰ Timer expired! Considering past commands, suggest the next action."
    )
    return Command(update={
        "messages": [
            ToolMessage("asynchronous_timer_expired", tool_call_id=tool_call_id)
        ]
    })


@tool
def blocking_timer(
    timer: float,
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """
    🔴 최대 0.5초 이하의 짧은 지연을 위한 **정밀 차단형 타이머**입니다.

    이 타이머는 매우 짧은 시간 동안 LLM의 흐름을 일시적으로 멈추고,  
    **정확한 타이밍 제어**(예: 잠깐 회전하거나 잠시 멈추는 동작)에 사용됩니다.

    Args:
        timer (float): 지연 시간 (단위: 초).  
        ⏱️ 0.01초 ~ 최대 0.5초 사이의 값만 허용됩니다.  
        예: `0.2`는 200밀리초 지연을 의미합니다.

    사용 예시:
    - 0.3초간 제자리 회전
    - 속도를 잠시 유지한 후 멈추기
    - 연속 동작 사이의 타이밍 동기화

    ⚠️ 주의:
    - **0.5초를 초과하는 지연에는 사용하지 마세요.**
      긴 이동(예: 1미터 걷기 등)에는 `asynchronous_timer`를 사용해야 합니다.
    - `blocking_timer`를 과도하게 사용하면 LLM 응답 지연이나 성능 저하가 발생할 수 있습니다.

    권장 용도:
    - 빠르고 결정적인 시간 제어가 필요한 경우에만 사용
    - LLM 흐름이 일시 정지되어도 괜찮은 상황
    """
    env = env_state.get_env()
    dt = env.sim.get_physics_dt()
    steps = int(timer / dt)

    start = env_state.get_timer_steps()
    target = start + steps

    # Block until simulation timer reaches zero
    milestones = {1, 2, 5, 10, 20}
    while env_state.get_timer_steps() < target:
        time.sleep(dt)
        remain = target - env_state.get_timer_steps()
        if remain in milestones:
            print(f"[blocking 타이머 진행] blocking_timer가 {remain} steps 남았습니다.")
            milestones.remove(remain)

    print("[blocking 타이머 만료] blocking_timer가 종료되었습니다.")

    return Command(update={
        "messages": [
            ToolMessage(f"blocking_timer {timer}s ({steps} steps) complete", tool_call_id=tool_call_id)
        ]
    })

# 3) 그래프 빌더 및 ToolNode 생성
# --- 툴 목록에 모든 도구 추가 ---
tools = [
    google_search_tool,
    human_assistance,
    time_travel,
    set_velocity,
    proprioception,
    switch_policy,
    process_rgb,
    process_depth,
    asynchronous_timer,
    asynchronous_timer_expired,
    blocking_timer,
]
tool_node = ToolNode(tools=tools)

# --- LLM 초기화 ---
llm = init_chat_model("openai:gpt-4.1")  # gpt 4.1 mini는 툴 설명이나 프롬프트를 빨리 잊어버리는 경향이 심하다.
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """
너는 맵이 주어지지 않은 낯선 환경에서 모바일 로봇(사족보행 로봇)을 제어하는 AI야. 목표는 주어진 명령을 수행하면서 장애물에 부딪히지 않고 안전하게 이동하는 것이야.
주변환경을 파악할 때 깊이 이미지를 우선적으로 사용하고, RGB 이미지는 보완적으로 사용해. 둘 다 동시에 사용하는 것은 딜레이가 발생하므로 지양해.

다음과 같은 원칙을 반드시 지켜야 해:

1. **로봇의 움직임은 항상 점진적으로 조절**해야 해. 속도를 빠르게 바꾸면 로봇이 불안정해지므로, 속도는 천천히 올리고 천천히 줄이도록 해.
    - 선속도의 경우는 0.5m/s를 0.2초 정도 유지하고 속도를 1m/s로 올리거나 반대의 경우 0.2초 정도 유지하고 0.0m/s로 줄이는 것이 좋음
    - 회전 속도의 경우는 0.5rad/s를 0.2초 정도 유지하고 1rad/s로 올리거나 반대의 경우 0.2초 정도 유지하고 0.0rad/s로 줄이는 것이 좋음

2. **로봇의 자세(Proprioception)**를 인식하는 것이 중요해. 자세와 접촉 정보, 관절 상태 등을 활용해서 로봇이 균형을 유지하고 있는지, 안정적인지 판단해야 해.

3. **장애물에 가까워지면 속도를 줄여야 해.** 충돌을 방지하기 위해 환경 분석 결과를 기반으로 위험한 상황에서는 멈추거나 천천히 이동하도록 해.

4. **카메라 시야로 주변이 잘 보이지 않으면**, 주변을 살펴야 해. 좌우 또는 뒤로 천천히 회전하면서 막다른 길이나 막힌 통로에서 벗어날 수 있는 방향을 탐색해.

5. **정확한 회전이 어려우므로**, 회전은 일정한 속도와 시간으로 수행하고, 그 결과를 기억해서 스스로 얼마나 회전했는지 추정하고 조정해야 해. 예를 들어:
   - `wz = 1.0` 속도로 0.3초 회전
   - `wz = 0.5` 속도로 0.5초 회전
   이처럼 회전의 결과를 추정하고 누적해서 기억해. 그래야 더 나은 회전 제어가 가능해질 거야.

6. 로봇의 크기는 **최대 높이 0.4m**, **너비 0.31m**야. 이 크기를 고려해서 좁은 통로 진입이나 장애물 사이 통과 여부를 판단해.

7. **한 번에 이동하는 거리는 앞에 복도가 아닌 경우 반드시 짧게 계획해야 해.** 
   - 지형에 대한 사전정보가 없는 경우, 로봇은 한 번에 **0.2~0.3초**, 또는 **0.2~0.3m 이하**로만 이동해야 해.
   - 이렇게 작은 단위로 조금씩 이동하면서 **자주 판단**하고 상황을 다시 분석하는 것이 훨씬 안전하고 유리해.
   - 연속된 판단을 통해 미지의 환경을 탐색하고, 적절한 타이밍에 회전하거나 정지할 수 있도록 계획하라.
   - 앞에 길게 이어진 복도가 있는 경우, 로봇은 한 번에 **0.5초 이상**, 또는 **0.5m 이상** 이동할 수 있다.

8. **깊이 이미지로부터 장애물에 대한 대략적인 거리를 얻은 경우, 조금 더 먼 거리를 계획할 수 있다.**
    - 정면의 장애물이 5m 정도라고 추정이 된다면, 3~4m 정도 간 다음 다시 판단하여 정면의 장애물과 얼마나 가까워졌는지 판단하면 이전의 추정치가 명확함을 알 수 있다.
    - 장애물까지의 거리를 기반으로 정면에 복도가 있다고 생각된다면, 약간 몸을 회전하면서 이동하여 모서리에 도달했을 때 복도를 볼 수 있도록 자세를 조정하라.

너는 이러한 원칙을 지키면서, 제공된 도구(tool)를 적절히 활용해 목표를 달성해야 해.
"""



def chatbot(state: State):
    """챗봇 노드: 이전 메시지를 받아 LLM 응답을 생성합니다."""
    msgs = [SYSTEM_PROMPT] + state["messages"]
    response = llm_with_tools.invoke(msgs)
    return {"messages": [response]}

# --- 그래프 빌더 설정 ---
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)


def route_tools(state: State):
    """
    마지막 메시지에 tool_calls가 있으면 'tools' 노드로,
    아니면 END로 라우팅합니다.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"tool_edge 입력 state에 메시지가 없습니다: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools": "tools", END: END},
)

graph_builder.add_edge("tools", "chatbot")


# --- 메모리 및 그래프 컴파일 ---
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
config = {
    "configurable": {"thread_id": "1"},
    "recursion_limit": 250
}

# === [대화 스트림 함수 및 메인 루프] ===
llm_response_queue = queue.Queue()
last_velocity = None
def stream_graph_updates(env, user_input: str):
    """유저 입력을 받아 LangGraph로 대화 스트림을 출력합니다."""
    global last_velocity
    if isinstance(user_input, Command):
        events = graph.stream(user_input, config, stream_mode="values")
    else:
        events = graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config,
            stream_mode="values",
        )

    for event in events:
        # human-in-the-loop 인터럽트 발생 시
        if "__interrupt__" in event:
            # interrupt()에서 넘어온 질문(prompt)를 꺼내서
            prompt = event["__interrupt__"][0].value
            answer = input(f"{prompt}")
            return stream_graph_updates(Command(resume={"data": answer}))
        if "messages" not in event:
            continue

        # ToolMessage 중 set_velocity만 파싱
        for msg in event["messages"]:
            if hasattr(msg, "name") and msg.name == "set_velocity":
                try:
                    data = json.loads(msg.content)
                    vx, vy, wz = data["velocity"]["vx"], data["velocity"]["vy"], data["velocity"]["wz"]
                    # print(f"[LLM output] vx={vx}, vy={vy}, wz={wz}")  # LLM 명령 출력
                    vel = torch.tensor([[vx, vy, wz]], device=env.device)
                    last_velocity = vel.repeat(env.num_envs, 1)
                except json.JSONDecodeError as e:
                    print(f"[WARNING] set_velocity 파싱 실패: {e}")
        
        # 일반적인 대화 스트림 처리
        event["messages"][-1].pretty_print()

def llm_worker(prompt: str, env):
    """
    별도 스레드에서 LLM(stream_graph_updates)을 호출하고,
    끝나면 응답 큐에 True를 넣어줍니다.
    """
    stream_graph_updates(env, prompt)
    llm_response_queue.put(True)

# — 4. 입력 큐 & 스레드 —
llm_command_queue = queue.Queue()

def start_input_listener():
    def _listen():
        while True:
            cmd = input().strip()  # "[시스템]로봇 명령 입력: "
            if cmd:
                llm_command_queue.put(cmd)
    threading.Thread(target=_listen, daemon=True).start()


# — 5. 시뮬레이션 콜백 함수 —
def llm_command_callback(env):
    """
    env.num_envs, env.device를 가진 시뮬레이션 환경에서
    매 스텝 호출하여 torch.Tensor 속도 명령을 돌려줍니다.
    """
    global last_velocity
    env_state.set_env(env)

    # 1) 시뮬레이션 스텝 카운트
    steps = env_state.get_timer_steps()
    env_state.set_timer(steps + 1)

    # 2) 입력 큐에서 새 프롬프트 꺼내기
    try:
        prompt = llm_command_queue.get_nowait()
    except queue.Empty:
        prompt = None

    # 3) 새 프롬프트가 있으면, 워커 스레드로 LLM 호출 분리
    if prompt:
        if prompt.lower() in ["quit", "exit", "q", "종료"]:
            print("Goodbye!")
            exit()

        threading.Thread(
            target=llm_worker,
            args=(prompt, env),
            daemon=True
        ).start()

    # 4) 응답 큐 확인: 워커가 끝났으면 last_velocity는 이미 갱신된 상태
    try:
        done = llm_response_queue.get_nowait()
        if done:
            # (필요시) 완료 로그나 추가 처리
            # print("\n[시스템] LLM 명령 실행 완료\n")
            pass
    except queue.Empty:
        pass

    if last_velocity is None:
        return torch.zeros((env.num_envs, 3), device=env.device)

    return last_velocity
