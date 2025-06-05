# env_state.py
"""
시뮬레이션 env 객체를 전역으로 저장/공유하기 위한 모듈
"""
import time


env = None

def set_env(e):
    global env
    env = e

def get_env():
    return env


policy_index = 0

def set_policy_index(index):
    global policy_index
    policy_index = index

def get_policy_index():
    return policy_index

timer_steps = 0
def set_timer(steps):
    global timer_steps
    timer_steps = steps


def get_timer_steps():
    global timer_steps
    return timer_steps
