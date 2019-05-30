from MPE.environment import MultiAgentEnv
import MPE.scenarios as scenarios
from  MPE.time_limit import TimeLimit
from MPE.make_env import make_env
import time
import numpy as np

#scenario = scenarios.load('single_agent_maze' + ".py").Scenario()
#N = 1
#env = TimeLimit(env = MultiAgentEnv(scenario.make_world(), scenario.reset_world, scenario.reward, scenario.observation,None,scenario.done),max_episode_steps=300)
N = 2
env = make_env('MAC_%d'%N)
obs = env.reset(if_eval = False)
action_space = env.action_space
step_nb = 2000
start = time.time()
for i in range(step_nb):
    env.render(mode="human",close=False)
    act = []
    for i in range(N):
        act.append(np.array([0.2+0.5*np.random.randn(),0.0]))
    obs_n, reward_n, done_n, info_n = env.step(act)
    all_done = True
    for done in done_n:
        all_done &= done
    if all_done or info_n['TimeLimit.truncated'] :
        obs = env.reset(if_eval = False)
end = time.time()
print(step_nb/(end-start))
    
