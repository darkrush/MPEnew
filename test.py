from MPE.environment import MultiAgentEnv
import MPE.scenarios as scenarios
from  MPE.time_limit import TimeLimit
from MPE.make_env import make_env
import time
import numpy as np

#scenario = scenarios.load('single_agent_maze' + ".py").Scenario()
#N = 1
#env = TimeLimit(env = MultiAgentEnv(scenario.make_world(), scenario.reset_world, scenario.reward, scenario.observation,None,scenario.done),max_episode_steps=300)
N = 1
env = make_env('MAC_%d'%N)
obs = env.reset(if_eval = True)
action_space = env.action_space
print(env.observation_space)
print(env.action_space)
step_nb = 1000
start = time.time()
for i in range(step_nb):
    env.render(mode="human",close=False)
    act = []
    for i in range(N):
        act.append(np.array([0.9+0.0*np.random.randn(),0.00]))
    #act[-1] = np.array([0.2,0.5])
    obs_n, reward_n, done_n, info_n = env.step(act)
    print(obs_n)
    print(reward_n)
    print(done_n)
    print(info_n)
    all_done = True
    #print(done_n)
    for done in done_n:
        all_done &= done
    if all_done or info_n['TimeLimit.truncated'] :
        obs = env.reset(if_eval = True)

end = time.time()
print(step_nb/(end-start))
    
