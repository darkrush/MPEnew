from MPE.environment import MultiAgentEnv
import MPE.scenarios as scenarios
from  MPE.time_limit import TimeLimit

def make_env(EnvName, timelimit = 300):
    Namelist = EnvName.split('_')
    scenario = scenarios.load(Namelist[0] + ".py").Scenario()
    N = 1
    potential = 10
    if len(Namelist)>1:    
        N=int(Namelist[1])
    if len(Namelist)>2:    
        potential=float(Namelist[2])
    if Namelist[0] == 'MAC':
        world = scenario.make_world(N = N,potential_reward = potential)
    else:
        world = scenario.make_world()

    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,scenario.info,scenario.done)
    if timelimit>0:
        env = TimeLimit(env,max_episode_steps=timelimit)
    return env


