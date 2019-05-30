import numpy as np
from MPE.core import World, Agent, Landmark,  Fence
from MPE.scenario import BaseScenario
from MPE.utils import *
import math


class properties_file(object):
    def __init__(self):
        self.fence_list = (
            #{'anchor' : [0,0], 'rotation' : math.pi*0 , 'vertices' : ([0.0,0.0],[0.0,1.2],[0.6,1.2],[0.6,-0.6],[-1.2,-0.6],[-1.2,0]), 'close' : True, 'filled': False, 'color' : [0, 0, 0]},
            #{'anchor' : [0,0], 'rotation' : math.pi*0 , 'vertices' : ([-2.2,-2.0],[2.2,-2.0],[2.2,2.0],[-2.2,2.0]), 'close' : True, 'filled': False, 'color' : [0, 0, 0]},
        )

proper = properties_file()


class Scenario(BaseScenario):
    def make_world(self, N = 2):
        world = World()
        # set any world properties first
        world.dim_c = 2
        world.N = N

        world.agents = []
        #for idx,prop_dict in enumerate(proper.agent_list):
        for idx in range(N):
            entity = Agent()
            entity.i = idx
            entity.name = 'agent %d'%idx
            entity.collide = True
            entity.silent = True
            entity.size = 0.15
            entity.color = hsv2rgb(360.0/N*idx,1.0,1.0)
            world.agents.append(entity)
        
        world.landmarks = []
        #for prop_dict in proper.landmark_list:
        for idx in range(N):
            entity = Landmark()
            entity.name = 'landmark %d'%idx
            entity.collide = False
            entity.movable = False
            entity.color = hsv2rgb(360.0/N*idx,1.0,1.0)
            world.landmarks.append(entity)

        world.fences = []
        for prop_dict in proper.fence_list:
            entity = Fence()
            entity.anchor = prop_dict['anchor']
            entity.rotation = prop_dict['rotation']
            entity.vertices = prop_dict['vertices']
            entity.close = prop_dict['close']
            entity.filled = prop_dict['filled']
            entity.color = np.array(prop_dict['color'])
            entity.calc_vertices()
            world.fences.append(entity)

        self.reset_world(world)
        return world

    def reset_world(self, world,if_eval = False):
        # random properties for agents
        #for idx,prop_dict in enumerate(proper.agent_list) :
        
        for idx in range(world.N):
            if if_eval:
                theta = idx*math.pi*2/world.N
                x = 2*np.cos(theta)
                y = 2*np.sin(theta)
                world.agents[idx].state.p_pos = np.array([x,y])
                world.agents[idx].state.theta = theta+math.pi
                world.landmarks[idx].state.p_pos = -np.array([x,y])
            else:
                world.agents[idx].state.p_pos = np.array([np.random.uniform(-2,2),np.random.uniform(-2,2)])
                world.agents[idx].state.theta = np.random.uniform(0,math.pi*2)
                world.landmarks[idx].state.p_pos = np.array([np.random.uniform(-2,2),np.random.uniform(-2,2)])
            world.agents[idx].state.p_vel = np.zeros(world.dim_p)
            world.agents[idx].state.c = np.zeros(world.dim_c)
            world.agents[idx].state.crash = False
            world.agents[idx].state.reach = False
            world.landmarks[idx].state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world, world_before):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        agent_idx = agent.i
        a_before, l_before, a, l = (world_before.agents[agent_idx], world_before.landmarks[agent_idx], world.agents[agent_idx], world.landmarks[agent_idx])

        dist_before = np.linalg.norm(a_before.state.p_pos - l_before.state.p_pos)
        dist        = np.linalg.norm(a.state.p_pos - l.state.p_pos)
        rew += 10*(dist_before - dist)
        rew -=0.1
        if(dist < agent.size):
            rew += 15

        if agent.state.crash:
            rew -= 15
        return rew
 
    def done(self, agent, world):
        done = agent.state.crash
        agent_idx = agent.i
        dist = np.linalg.norm(world.agents[agent_idx].state.p_pos - world.landmarks[agent_idx].state.p_pos)
        if(dist < agent.size):
            done = True
        return done
    
    def observation(self, agent, world):
        l_laser_min = np.array([agent.r_laser]*agent.dim_laser)
        for agent_i  in world.agents:
            if agent_i is agent:
                continue
            l_laser = laser_agent_agent(agent,agent_i)
            l_laser_min = np.min(np.vstack([l_laser_min,l_laser]),axis = 0)
        for fence in world.fences:
            l_laser = laser_agent_fence(agent,fence)
            l_laser_min = np.min(np.vstack([l_laser_min,l_laser]),axis = 0)
        agent.state.laser_state = l_laser_min
        return np.hstack([agent.state.p_pos,agent.state.theta,world.landmarks[agent.i].state.p_pos,l_laser_min])