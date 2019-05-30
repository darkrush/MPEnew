import numpy as np
from MPE.core import World, Agent, Landmark,  Fence
from MPE.scenario import BaseScenario
from MPE.utils import *
import math


class properties_file(object):
    def __init__(self):
        self.agent_list = (
            {'name' : 'agent 0', 'collide' : True, 'silent' : True, 'size' : 0.15,'theta' : 0, 'pos' : [-2.0,-0.3], 'color':[1, 0.25, 0.25],},
        )
        self.landmark_list = (
            {'name' : 'landmark 0', 'collide' : False, 'movable' : False, 'color':[1, 0.25, 0.25], 'pos' : [0.3,2.0]},
        )
        self.fence_list = (
            {'anchor' : [0,0], 'rotation' : math.pi*0 , 'vertices' : ([0.0,0.0],[0.0,2.2],[0.6,2.2],[0.6,-0.6],[-2.2,-0.6],[-2.2,0]), 'close' : True, 'filled': False, 'color' : [0, 0, 0]},
        )

proper = properties_file()


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        world.N = 1

        world.agents = []
        for prop_dict in proper.agent_list:
            entity = Agent()
            entity.name = prop_dict['name']
            entity.collide = prop_dict['collide']
            entity.silent = prop_dict['silent']
            entity.size = prop_dict['size']
            entity.color = np.array(prop_dict['color'])
            entity.state.p_pos = np.array(prop_dict['pos'])
            entity.state.theta = prop_dict['theta']
            entity.state.p_vel = np.zeros(world.dim_p)
            entity.state.c = np.zeros(world.dim_c)
            world.agents.append(entity)

        world.landmarks = []
        for prop_dict in proper.landmark_list:
            entity = Landmark()
            entity.name = prop_dict['name']
            entity.collide = prop_dict['collide']
            entity.movable = prop_dict['movable']
            entity.color = np.array(prop_dict['color'])
            entity.state.p_pos = np.array(prop_dict['pos'])
            entity.state.p_vel = np.zeros(world.dim_p)
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
            world.agents[idx].state.p_pos = np.array(proper.agent_list[idx]['pos'])
            world.agents[idx].state.theta = proper.agent_list[idx]['theta']            
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
