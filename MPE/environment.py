import numpy as np
import copy
import math

from  gym import Env
from gym import spaces

# update bounds to center around agent
cam_range = 4

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, post_step_callback=None,
                 shared_viewer=True, discrete_action=False):

        self.world = world
        self.world_before = world
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        self.post_step_callback = post_step_callback

        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        #self.shared_reward = False
        self.time = 0
        self.time_limit = 100
        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # physical action space
            u_action_space = spaces.Box(low=-1, high=+1, shape=(world.dim_p,), dtype=np.float32)# [-1,1]
            if agent.movable:
                total_action_space.append(u_action_space)
            self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))#[-inf,inf]

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def _seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    # step  this is  env.step()
    #@profile
    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents
        self.world_before = copy.deepcopy(self.world)
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step() # core.step()  
        # record observation for each agent
        self.time += 1
        
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent) or self.time>= self.time_limit )
            info_n['n'].append(self._get_info(agent))

        # all agents get total reward in cooperative case, if shared reward, all agents have the same reward, and reward is sum
        reward = np.sum(reward_n)  
        if self.shared_reward:
            reward_n = [reward] * self.n

        if self.post_step_callback is not None:
            self.post_step_callback(self.world)
        
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        self.time = 0
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world, self.world_before)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(2)
        # process action
        if agent.movable:
            # physical action
            agent.action.u = action


    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, mode='human', close=True):
        if close:
            # close any existic renderers
            for i,viewer in enumerate(self.viewers):
                if viewer is not None:
                    viewer.close()
                self.viewers[i] = None
            return []

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)   
                from . import rendering            
                self.viewers[i] = rendering.Viewer(700,700)
 

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            from . import rendering
            self.render_geoms = []
            self.render_geoms_xform = []         
            
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()                
                if 'agent' in entity.name:          
                    geom.set_color(*entity.color, alpha=0.5)

                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)                

            for fence in self.world.fences:
                if fence.filled :
                    geom = rendering.make_polygon(fence.global_vertices)
                else:
                    geom = rendering.make_polyline(fence.global_vertices)
                geom.set_color(*fence.color)
                self.render_geoms.append(geom)

            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        
        results = []
        for i in range(len(self.viewers)):
            from . import rendering
            
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0]-cam_range, pos[0]+cam_range, pos[1]-cam_range, pos[1]+cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)

                if 'agent' in entity.name:
                    self.render_geoms[e].set_color(*entity.color, alpha=0.5)
                    N = entity.dim_laser
                    theta = entity.state.theta
                    start = (0, 0)
                    for idx_laser in range(N):
                        d = entity.state.laser_state[idx_laser]
                        theta_i = theta+idx_laser*math.pi*2/N
                        end = (math.cos(theta_i)*d, math.sin(theta_i)*d)
                        arrow = self.viewers[i].draw_line(start = start, end = end,color = [0.0,1.0,0.0])
                        offset = rendering.Transform()
                        offset.set_translation(*entity.state.p_pos)
                        arrow.add_attr(offset)

                    if entity.movable:
                        radius = entity.size
                        theta = entity.state.theta
                        start = (0, 0)
                        end = (-math.cos(theta)*radius, -math.sin(theta)*radius)
                        arrow = self.viewers[i].draw_line(start = start, end = end,color = [0.0,0.0,0.0])
                        offset = rendering.Transform()
                        offset.set_translation(*entity.state.p_pos)
                        arrow.add_attr(offset)
                        
                        theta+=entity.state.defle
                        end = (math.cos(theta)*radius, math.sin(theta)*radius)
                        arrow = self.viewers[i].draw_line(start = start, end = end,color = [1.0,0.0,0.0])
                        offset = rendering.Transform()
                        offset.set_translation(*entity.state.p_pos)
                        arrow.add_attr(offset)
                else:
                    self.render_geoms[e].set_color(*entity.color)

            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))

        return results
