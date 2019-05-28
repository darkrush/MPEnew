import numpy as np
from MPE.core import World, Agent, Landmark,  Fence
from MPE.scenario import BaseScenario
import math

def hsv2rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return np.array((r, g, b))


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

def laser_agent_agent(agent,agent_i):
    R = agent.r_laser
    N = agent.dim_laser
    l_laser = np.array([R]*N)
    o_pos =  agent.state.p_pos
    theta = agent.state.theta
    oi_pos = agent_i.state.p_pos
    v_oio = o_pos-oi_pos
    l= np.linalg.norm(v_oio)
    r = agent_i.size
    if l>R+r:
        return l_laser

    for idx_laser in range(N):
        theta_i = theta+idx_laser*math.pi*2/N
        c_x = R*np.cos(theta_i)
        c_y = R*np.sin(theta_i)            
        c_pos = np.array([c_x,c_y])+o_pos
        v_oic = c_pos-oi_pos
        v_oc = c_pos-o_pos
        dist_oi_oc = np.abs(np.cross(v_oio,v_oic)/R)
        if dist_oi_oc > r:
            continue
        a = np.dot(-v_oio,v_oc)
        delta = a**2-R**2*(l**2-r**2)
        if delta<0:
            continue
        laser = (a-np.sqrt(delta))/R
        if laser<0:
            continue
        l_laser[idx_laser] = laser
    return l_laser

def laser_agent_fence(agent,fence):
    R = agent.r_laser
    N = agent.dim_laser
    l_laser = np.array([R]*N)
    o_pos =  agent.state.p_pos
    theta = agent.state.theta
    for i in range(len(fence.global_vertices)-1):
        a_pos = fence.global_vertices[i]
        b_pos = fence.global_vertices[i+1]
        oxaddR = o_pos[0]+R
        oxsubR = o_pos[0]-R
        oyaddR = o_pos[1]+R
        oysubR = o_pos[1]-R
        if oxaddR<a_pos[0] and  oxaddR<b_pos[0]:
            continue
        if oxsubR>a_pos[0] and  oxsubR>b_pos[0]:
            continue
        if oyaddR<a_pos[1] and  oyaddR<b_pos[1]:
            continue
        if oysubR>a_pos[1] and  oysubR>b_pos[1]:
            continue

        v_oa = a_pos - o_pos
        v_ob = b_pos - o_pos
        v_ab = b_pos - a_pos
        dist_o_ab = np.abs(np.cross(v_oa,v_ab)/np.linalg.norm(v_ab))
        #if distance(o,ab) > R, laser signal changes
        if dist_o_ab > R:
            continue
        S1 = np.cross(v_ab,-v_oa)
        aa = np.dot(v_oa,v_oa)
        bb = np.dot(v_ob,v_ob)
        ab = np.dot(v_oa,v_ob)
        numerator = ab*ab-aa*bb
        #v_ab_normal = np.array([v_ab[1],-v_ab[0]])
        #if np.dot(v_oa,v_ab_normal) < 0:
        #    v_ab_normal = -v_ab_normal
        max_adot = 0
        max_aid = 0
        max_bdot = 0
        max_bid = 0
        for idx_laser in range(N):
            theta_i = theta+idx_laser*math.pi*2/N
            c_x = R*np.cos(theta_i)
            c_y = R*np.sin(theta_i)
            v_oc = np.array([c_x,c_y])
            adot = np.dot(v_oc,v_oa)
            bdot = np.dot(v_oc,v_ob)
            if adot > max_adot:
                max_adot = adot
                max_aid = idx_laser
            if bdot > max_bdot:
                max_bdot = bdot
                max_bid = idx_laser
        if max_aid > max_bid:
            max_bid,max_aid = (max_aid,max_bid)
        if max_bid - max_aid > N//2:
            max_bid,max_aid = (max_aid,max_bid)
            max_bid+=N
            
        #range1 = N//2
        #range2 = N - range1
        for idx_laser in range(max_aid,max_bid+1):
            idx_laser %= N
            theta_i = theta+ idx_laser*math.pi*2/N
            c_x = R*np.cos(theta_i)
            c_y = R*np.sin(theta_i)            
            c_pos = np.array([c_x,c_y])+o_pos
            v_ac = c_pos - a_pos
            S2 = np.cross(v_ab,v_ac)
            if S1*S2>0:
                continue
            v_oc = c_pos - o_pos
            if np.cross(v_oc,v_oa)*np.cross(v_oc,v_ob) >0:
                continue
            cb = np.dot(v_oc,v_ob)
            ca = np.dot(v_oc,v_oa)
            denominator = (ab-bb)*ca-(aa-ab)*cb
            d = abs(numerator/denominator*np.linalg.norm(v_oc))
            l_laser[idx_laser] = min(l_laser[idx_laser],d)
    return l_laser
