import numpy as np
import seaborn as sns
import math

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity linar velocity angular velocity
        self.p_vel = None
        # axis
        self.theta = 0
        # Deflection angle of front wheel
        self.defle = 0


# state of agents (including internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # crash into sth
        self.crash = False
        # crash into sth
        self.reach = False

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None

# properties of wall entities
class Fence(object):
    def __init__(self, anchor=[0,0], rotation = 0, vertices=([0,0],), close = False, filled = False, color = [0.0, 0.0, 0.0]):
        # the anchor point in global coordinate
        self.anchor = anchor
        # the rotation angle by radian global coordinate
        self.rotation = rotation
        # the coordinate of vertices related to anchor
        self.vertices = vertices
        # Fill the fence by color inside if True
        self.filled = filled
        # A close fence means a fence between vertices[-1] and vertices[0], forced to be True if filled 
        self.close = close or filled
        # color
        self.color = color

        self.calc_vertices()

    def calc_vertices(self):
        self.global_vertices = []
        for v in self.vertices:
            c = np.cos(self.rotation)
            s = np.sin(self.rotation)
            g_v_x = v[0]*c - v[1]*s +self.anchor[0]
            g_v_y = v[1]*c + v[0]*s +self.anchor[1]
            self.global_vertices.append(np.array([g_v_x,g_v_y]))
        if self.close:
            self.global_vertices.append(self.global_vertices[0])

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # index among all entities (important to set for distance caching)
        self.i = 0
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # color
        self.color = None
        # max speed
        self.max_linear_speed = 1.0
        self.min_angular_speed = 0.0001
        # min radius
        self.max_angular_speed = 1.0
        # accel
        self.accel = None
        # state: including internal/mental state p_pos, p_vel
        self.state = EntityState()


# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agent are adversary
        self.adversary = False
        # agent are dummy
        self.dummy = False
        # agents are movable by default
        self.movable = True
        # control range
        self.linear_gain = 1.0
        self.angle_gain = math.pi/6.0
        #distance between front wheel and back wheel
        self.car_length = 0.1
        # state: including internal/mental state p_pos, p_vel
        self.state = AgentState()
        # action: physical action u 
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        # dim of laser data
        self.dim_laser = 32
        # range of laser data
        self.r_laser = 1.0

        self.laser_state = np.array([self.r_laser]*self.dim_laser)

def check_AA_collisions(agent_a,agent_b):
    min_dist = agent_a.size + agent_b.size
    ab_dist = np.linalg.norm(agent_a.state.p_pos - agent_b.state.p_pos)
    return ab_dist<=min_dist

def check_AF_collisions(agent,fence):
    r = agent.size
    o_pos =  agent.state.p_pos
    for i in range(len(fence.global_vertices)-1):
        a_pos = fence.global_vertices[i]
        v_oa = a_pos - o_pos
        av_dist = np.linalg.norm(v_oa)
        #crash if a vertex inside agent
        if av_dist<=r:
            return True
        b_pos = fence.global_vertices[i+1]
        v_ab = b_pos - a_pos
        v_ob = b_pos - o_pos
        #crash  if two vertex in different sides of perpendicular and d(o,ab)<r
        if (np.dot(v_oa,v_ab)<0) and (np.dot(v_ob,-v_ab)<0):
            dist_o_ab = np.abs(np.cross(v_oa,v_ab)/np.linalg.norm(v_ab))
            if dist_o_ab <= r:
                return True
    return False


# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        self.fences = []
        # position dimensionality
        self.dim_p = 2
        # simulation timestep
        self.dt = 0.1
        self.N = 1

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]
    
    def assign_agent_colors(self):
        n_dummies = 0
        if hasattr(self.agents[0], 'dummy'):
            n_dummies = len([a for a in self.agents if a.dummy])
        n_adversaries = 0
        if hasattr(self.agents[0], 'adversary'):
            n_adversaries = len([a for a in self.agents if a.adversary])
        n_good_agents = len(self.agents) - n_adversaries - n_dummies
        dummy_colors = [(0, 0, 0)] * n_dummies
        adv_colors = [(0.75, 0.25, 0.25)] * n_adversaries #sns.color_palette("OrRd_d", n_adversaries)
        good_colors = [(0.25, 0.25, 0.75)] * n_good_agents#sns.color_palette("GnBu_d", n_good_agents)
        colors = dummy_colors + adv_colors + good_colors
        for color, agent in zip(colors, self.agents):
            agent.color = color
    
    # landmark color
    def assign_landmark_colors(self):
        for landmark in self.landmarks:
            landmark.color = np.array([0.25, 0.25, 0.25])
    
    # update state of the world
    def step(self):
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)

        # gather forces applied to entities
        p_u = [None] * len(self.entities)
        # apply agent physical controls
        p_u = self.apply_action_u(p_u)

        # integrate physical state
        self.integrate_state(p_u)
        self.check_collisions()
    
    def check_collisions(self):
        for ia, agent_a in enumerate(self.agents):
            if agent_a.state.crash :
                continue
            for ib in range(len(self.agents)):
                if ia==ib :
                    continue
                agent_b = self.agents[ib]
                crash = check_AA_collisions(agent_a,agent_b)
                agent_a.state.crash = crash
                #agent_b.state.crash = crash
                if agent_a.state.crash :
                    break
            if agent_a.state.crash :
                continue
            for fence in self.fences:
                agent_a.state.crash = check_AF_collisions(agent_a,fence)
                if agent_a.state.crash :
                    break
        

    # gather agent action forces
    def apply_action_u(self, p_u):
        # set applied forces
        for i, agent in enumerate(self.agents):
            if agent.movable:
                p_u[i] = np.clip(agent.action.u, -1.0, 1.0)
        return p_u

    # integrate physical state
    def integrate_state(self, p_u):
        for i, agent in enumerate(self.agents):
            if not agent.movable: continue
            if agent.state.crash: continue
            if (p_u[i] is not None):
                agent.state.p_vel[0] = p_u[i][0]*agent.linear_gain
                agent.state.p_vel[1] = agent.state.p_vel[0]*math.tan(agent.angle_gain*p_u[i][1])/agent.car_length

            #calculate radius
            if ( abs(agent.state.p_vel[1])< agent.min_angular_speed ):
                theta_temp = agent.state.p_vel[1] * self.dt
                x = agent.state.p_vel[0] * self.dt*(1-theta_temp**2/6.0)
                y = agent.state.p_vel[0] * self.dt*(theta_temp/2.0)
                
            else:
                r = agent.state.p_vel[0] / (agent.state.p_vel[1])
                theta_temp = agent.state.p_vel[1] * self.dt
                x = r * math.sin(theta_temp)
                y = r * (1 - math.cos(theta_temp))

            agent.state.defle = agent.angle_gain*p_u[i][1]
            agent.state.p_pos[0] += x * math.cos(-agent.state.theta) + y * math.sin(-agent.state.theta)
            agent.state.p_pos[1] += y * math.cos(-agent.state.theta) - x * math.sin(-agent.state.theta)

            agent.state.theta += theta_temp
            if (agent.state.theta > (2 * math.pi)):
                agent.state.theta -= 2 * math.pi
            if (agent.state.theta < 0):
                agent.state.theta += 2 * math.pi
