import math
import numpy as np

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
    #r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return np.array((r, g, b))

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
