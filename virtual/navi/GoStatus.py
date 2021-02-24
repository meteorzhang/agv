import numpy as np


class AgentState(object):
    def __init__(self, px, py, vx, vy, gx, gy, radius, v_mean, shortest_path_action, shortest_path_length, id, state):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.gx = gx
        self.gy = gy
        self.radius = radius
        self.v_mean = v_mean
        self.shortest_path_action = shortest_path_action
        self.shortest_path_length = shortest_path_length
        self.id = id
        self.state = state

    def update(self, px, py, vx, vy, gx, gy, radius, v_mean, shortest_path_action, shortest_path_length, id, state):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.gx = gx
        self.gy = gy
        self.radius = radius
        self.v_mean = v_mean
        self.shortest_path_action = shortest_path_action
        self.shortest_path_length = shortest_path_length
        self.id = id
        self.state = state

    def to_main_array(self):
        status = [round(self.px, 3), round(self.py, 3), self.state]
        return status

    def to_numpy_array(self):
        status = [self.px, self.py, self.vx, self.vy, self.gx, self.gy, self.radius, self.v_mean,
                  self.shortest_path_action, self.shortest_path_length, self.id, self.state]
        return np.array(status)


class FullStatus:
    def __init__(self, static_map, self_status, other_status):
        self._map = static_map
        self._self = self_status
        self._other = other_status

    def to_numpy_array(self):
        np_self = self._self.to_numpy_array()
        np_map = self._map
        np_other = np.array(self._other)
        return {'map': np_map, 'self': np_self, 'other': np_other}


class LocalFullStatus:
    def __init__(self, self_observation, self_status, other_status):
        self.observation = self_observation
        self.self = self_status
        self.other = other_status

    def update(self, self_observation, self_status, other_status):
        self.observation = self_observation
        self.self = self_status
        self.other = other_status

    def to_numpy_array(self):
        np_observation = np.array(self.observation)
        np_self = np.array(self.self)
        np_other = np.array(self.other)
        return {'observation': np_observation, 'self': np_self, 'other': np_other}
