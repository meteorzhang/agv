from collections import namedtuple
import math
ActionXY = namedtuple('ActionXY', ['vx', 'vy'])
ActionRot = namedtuple('ActionRot', ['v', 'r'])
ActionVTh = namedtuple('ActionVTh', ['v', 'theta'])


def map_action(action, stepping_velocity):
    if action == 0:
        vx = 0
        vy = stepping_velocity
    elif action == 1:
        vx = 0
        vy = - stepping_velocity
    elif action == 2:
        vx = stepping_velocity
        vy = 0
    elif action == 3:
        vx = - stepping_velocity
        vy = 0
    else:
        raise AttributeError("Invalid Action: {}".format(action))

    return vx, vy


def polar_action(action, stepping_velocity):
    if action == 0:
        v = 0
        theta = math.pi * 2
    elif action == 1:
        v = stepping_velocity
        theta = math.pi / 2
    elif action == 2:
        v = stepping_velocity
        theta = - math.pi / 2 + 2 * math.pi
    elif action == 3:
        v = stepping_velocity
        theta = math.pi * 2
    elif action == 4:
        v = stepping_velocity
        theta = math.pi
    else:
        raise AttributeError("Invalid Action: {}".format(action))

    return v, theta
