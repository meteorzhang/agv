#AgentManagerment, batch
import sys

import networkx as nx

sys.path.append('../')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from virtual.navi.GoStatus import FullStatus,LocalFullStatus
from virtual.navi.GoConfig import GoConfig
from virtual.navi.GoStatus import AgentState
from multiprocessing import Manager

from virtual.navi.GoEnvironment import GoEnvironment
from virtual.navi.GoAgentAction import GoAgentAction

import time
import math
#import numba as nb
#from numba import jit
from threading import Condition
from pysim2d import pysim2d

'''
by zoulibing
notes:
    1. create and delete ProcessAgent with same map(same of roads and allow-areas)
    2. sharing status of agents.
    3. dispatch tasks.(start-pos / goal-pos)
'''


class GoSharedStatus(object):
    def __init__(self,map_name):
        print("GoSharedStatus------------------------->")

        self.name = None
        self.allow_area_gunplot_cfg_url= None
        self.roads_gunplot_cfg_url=None
        self.staic_map_array = None
        self.agents_dynamic_status = dict()  # 多线程共享字典,保存agent状态,键为id，值为什么？？state
        self.agents_con = Condition()
        self.share_to_main = None

        self.actions = GoAgentAction()

        self.env = GoEnvironment(0, self, self.actions, False)

        self.map_pic_str = None

        self.map_width = 910*0.05

        self.map_height = 910*0.05

        self.map_name = map_name[0:-4]
        self.one_channel_map_pic_str = "custom_map/final3/new_map.png"
        self.allow_area_url = self.map_name+"_tongxing_grid"+".txt"
        self.road_url = self.map_name+"_road_grid"+".txt"

        self.emulator = pysim2d.pysim2d()
        #self.emulator.init(self.one_channel_map_pic_str, self.allow_area_url, self.road_url, True, True, False)

        self.emulator.set_parameter(0.2, 0.25, 1.0, 0.2, 0.5, 5, 10)
        self.emulator.set_title(0, 1)

        # process environment
        # self.init_map()

    def init_agents(self, car_list):
        for a in car_list:
            self.update_agent_status(a.get_id(),a.get_location().x,a.get_location().y,a.get_goal().x, a.get_goal().y)

    def get_other_agents_status(self, myid=-1):
        results = list()
        for (k, v) in self.agents_dynamic_status.items():
            if k == myid:
                continue
            results.append(v)
        return results

    def get_other_agents_xy(self, myid=-1):
        results = list()
        for (k, v) in self.agents_dynamic_status.items():
            if k == myid:
                continue
            results.append((v.px, v.py, v.vx, v.vy))
        return results

    def delete_agent_status(self, id):
        for (k, v) in self.agents_dynamic_status.items():
            if k == id:
                del self.agents_dynamic_status[k]
                break

    def get_local_full_status(self, myid=-1):

        other_status_local = list()
        self_status = self.agents_dynamic_status.get(myid)  # 自身状态
        #print('static_map_shape:')
        #print(self.staic_map_array.shape)

        #return self_status
        # print("self_status:", self_status.px, self_status.py, self_status.vx, self_status.vy, self_status.gx,
        #       self_status.gy, self_status.dis_road, self_status.dis_area, self_status.dis_other, self_status.radius)
        other_status = self.get_other_agents_status(myid)  # 其他状态
        # for i in range(len(other_status)):
        #     print("other_status", other_status[i].px, other_status[i].py, other_status[i].vx, other_status[i].vy,
        #           other_status[i].gx, other_status[i].gy, other_status[i].dis_road, other_status[i].dis_area,
        #           other_status[i].dis_other, other_status[i].radius)
        # self state
        self_status_local = [self_status.vx, self_status.vy, self_status.gx - self_status.px,
                             self_status.gy - self_status.py,
                             np.sqrt((self_status.gx - self_status.px) ** 2 + (self_status.gy - self_status.py) ** 2),
                             self_status.dis_road,
                             self_status.dis_area, self_status.dis_other, self_status.radius]
        # print("self_status_local", self_status_local)
        # self observation
        self_obervation_local = self_status.virtual_laser
        # print("self_obervation_local", self_obervation_local)
        # print("self_obervation_local", self_obervation_local)
        # other status
        for i in range(len(other_status)):  # 坐标转换
            # print(range(len(other_status)))
            # print("index i", i)
            # print("len(other_status)", len(other_status))
            other_status_local.append([other_status[i].px - self_status.px, other_status[i].py - self_status.py,
                                       other_status[i].vx - self_status.vx,
                                       other_status[i].vy - self_status.vy, other_status[i].gx - self_status.px,
                                       other_status[i].gy - self_status.py,
                                       np.sqrt((other_status[i].px - self_status.px) ** 2 + (
                                                   other_status[i].py - self_status.py) ** 2),
                                       other_status[i].radius, other_status[i].radius + self_status.radius])

        if len(other_status_local) >= 2:
            for i in range(1, len(other_status_local)):
                for j in range(0, len(other_status_local) - i):
                    if other_status_local[j][6] < other_status_local[j + 1][6]:
                        other_status_local[j], other_status_local[j + 1] = other_status_local[j + 1], \
                                                                           other_status_local[j]
        # print("other_status_local", other_status_local)
        # local map
        image_position_x = int((self_status.px / self.map_width) * int(self.map_width / GoConfig.MAP_RESOLUTION))
        image_position_y = int((self_status.py / self.map_height) * int(self.map_height / GoConfig.MAP_RESOLUTION))
        # local_map_image_1 = Image.fromarray(self.staic_map_array.astype('uint8'))
        # local_map_image_1.show()
        # local_map_array = self.staic_map_array[0:224, 100:324]
        width_start=(int(self.map_width / GoConfig.MAP_RESOLUTION) - image_position_y)
        width_end=((int(self.map_width / GoConfig.MAP_RESOLUTION) - image_position_y) + GoConfig.WIDTH)
        height_start=image_position_x
        height_end=(image_position_x + GoConfig.HEIGHT)
        # print('width_start=%d,width_end=%d,height_start=%d,height_end=%d'%(width_start)%(width_end)%(height_start)%(height_end))
        local_map_array = self.staic_map_array[width_start:width_end,height_start:height_end]

        # local_map_image_2 = Image.fromarray(local_map_array.astype('uint8'))
        # draw_local_map_image_1 = ImageDraw.Draw(local_map_image_2)
        # title1 = str(self_status.px/0.05)
        # title2 = str(self_status.py/0.05)
        # title3 = str(image_position_x)
        # title4 = str(image_position_y)
        # draw_local_map_image_1.text((0, 0), title1+', '+title2+', '+title3+', '+title4)
        # local_map_image_2.show()ifo_n
        # local_map_array：局部地图
        #print("local_map_array", local_map_array.shape)

        # if len(other_status_local) > 7:
        #     other_status_local = other_status_local[0:7]

        return LocalFullStatus(local_map_array, self_obervation_local, self_status_local, other_status_local[-19:])

    def get_local_full_polar_status(self, myid=-1):
        other_status_local = []
        self_status = self.agents_dynamic_status.get(myid)
        other_status = self.get_other_agents_status(myid)

        # self state
        self_vel = math.sqrt(self_status.vx ** 2 + self_status.vy ** 2)

        self_vel_theta = math.atan2(self_status.vy, self_status.vx)

        if - math.pi <= self_vel_theta <= 0:
            self_vel_theta = self_vel_theta + math.pi * 2

        self_goal = math.sqrt((self_status.gx - self_status.px) ** 2 + (self_status.gy - self_status.py) ** 2)

        self_goal_theta = math.atan2(self_status.gy - self_status.py, self_status.gx - self_status.px)

        if - math.pi <= self_goal_theta <= 0:
            self_goal_theta = self_goal_theta + math.pi * 2

        self_status_local = [self_vel, self_vel_theta, self_goal, self_goal_theta, self_status.shortest_path_length,
                             self_status.v_mean, self_status.radius]

        # self observation
        self_observation_local = self_status.shortest_path_action

        # other observation
        for item in other_status:
            other_pos = math.sqrt((item.px - self_status.px)**2 + (item.py - self_status.py)**2)
            other_pos_theta = math.atan2(item.py - self_status.py, item.px - self_status.px)
            if - math.pi <= other_pos_theta <= 0:
                other_pos_theta = other_pos_theta + math.pi * 2

            other_vel = math.sqrt((item.vx - self_status.vx)**2 + (item.vy - self_status.vy)**2)
            other_vel_theta = math.atan2(item.vy - self_status.vy, item.vx - self_status.vx)
            if - math.pi <= other_vel_theta <= 0:
                other_vel_theta = other_vel_theta + math.pi * 2

            other_goal = math.sqrt((item.gx - self_status.px)**2 + (item.gy - self_status.py)**2)
            other_goal_theta = math.atan2(item.gy - self_status.py, item.gx - self_status.px)
            if - math.pi <= other_goal_theta <= 0:
                other_goal_theta = other_goal_theta + math.pi * 2

            other_status_local.append([other_pos, other_pos_theta, other_vel, other_vel_theta, other_goal,
                                       other_goal_theta, item.shortest_path_length, item.radius,
                                       item.radius + self_status.radius])

        length_agent_s_to_agent_e = self.shortest_length_node_to_nodes(self_status.px, self_status.py, other_status,
                                                                       self.G_noDi, self.road_node_Nos, self.road_node_info,
                                                                       self.road_lines, self.node_edges)

        for i in range(len(other_status)):
            other_status_local[i].insert(6, length_agent_s_to_agent_e[i])

        if len(other_status_local) >= 2:
            for i in range(1, len(other_status_local)):
                for j in range(0, len(other_status_local)-i):
                    if other_status_local[j][6] < other_status_local[j+1][6]:
                        other_status_local[j], other_status_local[j+1] = other_status_local[j+1], other_status_local[j]
                    if other_status_local[j][6] == other_status_local[j+1][6]:
                        if other_status_local[j][0] < other_status_local[j+1][0]:
                            other_status_local[j], other_status_local[j + 1] = other_status_local[j + 1], other_status_local[j]

        current_local_full_status = LocalFullStatus(self_observation_local, self_status_local, other_status_local[-19:])

        return current_local_full_status

    def get_local_full_polar_status_Eular(self, myid=-1):
        other_status_local = []
        self_status = self.agents_dynamic_status.get(myid)
        other_status = self.get_other_agents_status(myid)

        # self state
        self_vel = math.sqrt(self_status.vx**2 + self_status.vy**2)

        self_vel_theta = math.atan2(self_status.vy, self_status.vx)

        if - math.pi <= self_vel_theta <= 0:
            self_vel_theta = self_vel_theta + math.pi * 2

        self_goal = math.sqrt((self_status.gx - self_status.px)**2 + (self_status.gy - self_status.py)**2)

        self_goal_theta = math.atan2(self_status.gy - self_status.py, self_status.gx - self_status.px)

        if - math.pi <= self_goal_theta <= 0:
            self_goal_theta = self_goal_theta + math.pi * 2

        self_status_local = [self_vel, self_vel_theta, self_goal, self_goal_theta, self_status.shortest_path_length,
                             self_status.v_mean, self_status.radius]

        # self observation
        self_observation_local = self_status.shortest_path_action

        # other state
        for item in other_status:
            other_pos = math.sqrt((item.px - self_status.px)**2 + (item.py - self_status.py)**2)
            other_pos_theta = math.atan2(item.py - self_status.py, item.px - self_status.px)
            if - math.pi <= other_pos_theta <= 0:
                other_pos_theta = other_pos_theta + math.pi * 2

            other_vel = math.sqrt((item.vx - self_status.vx)**2 + (item.vy-self_status.vy)**2)
            other_vel_theta = math.atan2(item.vy - self_status.vy, item.vx - self_status.vx)
            if - math.pi <= other_vel_theta <= 0:
                other_vel_theta = other_vel_theta + math.pi * 2

            other_goal = math.sqrt((item.gx - self_status.px)**2 + (item.gy - self_status.py)**2)
            other_goal_theta = math.atan2(item.gy - self_status.py, item.gx - self_status.px)
            if - math.pi <= other_goal_theta <= 0:
                other_goal_theta = other_goal_theta + math.pi * 2

            other_status_local.append([other_pos, other_pos_theta, other_vel, other_vel_theta, other_goal, other_goal_theta,
                                       item.shortest_path_length, item.radius, item.radius+self_status.radius])

        if len(other_status_local) >= 2:
            for i in range(1, len(other_status_local)):
                for j in range(0, len(other_status_local)-i):
                    if other_status_local[j][0] < other_status_local[j+1][0]:
                        other_status_local[j], other_status_local[j+1] = other_status_local[j+1], other_status_local[j]

        current_local_full_status = LocalFullStatus(self_observation_local, self_status_local, other_status_local[-19:])

        return current_local_full_status

    # update agent's state to share dict
    def update_agent_status(self, curr_status):
        update = True
        new_value = {curr_status.id: curr_status}

        for (k, v) in self.agents_dynamic_status.items():
            if k == curr_status.id:
                continue
            if curr_status.px == v.px and curr_status.py == v.py:
                update = False

        if update:
            self.agents_dynamic_status.update(new_value)

        return update
        #print('GoSharedStatus name=%s update_agent_status id=%s pre_agents_dynamic_status_size=%d' % (self.name, id, len(self.agents_dynamic_status)))
        #print(list(self.agents_dynamic_status.keys()))
        # self.agents_dynamic_status.update(new_value)
        #print(' update_agent_status id=%s after_agents_dynamic_status_size=%d' % (id, len(self.agents_dynamic_status)))
        #print(list(self.agents_dynamic_status.keys()))

    def get_other_robots_xy(self, id, x, y):

        other_agents = self.get_other_agents_status(id)

        min_dis = 1000

        for agent in other_agents:
            dis = math.sqrt((agent.px - x) ** 2 + (agent.py - y) ** 2)
            if min_dis>dis:
                min_dis = dis
        return min_dis

    def get_all_state(self):
        # return self.agents_dynamic_status.copy()
        results = {}  # 返回到ui界面
        for (k, v) in self.agents_dynamic_status.items():
            results[k] = v.to_main_array()
        return results

    def get_full_status(self, myid=-1):
        other_status = self.get_other_agents_status(myid)
        self_status = self.agents_dynamic_status.get(myid)
        return FullStatus(self.staic_map_array, self_status, other_status)

    def full_status_to_numpy(self, status):
        self_np=[]

        pass


    '''
    process static 
    '''

    def init_map(self, map_width, map_height, map_pic_str):  #
        # 加载地图，处理地图等
        self.map_pic_str = map_pic_str
        self.map_width = map_width
        self.map_height = map_height
        if self.map_pic_str is not None:
            map_global = Image.open(self.map_pic_str).convert('RGB')
            map_global = map_global.resize(
                (int(map_width / GoConfig.MAP_RESOLUTION), int(map_height / GoConfig.MAP_RESOLUTION)), Image.ANTIALIAS)
            map_img = np.array(map_global)
            axis_0 = np.zeros((int(GoConfig.HEIGHT / 2), int(map_width / GoConfig.MAP_RESOLUTION), 3), dtype=np.int)
            print(self.map_width)
            print(axis_0.shape)
            print(map_img.shape)
            map_img = np.concatenate((map_img, axis_0), axis=0)
            map_img = np.concatenate((axis_0, map_img), axis=0)
            axis_1 = np.zeros((int(map_height / GoConfig.MAP_RESOLUTION) + GoConfig.HEIGHT, int(GoConfig.WIDTH / 2), 3),
                              dtype=np.int)
            map_img = np.concatenate((map_img, axis_1), axis=1)
            map_img = np.concatenate((axis_1, map_img), axis=1)
            # local_map_image_1 = Image.fromarray(map_img.astype('uint8'))
            # local_map_image_1.show()
            self.staic_map_array = map_img
            print("self.staic_map_array", self.staic_map_array.shape)
