# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from datetime import datetime
from multiprocessing import Process, Queue, Value
import networkx as nx
import numpy as np
import time
import sys
from PyQt5.Qt import *
from PyQt5.QtWidgets import *
sys.path.append('../')

from virtual.navi.GoStatus import AgentState

from virtual.navi.GoSharedStatus import GoSharedStatus

from virtual.navi.GoConfig import GoConfig

#from ga3c.Environment import Environment
import random

from threading import Thread

from virtual.navi.action_mapper import ActionXY, ActionVTh

from PyQt5.QtCore import *

import math
#from numba import jit
import  copy


class GoProcessAgent(QThread):

    def __init__(self, fid, share_env, actions, prediction_q, planning,road_node_Nos, road_node_info, road_lines, road_lines_num, node_edges):
        super(GoProcessAgent, self).__init__()
        self.prediction_q = prediction_q
        # self.map_path = map_path
        self.share_env = share_env
        self.x = 0
        self.y = 0
        self.goal_x = 0
        self.goal_y = 0
        self.vx = 0
        self.vy = 0
        self.collision_number = 0
        self.agent_state = None
        self._actions = actions
        self.num_actions = self._actions.get_actions_nums()
        # self.actions = np.arange(self.num_actions)
        # one frame at a time
        self.wait_q = Queue(maxsize=1)
        self.state = 0  # 1,正常开始状态，2,暂停 0删除＼清除
        self.__id = fid
        self.__image = None
        self.route_palnning = planning
        self.goal_list = None
        self.goal_list_index = 1
        self.is_second_task = 0
        self.radius = GoConfig.RADIUS
        self.current_goal_x = 0
        self.current_goal_y = 0
        self.con = self.share_env.agents_con
        # new
        self.G = nx.DiGraph()
        # self.road_node_Nos, self.road_node_info, self.road_lines, self.road_lines_num, self.node_edges = \
        #     self._init_road_network(self.share_env.road_url)
        self.road_node_Nos=copy.deepcopy(road_node_Nos)
        self.road_node_info=copy.deepcopy(road_node_info)
        self.road_lines=copy.deepcopy(road_lines)
        self.road_lines_num=copy.deepcopy(road_lines_num)
        self.node_edges=copy.deepcopy(node_edges)


        # self.road_node_Nos, self.road_node_info, self.road_lines, self.road_lines_num, self.node_edges = \
        #     road_node_Nos, road_node_info, road_lines, road_lines_num, node_edges


        print("road_node_Nos---",self.road_node_Nos)
        print("road_node_info---",self.road_node_info)

        print("road_lines---",self.road_lines)

        print("road_lines_num---",self.road_lines_num)

        print("node_edges---",self.node_edges)



        self.node_labels = dict(zip(self.road_node_Nos, self.road_node_Nos))


    def init_data(self, id, x, y, goal_x, goal_y):
        # print("x: ", x, " y:", y)
        self.__id = id
        self.x, self.y = self.__world_coordinate(x, y)
        self.goal_x, self.goal_y = self.__world_coordinate(goal_x, goal_y)

        # add
        self._start_x = self.x
        self._start_y = self.y

        self._goal_x = self.goal_x
        self._goal_y = self.goal_y

        self.v_mean = np.array([1e-5, 1e-5, 1e-5])

        self.shortest_path_action = [1, 0.1, 0.1, 0.1, 0.1]

        self.shortest_path_length = 0

        self.state = 0



        curr_status = AgentState(self._start_x, self._start_y, 0, 0, self._goal_x, self._goal_y, self.radius,
                            np.mean(self.v_mean), self.shortest_path_action, self.shortest_path_length, self.__id, 0)
        s = time.time()
        self.share_env.update_agent_status(curr_status)  # 共享位姿
        print("初始化时，共享数据时间：", time.time()-s)


    def _init_road_network(self, road_config_file):
        with open(road_config_file) as f:
            # read the context in "road_grid" text
            list_text = []
            for line in f.readlines():
                list_text.append(line.strip().split('\n'))
            image_height = int(list_text[0][0])
            road_lines_num = int(list_text[1][0])

            # extract the road lines
            road_lines = np.zeros((road_lines_num, 4))
            row = 0
            for i in range(2, road_lines_num + 2):
                road_line = list_text[i][0].strip().split()
                road_lines[row, :] = road_line[0:4]
                # convert to local world coordinate system
                road_lines[row, 1] = image_height - road_lines[row, 1]
                road_lines[row, 3] = image_height - road_lines[row, 3]
                row += 1

            # extract the coordinates
            road_node_coordinates = road_lines.reshape((road_lines_num * 2, 2))
            road_node_coordinates = np.unique(road_node_coordinates, axis=0)

            # create the node: No. + Coordinate
            road_node_Nos = list(range(1, road_node_coordinates.shape[0] + 1))
            road_node_info = dict(zip(road_node_Nos, road_node_coordinates))

            # create edges
            node_edges = []
            for i in range(road_lines_num):
                start_point = np.array([road_lines[i][0], road_lines[i][1]])
                start_index = None
                end_point = np.array([road_lines[i][2], road_lines[i][3]])
                end_index = None
                for (key, value) in road_node_info.items():
                    if (value == start_point).all() and (not start_index):
                        start_index = key
                    if (value == end_point).all() and (not end_index):
                        end_index = key
                node_edges.append([start_index, end_index, {
                    'len': np.sqrt((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2)}])
        return road_node_Nos, road_node_info, road_lines, road_lines_num, node_edges




    def search_shortest_path_action(self, robot_x, robot_y, goal_x, goal_y, G, road_node_Nos,
                                    road_node_info, road_lines, road_lines_num, node_edges):
        # start = time.time()
        # add target node
        target_node_coordinate = np.zeros((1, 2))
        target_node_coordinate[0][0] = int(goal_x / 0.05)
        target_node_coordinate[0][1] = int(goal_y / 0.05)
        target_node = None
        for (key, value) in road_node_info.items():
            if abs(value[0]) < abs(target_node_coordinate[0][0]) + 1 and \
                    abs(value[1]) < abs(target_node_coordinate[0][1]) + 1:
                target_node = key

        if target_node == 0:
            #print(target_node)
            raise Exception("wrong target node", target_node)

        # add agent node
        agent_node_No = 0
        agent_node_coordinate = np.zeros((1, 2))
        agent_node_coordinate[0][0] = int(robot_x / 0.05)
        agent_node_coordinate[0][1] = int(robot_y / 0.05)
        agent_node = dict(zip([agent_node_No], agent_node_coordinate))
        road_node_info.update(agent_node)
        env_node_info = road_node_info

        # add node
        env_node_Nos = [agent_node_No] + road_node_Nos
        G.add_nodes_from(env_node_Nos)

        # add edges from agent to the nearest road line
        # calculate the distance from the agent to the lines
        env_node_labels = dict(zip(env_node_Nos, env_node_Nos))

        # agent_line_dist = []
        # for i in range(road_lines_num):
        #     dist = self.distance_p2seg(road_lines[i][0], road_lines[i][1], road_lines[i][2], road_lines[i][3],
        #                                agent_node_coordinate[0][0], agent_node_coordinate[0][1])
        #     agent_line_dist.append(dist)
        #
        # # find the nearest line index
        # agent_line_dist_shortest = float("inf")
        # agent_line_shortest_index = 0
        # for index, item in enumerate(agent_line_dist):
        #     if item < agent_line_dist_shortest:
        #         agent_line_shortest_index = index
        #         agent_line_dist_shortest = item``

        agent_line_shortest_index = self.distance_p2network(road_lines, agent_node_coordinate[0][0], agent_node_coordinate[0][1])

        # print("target line index:", agent_line_shortest_index)

        # find the shortest line's node
        agent_line_shortest_node0 = None
        agent_line_shortest_node1 = None
        for (key, value) in road_node_info.items():
            if value[0] == road_lines[agent_line_shortest_index][0] and value[1] == \
                    road_lines[agent_line_shortest_index][1]:
                agent_line_shortest_node0 = key
            if value[0] == road_lines[agent_line_shortest_index][2] and value[1] == \
                    road_lines[agent_line_shortest_index][3]:
                agent_line_shortest_node1 = key
        # end = time.time()
        # print("-----------------------shortest path action time--------------------------------------", start - end)

        # start = time.time()
        # add new edges from the agent node to road note
        # node_edges.append([agent_node_No, agent_line_shortest_node0, {'len': np.sqrt(
        #     (road_node_info[agent_line_shortest_node0][0] - agent_node_coordinate[0][0]) ** 2 + (
        #                 road_node_info[agent_line_shortest_node0][1] - agent_node_coordinate[0][1]) ** 2)}])
        node_edges.append([agent_node_No, agent_line_shortest_node1, {'len': np.sqrt(
            (road_node_info[agent_line_shortest_node1][0] - agent_node_coordinate[0][0]) ** 2 + (
                        road_node_info[agent_line_shortest_node1][1] - agent_node_coordinate[0][1]) ** 2)}])

        G.add_edges_from(node_edges)

        # The robot is at the road node or not
        at_node = False
        coincide_node = None
        for (key, value) in road_node_info.items():
            if key == 0:
                continue
            if value[0] == agent_node_coordinate[0][0] and value[1] == agent_node_coordinate[0][1]:
                at_node = True
                coincide_node = key

        shortest_path_direction = []
        shortest_path_action = [1, 0.1, 0.1, 0.1, 0.1]

        # start = time.time()
        if at_node:
            all_shortest_paths = nx.all_shortest_paths(G, source=coincide_node, target=target_node, weight='len')
            shortest_path_length = nx.shortest_path_length(G, source=coincide_node, target=target_node,
                                                           weight='len')
            for shortest_path in all_shortest_paths:
                if len(shortest_path) >= 2:
                    shortest_path_direction.append(
                        math.atan2((road_node_info[shortest_path[1]][1] - road_node_info[shortest_path[0]][1]),
                                   (road_node_info[shortest_path[1]][0] - road_node_info[shortest_path[0]][0])))
                else:
                    continue
        else:
            all_shortest_paths = nx.all_shortest_paths(G, source=agent_node_No, target=target_node, weight='len')
            shortest_path_length = nx.shortest_path_length(G, source=agent_node_No, target=target_node, weight='len')

            for shortest_path in all_shortest_paths:
                shortest_path_direction.append(
                    math.atan2((road_node_info[shortest_path[1]][1] - agent_node_coordinate[0][1]),
                               (road_node_info[shortest_path[1]][0] - agent_node_coordinate[0][0])))

        # return the shortest path action
        if len(shortest_path_direction) == 0:
            raise Exception("None shortest path direction")
        #print("shortest_path_direction:", shortest_path_direction)

        for i in shortest_path_direction:
            if math.pi / 2 - 0.01 <= i <= math.pi / 2 + 0.01:
                shortest_path_action[1] = 1
            elif - math.pi / 2 - 0.01 <= i <= - math.pi / 2 + 0.01:
                shortest_path_action[2] = 1
            elif 0 - 0.01 <= i <= 0 + 0.01:
                shortest_path_action[3] = 1
            elif math.pi - 0.01 <= i <= math.pi + 0.01:
                shortest_path_action[4] = 1
            else:

                raise AttributeError("Error shortest path direction")

        # node_edges.remove([agent_node_No, agent_line_shortest_node0, {'len': np.sqrt(
        #     (road_node_info[agent_line_shortest_node0][0] - agent_node_coordinate[0][0]) ** 2 + (
        #                 road_node_info[agent_line_shortest_node0][1] - agent_node_coordinate[0][1]) ** 2)}])
        node_edges.remove([agent_node_No, agent_line_shortest_node1, {'len': np.sqrt(
            (road_node_info[agent_line_shortest_node1][0] - agent_node_coordinate[0][0]) ** 2 + (
                        road_node_info[agent_line_shortest_node1][1] - agent_node_coordinate[0][1]) ** 2)}])

        G.clear()
        # end = time.time()
        # print("-----------------------shortest path action time--------------------------------------", start - end)
        return shortest_path_action, shortest_path_length * 0.05

    @staticmethod
    # @jit(nopython=True, nogil=True, parallel=True)
    def distance_p2network(road_lines, p_x, p_y):
        agent_line_dist = np.zeros(road_lines.shape[0])
        for i in np.arange(road_lines.shape[0]):
            cross = (road_lines[i][2] - road_lines[i][0]) * (p_x - road_lines[i][0]) \
                    + (road_lines[i][3] - road_lines[i][1]) * (p_y - road_lines[i][1])

            if cross <= 0:
                agent_line_dist[i] = np.sqrt((p_x - road_lines[i][0]) ** 2 + (p_y - road_lines[i][1]) ** 2)
                continue

            d2 = (road_lines[i][2] - road_lines[i][0]) ** 2 + (road_lines[i][3] - road_lines[i][1]) ** 2
            if cross >= d2:
                agent_line_dist[i] = np.sqrt((p_x - road_lines[i][2]) ** 2 + (p_y - road_lines[i][3]) ** 2)
                continue

            r = cross / d2
            p0 = road_lines[i][0] + (road_lines[i][2] - road_lines[i][0]) * r
            p1 = road_lines[i][1] + (road_lines[i][3] - road_lines[i][1]) * r
            agent_line_dist[i] = np.sqrt((p_x - p0) ** 2 + (p_y - p1) ** 2)

        # find the nearest line index
        agent_line_dist_shortest = 99999
        agent_line_shortest_index = 0
        for i in np.arange(agent_line_dist.shape[0]):
            if agent_line_dist[i] < agent_line_dist_shortest:
                agent_line_shortest_index = i
                agent_line_dist_shortest = agent_line_dist[i]

        return agent_line_shortest_index

    def __world_coordinate(self, x, y):
        return x*0.05, (915-y)*0.05

    def set_goal(self, goal):
        self.__goal = goal

    def set_goal_xy(self, x, y, goal_x, goal_y, car):
        self.car = car

        self._start_x = car.x * 0.05
        self._start_y = (915 - car.y) * 0.05

        self.vx = 0
        self.vy = 0

        self._goal_x = car.goal_x * 0.05
        self._goal_y = (915 - car.goal_y) * 0.05

        self.v_mean = np.array([1e-5, 1e-5, 1e-5])

        # self.collision_number = 0
        # s = str(x) + "-" + str(y)
        # t = str(goal_x) + "-" + str(goal_y)
        # print("s: ", s)
        # print("t: ", t)
        self.shortest_path_action, self.shortest_path_length = self.search_shortest_path_action(self._start_x, self._start_y, self._goal_x,
                                                                        self._goal_y, self.G, self.road_node_Nos,
                                                                        self.road_node_info, self.road_lines,
                                                                        self.road_lines_num, self.node_edges)

        # print("self.shortest_path_action, self.shortest_path_length", self.shortest_path_action, self.shortest_path_length)

        curr_status = AgentState(self._start_x, self._start_y, self.vx, self.vy, self._goal_x, self._goal_y,
                                 self.radius, np.mean(self.v_mean), self.shortest_path_action,
                                 self.shortest_path_length, self.__id, 0)

        self.share_env.update_agent_status(curr_status)

        self.state = 2

    def __get_x_y(self, name):

        # 坐标转换为世界坐标系

        print("name: ", name)

        xx, yy = map(int, str(name).split("-"))

        return self.__world_coordinate(xx, yy)

    def get_image(self):
        return self.__image

    def get_id_image(self):
        return self.__ID_image

    def convert_data(self, experiences):
        x_ = np.array([exp.state for exp in experiences])
        a_ = np.eye(self.num_actions)[np.array([exp.action for exp in experiences])].astype(np.float32)
        r_ = np.array([exp.reward for exp in experiences])
        return x_, r_, a_

    # 需要修改成(self,other,map格式)
    def predict(self, full_state):
        # put the state in the prediction q
        self.prediction_q.put((self.__id, full_state))
        # wait for the prediction to come back
        p, v = self.wait_q.get()
        return p, v

    def pause(self):
        self.state = 2
        pass

    def resume(self):  # 复位
        self.state = 1
        pass

    def stop(self):
        self.state = 0
        pass

    # 基于action推导出来下一个位置
    def run_action(self, ac):
        vx = round(ac.v * math.cos(ac.theta) + 0.001, 2)
        vy = round(ac.v * math.sin(ac.theta) + 0.001, 2)

        other_xy = self.share_env.get_other_agents_xy(self.__id)

        next_x = self.x + vx
        next_y = self.y + vy

        collision = False

        for a in other_xy:
            if math.sqrt((next_x - a[0]) ** 2 + (next_y - a[1]) ** 2) < 0.5:
                collision = True
                continue
            # if math.sqrt((next_x - (a[0] + a[2])) ** 2 + (next_y - (a[1] + a[3])) ** 2) < 0.5:
            #     collision = True
            #     continue

        if collision:
            self.vx = 0
            self.vy = 0
            if self.is_second_task == 1:
                #print("碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞")

                self.collision_number += 1
        else:
            self.x = self.x + vx
            self.y = self.y + vy
            self.vx = vx
            self.vy = vy

        for i in range(1, self.v_mean.shape[0]):
            self.v_mean[i-1] = self.v_mean[i]
        self.v_mean[-1] = np.sqrt(self.vx**2 + self.vy**2)

        return collision
        # print("after step:", self.__id, self.x, self.y, self.vx, self.vy)

        # 判断是否到达目标
        # time.sleep()
        # def check_collision(self):
        #
        #     other_robot = self.share_env.get_all_state()
        #     for d,l in other_robot.items

    def check_action_collision(self, action):
        vx = round(action.v * math.cos(action.theta) + 0.001, 2)
        vy = round(action.v * math.sin(action.theta) + 0.001, 2)

        other_xy = self.share_env.get_other_agents_xy(self.__id)

        next_x = self.x + vx
        next_y = self.y + vy

        for a in other_xy:
            if math.sqrt((next_x - a[0]) ** 2 + (next_y - a[1]) ** 2) < 0.5:
                return True
        return False

    def un_run_action(self, ac):
        vx = round(ac.v * math.cos(ac.theta) + 0.001, 2)
        vy = round(ac.v * math.sin(ac.theta) + 0.001, 2)

        self.x = self.x - vx
        self.y = self.y - vy
        self.vx = 0
        self.vy = 0

        self.v_mean[-1] = 0

        self.shortest_path_action, self.shortest_path_length = self.search_shortest_path_action(self.x,
                                                                                                self.y,
                                                                                                self._goal_x,
                                                                                                self._goal_y,
                                                                                                self.G,
                                                                                                self.road_node_Nos,
                                                                                                self.road_node_info,
                                                                                                self.road_lines,
                                                                                                self.road_lines_num,
                                                                                                self.node_edges)

    def create_action(self):
        dx = self.current_goal_x - self.x
        dy = self.current_goal_y - self.y

        if dy == 0:
            if dx > 0:
                return ActionVTh(0.5, math.pi*2)
            elif dx < 0:
                return ActionVTh(0.5, math.pi)
        if dx == 0:
            if dy > 0:
                return ActionVTh(0.5, math.pi/2)
            elif dy<0:
                return ActionVTh(0.5, - math.pi / 2 + 2 * math.pi)
        return ActionVTh(0, math.pi*2)

    def check_collision(self, vx, vy):
        other_xy = self.share_env.get_other_agents_xy(self.__id)

        next_x = self.x + vx
        next_y = self.y + vy

        for a in other_xy:
            if math.sqrt((next_x - a[0]) ** 2 + (next_y - a[1]) ** 2)<0.1:
               if self.is_second_task ==1:
                #print("碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞")

                self.collision_number += 1

                return False
        return True

    def get_colli(self):
        print("=====================", self.collision_number)
        return self.collision_number

    def start_colli_num(self):
        self.is_second_task = 1

    def end_colli_num(self):
        self.is_second_task = 0
        self.collision_number = 0

    def check_direction(self, vx, vy):

        # if (action.vx > 0 and self.vx <0) or (action.vy > 0 and self.vy <0):
        #      return False

        dx = self.current_goal_x - self.x
        dy = self.current_goal_y - self.y
        # print("碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞碰撞")

        # print("current_goal_x, current_goal_y", self.current_goal_x, self.current_goal_y)
        # print("self.x, self.y", self.x, self.y)
        # print("dx,dy, vx, vy: ", dx, dy, vx, vy)

        if dx == 0:
            if vx == 0 and ((dy >= 0 and vy >= 0) or (dy <= 0 and vy <= 0)):
                return True
            else:
                return False
        elif dy == 0:
            if vy == 0 and ((dx >= 0 and vx >= 0) or (dx <= 0 and vx <= 0)):
                return True
            else:
                return False
        return False

    def check_goal(self, ac):
        dis = math.sqrt((self.x - self._goal_x) ** 2 + (self.y - self._goal_y) ** 2)
        # print("-----------dis:", dis)

        if dis < 0.1:
            self.state = 0
            # print("update_agent_status: 1")

            # average velocity
            self.v_mean = np.array([1e-5, 1e-5, 1e-5])

            # shortest path action
            self.shortest_path_action = [1, 0.1, 0.1, 0.1, 0.1]

            # shortest path length
            self.shortest_path_length = 0

            curr_status = AgentState(self.x, self.y, self.vx, self.vy, self._goal_x, self._goal_y,
                                     self.radius, np.mean(self.v_mean), self.shortest_path_action, self.shortest_path_length,
                                     self.__id, 1)

            if not self.share_env.update_agent_status(curr_status):
                self.un_run_action(ac)
        else:
            self.state = 2

            self.shortest_path_action, self.shortest_path_length = self.search_shortest_path_action(self.x,
                                                                                             self.y,
                                                                                             self._goal_x,
                                                                                             self._goal_y,
                                                                                             self.G,
                                                                                             self.road_node_Nos,
                                                                                             self.road_node_info,
                                                                                             self.road_lines,
                                                                                             self.road_lines_num,
                                                                                             self.node_edges)

            curr_status = AgentState(self.x, self.y, self.vx, self.vy, self._goal_x, self._goal_y,
                                     self.radius, np.mean(self.v_mean), self.shortest_path_action, self.shortest_path_length,
                                     self.__id, 0)

            if not self.share_env.update_agent_status(curr_status):
                self.un_run_action(ac)

    def run(self):
        step_time = 0.35

        while True:
            # self.con.acquire()
            # print("state: ", self.__id, self.state)
            if self.state != 0 and self.state != 1:
                # current_full_state = self.env._get_full_current_state()
                start_time = time.time()

                current_full_state = self.share_env.get_local_full_polar_status_Eular(self.__id)  # 从共享环境中获取实验数据

                prediction, value = self.predict(current_full_state)
                # 根据预测的action的概率，选择action

                # prediction = []
                # for x in range(9):
                #     prediction.append(random.uniform(0, 1))

                action = np.argmax(prediction)

                # Correct the action to the route action, enumerate version
                if current_full_state.observation[action] != 1 or current_full_state.self[5] == 0:
                    revise_action = []
                    if current_full_state.self[5] == 0:  # v_mean = 0
                        for index, item in enumerate(current_full_state.observation[1:], start=1):
                            if item == 1:
                                revise_action.append(index)

                        action = revise_action[0]
                        for i in revise_action:
                            if prediction[i] >= prediction[action]:
                                action = i

                        a_v, a_theta = self._actions.get_action_by_indx(action)  # stay, if collision

                        if len(current_full_state.other) > 0:
                            if (abs(a_theta - current_full_state.other[-1][1]) <= 0.1) and \
                                    (a_v * 1 >= current_full_state.other[-1][0]):
                                action = 0
                    else:
                        for index, item in enumerate(current_full_state.observation):
                            if item == 1:
                                revise_action.append(index)

                        action = 0
                        for i in revise_action:
                            if prediction[i] >= prediction[action]:
                                action = i

                ac = self._actions.get_action_by_indx(action)

                # print("------action-----", self.__id, ac)

                if self.run_action(ac):
                    action = 0
                    ac = self._actions.get_action_by_indx(action)
                    self.check_goal(ac)
                else:
                    self.check_goal(ac)

                # if self.check_action_collision(ac):
                #     action = 0
                #     ac = self._actions.get_action_by_indx(action)
                #     self.run_action(ac)
                #     self.check_goal(ac)
                #     # print("collision")
                # else:
                #     self.run_action(ac)
                #     self.check_goal(ac)
                    # print("predict: ", self.__id, ac)
                    # print("agent step cost time: ", time.time() - start_time, self.__id, self.state)

                # self.con.notify()
                # self.con.wait()

                cost_time = time.time() - start_time
                if cost_time < step_time:
                    time.sleep(step_time - cost_time)

                # if self.check_direction(vx, vy):
                #     if self.check_collision(vx, vy):
                #         self.run_action(vx, vy)
                #         self.check_goal(vx, vy)
                #         print("predict: ", self.__id, vx, vy)
                #         print("agent step cost time: ", time.time() - start_time, self.__id, self.state)
                #     cost_time = time.time() - start_time
                #     if cost_time < step_time:
                #         time.sleep(step_time - cost_time)
                # else:
                #     ac = self.create_action()
                #     vx = round(ac.v * math.cos(ac.theta), 2)
                #     vy = round(ac.v * math.sin(ac.theta), 2)
                #     self.run_action(vx, vy)
                #     self.check_goal(vx, vy)
                #     cost_time = time.time() - start_time
                #     if cost_time < step_time:
                #         time.sleep(step_time - cost_time)
            else:
                # self.con.notify()
                # self.con.wait()
                time.sleep(0.001)
