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

from multiprocessing import Queue
import sys
#sys.path.append('../')

from virtual.navi.GoConfig import GoConfig
from virtual.navi.ThreadPredictor import ThreadPredictor
#from threading import Thread
from virtual.navi.GoProcessAgent import GoProcessAgent
from virtual.navi.GoNetwork import GoNetwork
from virtual.navi.GoAgentAction import GoAgentAction
from virtual.navi.GoSharedStatus import GoSharedStatus
from virtual.navi.route_planning import Route_planning
from PyQt5.QtCore import *
import numpy as np
import time

class Navigator1(object):

    def __init__(self):
        pass

        # if GoConfig.LOAD_CHECKPOINT:  # 加载模型
        #     self.stats.episode_count.value = self.model.load()
        # self.model = None

    def init(self, path):
        self.map_m = None
        self.agents = dict()  # AGENT字典
        self.isready = False  # indicate the Network loaded or didn't

        # self.map_path = "/home/brook/NaviPlanningEmulatorGrid/test/custom_map/11_13/init_road.png"

        self.share = GoSharedStatus(path)

        self.prediction_q = Queue(maxsize=GoConfig.MAX_QUEUE_SIZE)  # 预测队列

        self.actions = GoAgentAction()
        time1 = time.time()
        self.model = GoNetwork(GoConfig.DEVICE, GoConfig.NETWORK_NAME, self.actions.action_space_nums)
        time2 = time.time()

        tm = time2-time1
        print("time-------",tm)

        if GoConfig.LOAD_CHECKPOINT:
            self.model.load()

        self.map_path = "11_13/init_road.png"


        # Build the directed road network from the road network file
        self.route_palnning = Route_planning(path)

        self.predictors = []
        for i in range(8):
            self.add_predictor()  # 预测线程


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



    def get_all_state(self):

        # 执行一次循环
        # for d, agent in self.agents.items():
        #     agent.run_agent()
        return self.agents[0].share_env.get_all_state()

    def init_map_m(self, m):
        self.map_m = m

    def init_data(self, map_path, car_list):
        # self.model.init_map(map_path)  # 初始化网络
        # print("navi:init_data!")
        self.agvs = car_list
        print(car_list)

        map_path = "custom_map/final2/init_road.png"


        self.share.init_map(915*0.05, 915*0.05, map_path)


        self.update_agents(self.agvs)  # 更新carlist ,启动每个线程
        # self.share.init_map()  # 共享位姿？


    def delete(self):  # 销毁预测线程
        for i in range(10):
            self.remove_predictor()

    def loadmap(self, map_url):  # 初始化,如何应对地图大小的问题
        self.map_path = map_url
        pass

    def update_agents(self, car_list):
        # print("update_agents!!!")


        self.road_node_Nos, self.road_node_info, self.road_lines, self.road_lines_num, self.node_edges=self._init_road_network(self.share.road_url)
        print("road_lines_num",self.road_lines_num)
        for x in car_list:
            self.add_agent(x,self.road_node_Nos, self.road_node_info, self.road_lines, self.road_lines_num, self.node_edges)

        self.start_all_agent()  # 开始所有agent线程
    # 添加新的agent
    def add_agent(self, a,road_node_Nos, road_node_info, road_lines, road_lines_num, node_edges):  # 为每个agent创建
        s = time.time()

        agent = GoProcessAgent(a.get_id(), self.share, self.actions, self.prediction_q, self.route_palnning,road_node_Nos, road_node_info, road_lines, road_lines_num, node_edges)
        print("init_map cost time:", time.time() - s)

        print("navi: ", a.get_id(), a.x, a.y)
        t = time.time()

        agent.init_data(a.get_id(), a.x, a.y, a.x, a.y)
        print("init_map cost time111:", time.time() - t)

        print("key: ", type(a.get_id()))

        self.agents[a.get_id()] = agent

    def set_goal(self, car):

        print("navi------: ", car.get_id(), car.x, car.y, car.goal_x, car.goal_y)

        self.agents[car.get_id()].set_goal_xy(car.x, car.y, car.goal_x, car.goal_y, car)

        #  self.agents[car.get_id()].start()

    def set_second(self):
        for i in range(len(self.agents)):
            self.agents[i].start_colli_num()

    def end_collision(self):
        for i in range(len(self.agents)):
            self.agents[i].end_colli_num()

    def get_all_collision_number(self):

        for i in range(len(self.agents)):

            self.agvs[i].set_collision_number(self.agents[i].get_colli())
            #print("i++++++++++++++++++++", self.agents[i].get_colli())


    def delete_agent(self,x): # 删除
        agent=self.agents[x]
        #if agent is not None and isinstance(agent,GoProcessAgent):
        agent.stop()




        pass
    def delete_all_agent(self):
        for (k, v) in self.agents.items():
            self.delete_agent(k)

    def pause_agent(self,id):  # 暂停
        agent = self.agents[id]
        if agent is not None and isinstance(agent, GoProcessAgent):
            agent.pause()

    def pause_all_agent(self,id):  # 暂停所有AGENT
        for (k, v) in self.agents.items():
            self.pause_agent(k)
        pass

    def enable_show(self, status):  # 使能显示端
        if status:
            pass

    def start_all_agent(self):
        for (k, v) in self.agents.items():
            self.start_agent(k)
        pass

    def start_agent(self, fid):
        agent = self.agents[fid]
        #print("start:", id)
        if agent is not None and isinstance(agent, GoProcessAgent):
            agent.resume()  # 等待任务状态
            agent.start()  # 启动线程

    def get_agent_by_id(self, fid):
        return self.agents[fid]

    def add_predictor(self):  # 增加预测线程
        print("增加预测线程", len(self.predictors))
        self.predictors.append(ThreadPredictor(self, len(self.predictors)))
        self.predictors[-1].start()

    def remove_predictor(self):  # 减少预测线程
        self.predictors[-1].exit_flag = True
        self.predictors[-1].join()
        self.predictors.pop()



