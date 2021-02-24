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
import json
import math

from GoNavigation.route_planning import Route_planning

from GoNavigation.Environment import Environment

class Navigator(object):

    def __init__(self, path):

        self.route_palnning = Route_planning(path)  # 初始化有向图
        self.environment = Environment()
        self.solution = None







    def navigation_simulation(self, agent,way):
        solution = self.__search(agent,way)
        return solution



    def __search(self, car,way):

        # 有向图路径规划

        print("name: ",car.get_id())



        s = str(car.x) + "-" + str(car.y)
        t = str(car.goal_x) + "-" + str(car.goal_y)

        print("s: ", s)
        print("t: ", t)

        solution = dict()
        
        solution['name'] = str(car.get_id())
        #print("-------solution_name",solution['name'])
        solution['path'] = self.route_palnning.search_fff(s, t,way)
        #print("-------solution_path",solution['path'])

        return solution

        # 检测是否有已规划的路径并进行截取

        # s = time.time()
        #
        # if self.solution is not None:  # 先检测是否agent已做过路径规划
        #     for ii, agent in enumerate(agents):
        #         if self.solution[ii]['goal'] == agent['goal']:
        #
        #             if agent['start'] == agent['goal']:
        #                 xx, yy = map(int, agent['start'].split("-"))
        #
        #                 agent["split_path"] = [(xx, yy)]
        #
        #                 continue
        #
        #             else:
        #                 #  路径分割
        #                 #  经过的时间
        #                 pass_t, new_split_path = self._get_pass_time(self.solution[ii]['split_path'], agent['start'])
        #                 #  更新agent的变量
        #
        #                 agent["split_path"] = new_split_path
        #
        #                 crossing = self.solution[ii]["crossing"]
        #                 go_time = self.solution[ii]["go_time"]
        #                 new_crossing = dict()
        #                 new_go_time = []
        #                 for t,v in crossing.items():
        #                     t1 = int(t)-pass_t
        #                     if t1 >= 0:
        #                         new_crossing[str(t1)] = v
        #                 agent["crossing"] = new_crossing
        #
        #                 for tt in go_time:
        #                     if (int(tt)-pass_t) >= 0:
        #                         new_go_time.append(str(int(tt)-pass_t))
        #                 agent["go_time"] = new_go_time
        #
        #                 #  环境更新
        #                 self.environment.update(agent["name"], agent["crossing"], agent["go_time"])
        # print("分割路径的时间：　",time.time()-s)



        # for agent in agents:
        #
        #     if 'split_path' in agent.keys():
        #         continue
        #
        #     if agent['start'] == agent['goal']:
        #         xx, yy = map(int, agent['start'].split("-"))
        #
        #         agent["split_path"] = [(xx, yy)]
        #
        #         continue
        #
        #     s = time.time()
        #
        #     # 1.根据环境路口字典,实现最短路径搜索功能,可否加入超级源汇点
        #     path, crossing, go_time = self.route_palnning.network_shortest_path(agent, self.environment)
        #     agent["path"] = path
        #     agent["crossing"] = crossing
        #
        #     agent["go_time"] = go_time
        #
        #     print("agents: ", agents)
        #
        #     print("路径搜索耗时：　", time.time() - s)
        #
        #     # 2.更新环境路口信息
        #
        #     s = time.time()
        #
        #     #print("self.environment.print_crossing_collision_dict()-----1-----: ", self.environment.print_crossing_collision_dict())
        #
        #     self.environment.update(agent["name"], agent["crossing"], agent["go_time"])
        #
        #     #print("self.environment.print_crossing_collision_dict()-----2-----: ", self.environment.print_crossing_collision_dict())
        #
        #     print("环境更新耗时：　", time.time() - s)
        #
        #     s = time.time()
        #
        #     # 3. 序列化
        #
        #     split_path = self.environment.get_split_path(agent["name"], agent["crossing"], agent["go_time"])
        #
        #     agent["split_path"] = split_path
        #
        #     print("时间序列耗时：　", time.time() - s)
        #
        # self._clean_value()  # 清空上次变量赋值
        #
        # self.solution = agents.copy()
        #
        # final_result = dict()
        # for agent in agents:
        #     final_result[agent['name']] = agent['split_path']

        #return final_result



    def _clean_value(self):

        self.environment.clean_value()

    def __get_direction(self,x1,y1,x2,y2):

        if x1 == x2 and y1>y2:
            return 0
        if x1 == x2 and y2>y1:
            return 2
        if y1 == y2 and x1>x2:
            return 3
        if y1 == y2 and x2>x1:
            return 1

    def __get_x_y(self, name):

        return map(int, str(name).split("-"))


    def __crossing_collision_check(self, agents):

        crossing_collision_dict = dict()

        for agent in agents:

            # 1.最短路径搜索
            path, crossing = self.route_palnning.network_shortest_path(agent) # 根据以规划的状态进行路径搜索
            agent["path"] = path
            agent["crossing"] = crossing

            # 2.增添路口字典


            # 3.路口等待规划


            # 4.路径修正


            # 5.路径生成

        crossing_collision_dict = dict()

        for agent in agents:

            for (k,v) in agent["crossing"].items():

                if v in crossing_collision_dict.keys():
                    crossing_collision_dict[v]["message"][k].append(agent['name'])
                else:

                    crossing_collision_dict[v] = dict()

                    crossing_collision_dict[v]["message"] = {k:[agent['name']]}
                    #crossing_collision_dict[v]["wait_queue"] = dict()
        #路口等待规则

        for agent in agents:

            wait_cost = 0

            wait_t = []

            res = sorted(agent['crossing'].items(), key=lambda d: d[0], reverse=False)

            for c in res:
                t = str(int(c[0])+wait_cost)

                if t in crossing_collision_dict[c[1]]['message'].keys():
                    agent_index = crossing_collision_dict[c[1]]['message'][t].index(agent['name'])
                    t1 = t
                    if agent_index>0:   # 前面有agent,需要进行等待
                        name = crossing_collision_dict[c[1]]['message'][t].pop(agent_index)
                        while t1 in crossing_collision_dict[c[1]]['message'].keys():
                            t1 = str(int(t)+1)
                        wait_cost = int(t1)-int(t)
                        crossing_collision_dict[c[1]]['message'][t1] = [name]
                    wait_t .append(t1)
            agent['wait_t'] = wait_t


        #
        # # 时间序列化
        #
        # # 1.时间修正
        #
        for agent in agents:

            res = sorted(agent['crossing'].items(), key=lambda d: d[0], reverse=False)
            new_res = dict()
            index = 0
            for (k,v) in res:
                new_res[agent['wait_t'][index]] = v
                index += 1

            agent['crossing'] = new_res
        pass

    def __collision_check(self):

        pass



    def __search_a_start(self, agent):

        solution = self.cbs.search_a_start(agent)  # 多远路径搜索算法

        if not solution:
            print(" Solution not found")
            return None
        else:
            return solution

    def load_no_obstacles(self, path):
        no_obstacles = []
        dimension = []
        x_max = 0
        y_max = 0
        with open(path, "r") as f:
            n = int(f.readline())
            for x in range(n):
                px, py = map(int, f.readline().split())
                x_max = max((px - 5) // 10, x_max)
                y_max = max((py - 5) // 10, y_max)
                no_obstacles.append(((px - 5) // 10, (py - 5) // 10))

        no_obstacles = list(set(no_obstacles))

        dimension.append(x_max + 1)
        dimension.append(y_max + 1)
        return no_obstacles, dimension

    def change_to_no_obstacles(self, no_obstacles, dimension):
        obstacles = []
        for i in range(dimension[0]):
            for j in range(dimension[1]):
                if (i, j) not in no_obstacles:
                    obstacles.append((i, j))

        return obstacles

    def get_no_obstacles(self):
        return self.no_obstacles

    def load_goal_list(self, path):
        goal_list = []
        with open(path, "r") as f:
            n = int(f.readline())
            for x in range(n):
                px, py = map(int, f.readline().split())
                goal_list.append(((px - 5) // 10, (py - 5) // 10))
        self.goal_list = goal_list

        #self.init_a_start_goal()

        with open('data.json', 'r') as f:
            self.goal_to_goal = json.load(f)

    def init_a_start_goal(self):

        agents = []
        a = dict()
        index = 0

        times = 0

        for i in range(len(self.goal_list)):

            for j in range(i, len(self.goal_list)):

                p1 = self.goal_list[i]
                p2 = self.goal_list[j]

                index += 1

                if p1[0]==p2[0] and p1[1]==p2[1]:
                    continue

                a.clear()
                p = []
                p.append(p1[0])
                p.append(p1[1])
                a['start'] = p.copy()
                p.clear()

                p.append(p2[0])
                p.append(p2[1])
                a['goal'] = p.copy()
                p.clear()
                a['name'] = str(p1[0])+"-"+str(p1[1])+"-"+str(p2[0])+"-"+str(p2[1])

                agents.clear()
                agents.append(a.copy())
                solution = self.__search_a_start(agents)





                self.goal_to_goal[a['name']] = solution

        with open('data.json', 'w') as f:

            json.dump(self.goal_to_goal, f)

    def random_agent(self, car_list):
        agents = []
        a = dict()
        for car in car_list:
            a['start'] = str(car.x)+"-"+str(car.y)
            a['goal'] = str(car.goal_x)+"-"+str(car.goal_y)
            a['name'] = "agent" + str(car.get_id())
            agents.append(a.copy())
            a.clear()
        return agents

