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


import time

# 1.构建有向图
# 2.时间离散化
# 3.路口检测
#

import networkx as nx
import math

class Route_planning(object):

    def __init__(self, path):

        print("init Route_planning")

        self.G = nx.DiGraph()

        self.__init_f(path)

    def __init_f(self, path):
        with open(path, "r") as f:
            #height = int(f.readline())
            n = int(f.readline())
            for x in range(n):

                x1, y1, x2, y2 = map(int, f.readline().split())

                n1 = str(x1) + "-" + str(y1)
                n2 = str(x2) + "-" + str(y2)

                weight = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)//10
                self.G.add_edge(n1, n2, weight=weight)

    def search_fff(self, s, t,way):

        #path,crossing = self.crossing_check(self.use_dijkstra(s,t))
        #path,crossing = self.crossing_check(self.use_astar(s,t))
        if way ==1:
         path,crossing = self.crossing_check(self.use_astar(s,t))
        elif way ==2:
            path, crossing = self.crossing_check(self.use_dijkstra(s, t))
        elif way ==3:
            path, crossing = self.crossing_check(self.use_bell(s, t))
        #path, crossing = self.crossing_check(self.use_johnson(s, t))

        #print("---------path----",path)
        #print("-----crossing-----", crossing)
        #print("length--crossing---",dis)
        return self.split_path(crossing)


    def use_dijkstra(self,s,t):

        path = nx.dijkstra_path(self.G, source=s, target=t)

        # path1 = nx.multi_source_dijkstra_path(self.G, {0, 4})
        # length1 = nx.multi_source_dijkstra_path_length(self.G, {0, 4})
        #
        # path1 = dict(nx.all_pairs_dijkstra_path(self.G))
        # length1 = dict(nx.all_pairs_dijkstra_path_length(self.G))
        #
        # # 双向搜索的迪杰斯特拉
        # length, path = nx.bidirectional_dijkstra(self.G, 0, 4)
        return path

    def use_astar(self,s,t):

        path = nx.astar_path(self.G, source=s, target=t)


        return path

    def use_bell(self, s, t):

        path = nx.bellman_ford_path(self.G, source=s, target=t)

        # path1 = nx.single_source_bellman_ford_path(self.G, 0)
        # length1 = dict(nx.single_source_bellman_ford_path_length(self.G, 0))
        #
        # path2 = dict(nx.all_pairs_bellman_ford_path(self.G))
        # length2 = dict(nx.all_pairs_bellman_ford_path_length(self.G))
        #
        # length, path = nx.single_source_bellman_ford(self.G, 0)
        # pred, dist = nx.bellman_ford_predecessor_and_distance(self.G, 0)
        # print('\n加权图最短路径长度和前驱: ', pred, dist)
        return path

    def use_johnson(self,s,t):
        path = nx.johnson(self.G, source=s, target=t)
        return path


    def split_path(self, crossing):

        split_path = []

        crossing = sorted(crossing.items(), key=lambda d: int(d[0]), reverse=False)

        #print("crossing: ", crossing)

        for i in range(len(crossing) - 1):

            x1, y1 = self.__get_x_y(crossing[i][1])
            x2, y2 = self.__get_x_y(crossing[i + 1][1])

            split_edage = self.__get_split_edage(x1, y1, x2, y2)

            if i < len(crossing) - 2:
                split_edage.pop()
            split_path.extend(split_edage)
        return split_path

    def __get_split_edage(self, x1, y1, x2, y2):
        point_list = []

        if x1==x2:
            d=0
            h=0
            if y1>y2:
                d=-10
                h=-1
            else:
                d=10
                h=1

            for y in range(y1,y2+h,d):
                point_list.append((x1,y))

        if y1==y2:

            d = 0
            h = 0
            if x1 > x2:
                d = -10
                h = -1
            else:
                d = 10
                h = 1

            for x in range(x1,x2+h,d):
                point_list.append((x,y1))

        return point_list








    def network_shortest_path(self, car, environment):

        print("car:", car)

        delete_side = []

        # 最短路径搜索路径
        crossing = None
        beyond_flag = None

        flag = 1

        time2222 = time.time()
        while flag != 0:
            path, crossing = self.__planning(car, beyond_flag, crossing)  # 根据当前搜索最短路径
            # 计算等待队列,比较等待容量,当等待队列超出容量时，重新进行规划
            beyond_flag, go_time = self.__check_wait_queue(car['name'], crossing, environment)

            print("beyond_flag:", beyond_flag)
            print("waiting_time:", go_time)

            flag = sum(beyond_flag)
        print("避障用时：　", time.time()-time2222)
        return path, crossing, go_time

    def __check_wait_queue(self, name, crossing, environment):

        beyond_flag = []

        go_time = []

        result = sorted(crossing.items(), key=lambda d: int(d[0]), reverse=False)



        # 路口等待队列容量判定

        wait_time = 0

        for i in range(len(result) - 1):

            if environment.is_in_crossing_collision_dict(result[i+1][1]):

                x1, y1 = self.__get_x_y(result[i][1])
                x2, y2 = self.__get_x_y(result[i + 1][1])

                direction = 'direction' + str(self.__get_direction(x1, y1, x2, y2))

                # 计算该路口的通过时间
                go_time_t = environment.get_go_time(result[i+1][1], int(result[i+1][0])+wait_time)

                wait_time = wait_time + go_time_t - (int(result[i+1][0])+wait_time)

                go_time.append(go_time_t)

                wait_queue_length = environment.get_waiting_queue(result[i+1][1], direction, result[i+1][0])

                print("wait_queue_length", wait_queue_length)

                # 若超出队列容量,则重新进行路径规划

                if int(wait_queue_length)+1 > self.get_cost(result[i][1], result[i+1][1]):
                    beyond_flag.append(1)
                else:
                    beyond_flag.append(0)
            else:
                beyond_flag.append(0)
                go_time.append(result[i+1][0])
        return beyond_flag, go_time

    def __planning(self, car, waiting_flag, crossing):

        if waiting_flag is None or crossing is None:
            path, crossing = self.crossing_check(nx.dijkstra_path(self.G, source=car['start'], target=car['goal']))

        else:

            map_t = self.G.copy()
            result = sorted(crossing.items(), key=lambda d: int(d[0]), reverse=False)


            # 删除超出等待队列容量的边

            for i in range(len(result) - 1):
                if waiting_flag[i] == 1:

                    map_t.remove_edge(result[i][1], result[i+1][1])

            path, crossing = self.crossing_check(nx.dijkstra_path(map_t, source=car['start'], target=car['goal']))

        return path, crossing

    def __index_wait_time(self, t, c_list):
        flag = None
        t1 = str(t)
        while flag is None:
            try:
                index = c_list.index(t1)
                t1 = str(int(t1)+1)
            except:
                flag = 0
        return int(t1)-int(t)

    def dynamic_search(self, crossing, wait_time, delete_edage,crossing_collision_dict):

        result = sorted(crossing.items(), key=lambda d: int(d[0]), reverse=True)

        wait_time.reverse()

        return self.f(result, wait_time, delete_edage, crossing_collision_dict)



    def f(self, result, wait_time, delete_edage, crossing_collision_dict):

        if len(result) <= 1:
            return 0

        t1 = int(result[1][0])
        p1 = result[1][1]
        t2 = int(result[0][0])
        p2 = result[0][1]


        t = t2-t1+wait_time[0]

        result.pop(0)

        wait_time.pop(0)



        graph = self.G.copy()

        for p1, p2 in delete_edage.items():
            graph.remove_edge(p1, p2)

        try:
            path, crossing = self.crossing_check(nx.dijkstra_path(self.G, source=p1, target=p2))

            wait_time_1 = sum(self.__check_wait_time(crossing, crossing_collision_dict))

            cost =  max(map(int,crossing.keys()))+wait_time_1

        except:
            print("无路径")
            cost = 10000

        if cost<t:   # 舍去该边
            delete_edage[p1] = p2

        return self.f(result, wait_time, delete_edage, crossing_collision_dict)

    def crossing_check(self, path):

        crossing = dict()

        weight = 0
        for i in range(len(path)):

            crossing[str(int(weight))] = path[i]
            if i<len(path)-1:
                weight += self.G[path[i]][path[i+1]]['weight']

        return path, crossing


    def dijkstra_path＿search(self,delete_edage,s,t):

        graph = self.G

        for p1, p2 in delete_edage.items():
            graph.remove_edge(p1, p2)

        path, crossing = self.crossing_check(nx.dijkstra_path(self.G, source=s, target=t))

        return path, crossing

    def time_series(self,path):
        pass

    def get_cost(self, p1, p2):

        return self.G[p1][p2]['weight']

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