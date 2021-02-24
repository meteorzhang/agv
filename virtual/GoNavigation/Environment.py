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
import time


class Environment(object):

    def __init__(self):

        print("init Environment")

        self.crossing_collision_dict = dict()  # 路口字典

    def clean_value(self):

        self.crossing_collision_dict.clear()

    def update(self, name, crossing, go_time):  # 更新环境状态

        #print("------update------!!!")



        result = sorted(crossing.items(), key=lambda d: int(d[0]), reverse=False)


        for i in range(len(result)-1):

            x1, y1 = self.__get_x_y(result[i][1])
            x2, y2 = self.__get_x_y(result[i+1][1])

            direction = 'direction' + str(self.__get_direction(x1, y1, x2, y2))

            if result[i+1][1] in self.crossing_collision_dict.keys():

                self.crossing_collision_dict[result[i + 1][1]].update(name, direction, result[i + 1][0], go_time[i])

            else:
                self.crossing_collision_dict[result[i+1][1]] = Crossing(result[i+1][1])

                self.crossing_collision_dict[result[i + 1][1]].update(name, direction, result[i+1][0], go_time[i])

    def get_split_path(self, agent_name, crossing, go_time):

        print("------------get_split_path------------")

        result = sorted(crossing.items(), key=lambda d: int(d[0]), reverse=False)

        split_path = []

        for i in range(len(result) - 1):

            x1, y1 = self.__get_x_y(result[i][1])
            x2, y2 = self.__get_x_y(result[i + 1][1])

            split_edage = self.__get_split_edage(x1, y1, x2, y2)

            direction = 'direction' + str(self.__get_direction(x1, y1, x2, y2))

            wait_point_index = self.crossing_collision_dict[result[i + 1][1]].get_waiting_queue_index(direction, result[i + 1][0], agent_name)

            times = int(go_time[i])-int(result[i + 1][0])

            ffff = split_edage[len(split_edage)-wait_point_index-1]

            for x in range(times):
                split_edage.insert(len(split_edage)-wait_point_index-1, ffff)

            if i<len(result) - 2:
                split_edage.pop()

            split_path.extend(split_edage)
        # goal_x, goal_y = self.__get_x_y(result[-1][1])
        # split_path.append((goal_x, goal_y))
        return split_path

    def get_corrsing_index(self, c_id, name, t):

        pass

    def get_wait_time(self):

        pass

    def get_agent_path(self, agent_name):

        pass

    def is_in_crossing_collision_dict(self,name):

        if name in self.crossing_collision_dict.keys():
            return True
        else:
            return False

    def get_go_time(self, name, t):    # 获取通过时间

        if name in self.crossing_collision_dict.keys():

            return self.crossing_collision_dict[name].get_go_time(t)

        else:
            return t

    def get_waiting_queue(self, name, direction, t):  # 获取等待队列的长度

        if name in self.crossing_collision_dict.keys():

            return self.crossing_collision_dict[name].get_waiting_queue(direction, t)

        else:
            return t

    def __get_direction(self, x1, y1, x2, y2):



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

    def print_crossing_collision_dict(self):

        for k, v in self.crossing_collision_dict.items():
            print("crossing: ", k)
            print("waiting_queue: ", v.print_direction())

class Crossing(object):

    def __init__(self, name):

        self.name = name
        self.waiting_queue0 = dict()  # 等待队列
        self.waiting_queue1 = dict()
        self.waiting_queue2 = dict()
        self.waiting_queue3 = dict()

        self.go_time_queue = dict()  # 通行时间

    def update(self, name, direction, t, go_time):

        t1 = int(t)

        go_time1 = int(go_time)

        self.go_time_queue[go_time] = name

        if t == go_time:
            return

        if int(direction[-1]) == 0:

            for x in range(t1, go_time1, 1):
                if x in self.waiting_queue0.keys():
                    self.waiting_queue0[str(x)].append(name)
                else:
                    self.waiting_queue0[str(x)] = [name]

        if int(direction[-1]) == 1:
            for x in range(t1, go_time1 , 1):
                if x in self.waiting_queue1.keys():
                    self.waiting_queue1[str(x)].append(name)
                else:
                    self.waiting_queue1[str(x)] = [name]
        if int(direction[-1]) == 2:
            for x in range(t1, go_time1 , 1):
                if x in self.waiting_queue2.keys():
                    self.waiting_queue2[str(x)].append(name)
                else:
                    self.waiting_queue2[str(x)] = [name]
        if int(direction[-1]) == 3:
            for x in range(t1, go_time1 , 1):
                if x in self.waiting_queue3.keys():
                    self.waiting_queue3[str(x)].append(name)
                else:
                    self.waiting_queue3[str(x)] = [name]

    def get_go_time(self, t):

        # print("self.go_time_queue:",type(self.go_time_queue.keys()))
        # print("self.go_time_queue:", self.go_time_queue.keys())

        t_list = sorted(self.go_time_queue.keys(), key=lambda d: int(d), reverse=False)

        flag = None
        t1 = str(t)

        while flag is None:
            try:
                t_list.index(t1)
                t1 = str(int(t1) + 1)
            except:
                flag = 0
        return int(t1)

    def get_waiting_queue(self, direction, t):
        if int(direction[-1]) == 0:
            return self.get_queue_length(self.waiting_queue0, t)
        if int(direction[-1]) == 1:
            return self.get_queue_length(self.waiting_queue1, t)
        if int(direction[-1]) == 2:
            return self.get_queue_length(self.waiting_queue2, t)
        if int(direction[-1]) == 3:
            return self.get_queue_length(self.waiting_queue3, t)

    def get_queue_length(self, d, t):

        if t in d.keys():
            return len(d[t])
        else:
            return 0

    def get_waiting_queue_index(self, direction, t, agent_name):

        if int(direction[-1]) == 0:
            return self.__get_queue_index(self.waiting_queue0, t, agent_name)
        if int(direction[-1]) == 1:
            return self.__get_queue_index(self.waiting_queue1, t, agent_name)
        if int(direction[-1]) == 2:
            return self.__get_queue_index(self.waiting_queue2, t, agent_name)
        if int(direction[-1]) == 3:
            return self.__get_queue_index(self.waiting_queue3, t, agent_name)

    def __get_queue_index(self, d, t, name):
        #print(" d: ", d)
        if t in d:
            return d[t].index(name)+1
        else:
            return 0

    def print_direction(self):

        print("waiting_queue0: ", self.waiting_queue0)
        print("waiting_queue1: ", self.waiting_queue1)
        print("waiting_queue2: ", self.waiting_queue2)
        print("waiting_queue3: ", self.waiting_queue3)
        print("go_time_queue: : ", self.go_time_queue)
















