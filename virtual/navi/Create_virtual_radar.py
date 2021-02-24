import sys
# sys.path.append('../')

from navi.GoConfig import GoConfig
from navi.GoStatus import AgentState

import numpy as np
import scipy.linalg as linalg
import math

class Virtual_radar():
    def __init__(self):

        self.lines = []
        self.width = None

        self.radar = []
        self.radar_x = []
        self.radar_y = []
        self.angular_resolution = 1*0.017444444444444446

        self.angle_min = 0
        self.angle_max = 6.28

        self.bias = 0.004



    def __init_radar(self):

        iteration = math.ceil((self.angle_max-self.angle_min)/self.angular_resolution)
        #print(iteration)

        for i in range(iteration):
            angle = self.angle_min + i * self.angular_resolution
            self.radar_x.append(math.cos(angle)*1-math.sin(angle)*0)
            self.radar_y.append(math.sin(angle)*1+math.cos(angle)*0)
            self.radar.append(self.width)

    def init_data(self, map_path):

        self.__load_map(map_path)

    def __load_map(self, map_path):

        print("load: "+map_path)

        with open(map_path, 'r') as f:
            self.height, self.width = map(int,(f.readline().split()))

            self.__init_radar()

            #print(self.radar_x)
            #print(self.radar_y)

            times = int(f.readline())
            for x in range(times):
                a, b, c, d = map(int, f.readline().split())
                y1 = self.width-a
                x1 = b
                y2 = self.width-c
                x2 = d
                self.lines.append([x1, y1, x2, y2])

    def get_lidar(self, x, y):
        x /= 0.05
        y /= 0.05
        for i, xx in enumerate(self.radar):
            re = self.cross_map(x, y, self.radar_x[i]*self.width+x, self.radar_y[i]*self.width+y)
            if re < self.radar[i]:
                self.radar[i] = re
        for i, d in enumerate(self.radar):
            self.radar[i] = d/self.width
        return self.radar

    def cross_map(self, x1, y1, x2, y2):

        min_range = 1000
        for x in self.lines:
            re = self.get_crossing([x1, y1, x2, y2], x)
            if re is not None:
                if min_range > re:
                    min_range = re
        return min_range

    def get_crossing(self, s1, s2):
        xa, ya = s1[0], s1[1]
        xb, yb = s1[2], s1[3]
        xc, yc = s2[0], s2[1]
        xd, yd = s2[2], s2[3]
        # 判断两条直线是否相交，矩阵行列式计算
        a = np.matrix(
            [
                [xb - xa, -(xd - xc)],
                [yb - ya, -(yd - yc)]
            ]
        )
        delta = np.linalg.det(a)
        # 不相交,返回两线段
        if np.fabs(delta) < 1e-6:
            #print(delta)
            return None
            # 求两个参数lambda和miu
        c = np.matrix(
            [
                [xc - xa, -(xd - xc)],
                [yc - ya, -(yd - yc)]
            ]
        )
        d = np.matrix(
            [
                [xb - xa, xc - xa],
                [yb - ya, yc - ya]
            ]
        )
        lamb = np.linalg.det(c) / delta
        miu = np.linalg.det(d) / delta
        # 相交
        if lamb <= 1 and lamb >= 0 and miu >= 0 and miu <= 1:
            x = xc + miu * (xd - xc)
            y = yc + miu * (yd - yc)
            return self.__get_point_dis(xa, ya,x,y)
        # 相交在延长线上
        else:
            return None

    def __get_point_dis(self,x1,y1,x2,y2):

        return math.sqrt((x1-x2)**2+(y1-y2)**2)







