from queue import Queue
import numpy as np
import math
import json

class Point:
    def __init__(self, x, y, id, identifier, lock=0):
        self.x = x
        self.y = y
        self.id = id
        self.identifier = identifier
        self.next_p = []
        self.cost_p = []            # 路径花费
        self.use_p = []             # 路径锁
        self.lock = lock            # 节点锁


    def add(self, next):
        self.next_p.append(next)

    def out_string(self):
        s = str(self.id)+" (" + str(self.x)+","+str(self.x)+") "+ self.identifier + "  next: "
        for x in self.next_p:
            s += str(x)
            s += ","
        return s

class Graph():
    def __init__(self, map_data):

        print(map_data)

        map_json = self.read_json(map_data)
        print(map_data)
        self.path = []
        self.huo = []
        self.list = []
        num = map_json.get("number_p").get("number_p")
        for i in range(0, num):
            index = 'p' + str(i)
            x = map_json.get(index).get("point.x")
            y = map_json.get(index).get("point.y")
            identifier = map_json.get(index).get("id")
            p = Point(x, y, i, identifier)
            self.list.append(p)
        num = map_json.get("number_l").get("number_l")
        for i in range(0, num):
            index = 'l' + str(i)
            x = self.lookfor_identifier(map_json.get(index).get("start.id"))
            y = self.lookfor_identifier(map_json.get(index).get("end.id"))
            self.list[x].add(y)
            self.list[y].add(x)

        self.update()

    def computer_near_node(self,x,y):
         dis = 10000
         index = 0
         for i, po in enumerate(self.list):
            if math.sqrt(pow((po.x - x), 2) + pow((po.y - y), 2))< dis:
                dis = math.sqrt(pow((po.x - x), 2) + pow((po.y - y), 2))
                index = i
         return index

    def navigation_to_goal(self, x1, y1, x2, y2):
        s = self.computer_near_node(x1, y1)
        t = self.computer_near_node(x2, y2)
        self.spfa(s, t)
        lu = []
        for x in self.path:
            lu.append(self.list[x])
        return lu

    def show_list(self):
        for x in self.list:
            print(x.out_string())

    def read_json(self, path):
        print(path)
        with open(path, 'r+', encoding='utf-8') as f:
            map_json = json.load(f)
            return map_json

    def lookfor_identifier(self, identifier):
        for i, x in enumerate(self.list):
            if x.identifier == identifier:
                return i
        return None

    def update(self):
        for x in range(0, len(self.list)):
            for y in range(0, len(self.list[x].next_p)):
                t = self.list[x].next_p[y]
                c = math.sqrt(
                    pow((self.list[x].x - self.list[t].x), 2) + pow((self.list[x].y - self.list[t].y), 2))
                self.list[x].cost_p.append(c)
                self.list[x].use_p.append(1)

        for x in range(0, len(self.list)):
            for y in range(0, len(self.list[x].next_p)):
                t = self.list[x].next_p[y]
                c = math.sqrt(
                    pow((self.list[x].x - self.list[t].x), 2) + pow((self.list[x].y - self.list[t].y), 2))
                self.list[x].cost_p.append(c)
                self.list[x].use_p.append(1)

    def __getPoint(cx, cy, r, stx, sty, edx, edy):
        """
            status
             cx: 圆心x轴
             cy: 圆心y轴
             r: 半径
             stx: 直线起点x
             sty: 直线起点y
             edx: 直线终点x
             edy: 直线终点y
            """
        """
            roadnet_shape
             0 : 放置节点
             1 ：放置边
            """

        k = (edy - sty) / (edx - stx)
        b = edy - k * edx
        c = cx * cx + (b - cy) * (b - cy) - r * r
        a = (1 + k * k)
        b1 = (2 * cx - 2 * k * (b - cy))

        tmp = math.sqrt(b1 * b1 - 4 * a * c)
        x1 = (b1 + tmp) / (2 * a)
        y1 = k * x1 + b
        x2 = (b1 - tmp) / (2 * a)
        y2 = k * x2 + b
        res = (x1 - cx) * (x1 - cx) + (y1 - cy) * (y1 - cy)
        if res == r * r:
            x = x1
            y = y1
        else:
            x = x2
            y = y2
        return x, y

    def show_path(self):
        s = "path: "
        for x in self.path:
            s += str(x)
            s += ", "
        print(s)

    def spfa(self, s, t):
        que = Queue()
        dis = []
        pre = []
        vis = []
        for x in range(0, len(self.list)):
            dis.append(10000)
            vis.append(0)
            pre.append(0)
        que.put(s)
        vis[s] = 1
        dis[s] = 0
        while not que.empty():
            ttt = que.get()
            vis[ttt] = 0
            for x in range(0, len(self.list[ttt].next_p)):
                ne = self.list[ttt].next_p[x]
                if self.list[ne].lock == 0 and self.list[ttt].use_p[x] > 0:   # 节点锁和边的方向锁
                    if dis[ne] > dis[ttt] + self.list[ttt].cost_p[x]:
                        dis[ne] = dis[ttt] + self.list[ttt].cost_p[x]
                        pre[ne] = ttt
                        if vis[ne] == 0:
                            que.put(ne)
                            vis[ne] = 1
                else:
                    if self.list[ne].lock == 0 and self.list[ttt].use_p[x] > 0 and dis[ne] > dis[ttt] + self.list[ttt].cost_p[x]:
                        dis[ne] = dis[ttt] + self.list[ttt].cost_p[x]
                        pre[ne] = ttt
                        if vis[ne] == 0:
                            que.put(ne)
                            vis[ne] = 1
        st = []
        st.append(t)
        goal = t
        num = len(self.list)
        dd = 0
        while pre[goal] != s:
            st.append(pre[goal])
            goal = pre[goal]
            dd = dd + 1
            if dd > num:
                return False
        st.append(s)
        path1 = list(reversed(st))
        self.path = path1
        if s == t:
            self.path = []
        dist = dis[t]
        if dist >= 1000:
            return False
        return True

# g = Graph("F:/zhangfuqiang/application/res/map/brook.json")
# g.spfa(10, 28)
# g.show_path()
