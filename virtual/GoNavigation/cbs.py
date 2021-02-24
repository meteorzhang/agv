"""

Python implementation of Conflict-based search

author: Ashwin Bose (@atb033)

"""
import yaml
from math import fabs
from itertools import combinations
from copy import deepcopy

from virtual.cbs.a_star import AStar
import numpy as np
import time
import random

import eventlet
eventlet.monkey_patch()

class Location(object):
    def __init__(self, x=-1, y=-1):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return str((self.x, self.y))


class State(object):
    def __init__(self, time, location):
        self.time = time
        self.location = location
        self.state_ = -1
        self.battery = 1000

    def __eq__(self, other):
        return self.time == other.time and self.location == other.location

    def __hash__(self):
        return hash(str(self.time)+str(self.location.x) + str(self.location.y))

    def is_equal_except_time(self, state):
        return self.location == state.location

    def __str__(self):
        return str((self.time, self.location.x, self.location.y ,self.state_,self.battery))


class Conflict(object):
    VERTEX = 1
    EDGE = 2

    def __init__(self):
        self.time = -1
        self.type = -1

        self.agent_1 = ''
        self.agent_2 = ''
    
        self.location_1 = Location()
        self.location_2 = Location()

    def __str__(self):
        return '(' + str(self.time) + ', ' + self.agent_1 + ', ' + self.agent_2 + \
             ', '+ str(self.location_1) + ', ' + str(self.location_2) + ')' 


class VertexConstraint(object):
    def __init__(self, time, location):
        self.time = time
        self.location = location

    def __eq__(self, other):
        return self.time == other.time and self.location == other.location

    def __hash__(self):
        return hash(str(self.time)+str(self.location))

    def __str__(self):
        return '(' + str(self.time) + ', '+ str(self.location) + ')' 


class EdgeConstraint(object):
    def __init__(self, time, location_1, location_2):
        self.time = time
        self.location_1 = location_1
        self.location_2 = location_2

    def __eq__(self, other):
        return self.time == other.time and self.location_1 == other.location_1 \
            and self.location_2 == other.location_2

    def __hash__(self):
        return hash(str(self.time) + str(self.location_1) + str(self.location_2))

    def __str__(self):
        return '(' + str(self.time) + ', '+ str(self.location_1) +', '+ str(self.location_2) + ')' 


class Constraints(object):  # 结构
    def __init__(self):
        self.vertex_constraints = set()
        self.edge_constraints = set()

    def add_constraint(self, other):
        self.vertex_constraints |= other.vertex_constraints
        self.edge_constraints |= other.edge_constraints

    def __str__(self):
        return "VC: " + str([str(vc) for vc in self.vertex_constraints])  + \
            "EC: " + str([str(ec) for ec in self.edge_constraints])


class Environment(object):
    def __init__(self, dimension, agents, obstacles):
        self.dimension = dimension
        self.obstacles = obstacles

        self.agents = agents
        self.agent_dict = {}

        self.constraints = Constraints()
        self.constraint_dict = {}  # 碰撞的情况

        self.a_star = AStar(self)

    def get_neighbors(self, state):
        neighbors = []
        
        # Wait action
        n = State(state.time + 1, state.location)
        if self.state_valid(n):
            neighbors.append(n)
        # Up action
        n = State(state.time + 1, Location(state.location.x, state.location.y+1))
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)
        # Down action
        n = State(state.time + 1, Location(state.location.x, state.location.y-1))
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)
        # Left action
        n = State(state.time + 1, Location(state.location.x-1, state.location.y))
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)
        # Right action
        n = State(state.time + 1, Location(state.location.x+1, state.location.y))
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)
        return neighbors

    def get_first_conflict(self, solution):
        max_t = max([len(plan) for plan in solution.values()])  # 最长路径
        result = Conflict()
        for t in range(max_t):
            for agent_1, agent_2 in combinations(solution.keys(), 2):  # 排列组合,两两顶点碰撞检测
                state_1 = self.get_state(agent_1, solution, t)  # 返回此时状态
                state_2 = self.get_state(agent_2, solution, t)  # 返回此时状态
                if state_1.is_equal_except_time(state_2):  # 位置相同
                    result.time = t
                    result.type = Conflict.VERTEX
                    result.location_1 = state_1.location
                    result.agent_1 = agent_1
                    result.agent_2 = agent_2
                    return result

            for agent_1, agent_2 in combinations(solution.keys(), 2):  # 两两边进行碰撞检测,相向而行
                state_1a = self.get_state(agent_1, solution, t)
                state_1b = self.get_state(agent_1, solution, t+1)

                state_2a = self.get_state(agent_2, solution, t)
                state_2b = self.get_state(agent_2, solution, t+1)

                if state_1a.is_equal_except_time(state_2b) and state_1b.is_equal_except_time(state_2a):
                    result.time = t
                    result.type = Conflict.EDGE
                    result.agent_1 = agent_1
                    result.agent_2 = agent_2
                    result.location_1 = state_1a.location
                    result.location_2 = state_1b.location
                    return result                
        return False

    def create_constraints_from_conflict(self, conflict):   #
        constraint_dict = {}
        if conflict.type == Conflict.VERTEX:  # 碰撞的是点
            v_constraint = VertexConstraint(conflict.time, conflict.location_1)
            constraint = Constraints()
            constraint.vertex_constraints |= {v_constraint}
            constraint_dict[conflict.agent_1] = constraint
            constraint_dict[conflict.agent_2] = constraint
        
        elif conflict.type == Conflict.EDGE:  # 碰撞的是边
            constraint1 = Constraints()
            constraint2 = Constraints()

            e_constraint1 = EdgeConstraint(conflict.time, conflict.location_1, conflict.location_2)
            e_constraint2 = EdgeConstraint(conflict.time, conflict.location_2, conflict.location_1)
        
            constraint1.edge_constraints |= {e_constraint1}
            constraint2.edge_constraints |= {e_constraint2}

            constraint_dict[conflict.agent_1] = constraint1
            constraint_dict[conflict.agent_2] = constraint2

        return constraint_dict

    def get_state(self, agent_name, solution, t):
        if t < len(solution[agent_name]):
            return solution[agent_name][t]
        else:
            return solution[agent_name][-1]

    def state_valid(self, state):
        # 判断坐标位于范围内,并且不在碰撞边和
        return state.location.x >= 0 and state.location.x < self.dimension[0] \
            and state.location.y >= 0 and state.location.y < self.dimension[1] \
            and VertexConstraint(state.time, state.location) not in self.constraints.vertex_constraints \
            and (state.location.x, state.location.y) not in self.obstacles

    def transition_valid(self, state_1, state_2):
        return EdgeConstraint(state_1.time, state_1.location, state_2.location) not in self.constraints.edge_constraints

    def is_solution(self, agent_name):
        pass

    def admissible_heuristic(self, state, agent_name):
        goal = self.agent_dict[agent_name]["goal"]

        # print("goal.location.x: ",goal.location.x)
        # print("goal.location.y: ", goal.location.y)
        return fabs(state.location.x - goal.location.x) + fabs(state.location.y - goal.location.y)


    def is_at_goal(self, state, agent_name):
        goal_state = self.agent_dict[agent_name]["goal"]
        return state.is_equal_except_time(goal_state)

    def make_agent_dict(self):
        self.agent_dict.clear()
        for agent in self.agents:
            start_state = State(0, Location(agent['start'][0], agent['start'][1]))
            goal_state = State(0, Location(agent['goal'][0], agent['goal'][1]))
            
            self.agent_dict.update({agent['name']:{'start': start_state, 'goal': goal_state}})

    def compute_solution(self, split_solution,goal_to_goal):  # 路径规划
        solution = {}
        print("self.agent_dict.keys(): ", len(self.agent_dict))
        for agent in self.agent_dict.keys():
            self.constraints = self.constraint_dict.setdefault(agent, Constraints())# 改变环境为自身的环境.记录碰撞情况
            start_time = time.time()
            name = str(self.agent_dict[agent]["start"].location.x) + "-" + str(self.agent_dict[agent]["start"].location.y) + "-" + str(self.agent_dict[agent]["goal"].location.x) + "-" + str(self.agent_dict[agent]["goal"].location.y)

            if self.agent_dict[agent]["start"].location.x==self.agent_dict[agent]["goal"].location.x and self.agent_dict[agent]["start"].location.y==self.agent_dict[agent]["goal"].location.y:
                continue
            print("name: ",name)
            local_solution = goal_to_goal[name]


            #local_solution = self.a_star.search(agent)  # 利用自身的环境进行A＊搜索算法
            print("cost_time: ", time.time()-start_time)
            #print(agent)
            #print(local_solution)
            if not local_solution:
                return False
            solution.update({agent: local_solution})  # 存入路径字典中
        if split_solution!=None:
            solution.update(self.change_split_solution(split_solution))
        return solution

    def compute_solution_a_start(self):  # 路径规划
        solution = {}
        print("self.agent_dict.keys(): ", len(self.agent_dict))
        for agent in self.agent_dict.keys():
            self.constraints = self.constraint_dict.setdefault(agent, Constraints())# 改变环境为自身的环境.记录碰撞情况
            start_time = time.time()
            local_solution = self.a_star.search(agent)  # 利用自身的环境进行A＊搜索算法
            print("cost_time: ", time.time()-start_time)




            #print(agent)
            #print(local_solution)
            if not local_solution:
                return False
            solution.update({agent: local_solution})  # 存入路径字典中
        return solution


    def change_split_solution(self,split_solution):

        new_split_solution = dict()

        for (k, v) in split_solution.items():
            new_v = []
            for i, x in enumerate(v):
                new_v.append(State(i, Location(x['x'], x['y'])))
            new_split_solution[k] = new_v

        return new_split_solution





    def compute_solution_cost(self, solution):
        return sum([len(path) for path in solution.values()])

class HighLevelNode(object):
    def __init__(self):
        self.solution = {}
        self.constraint_dict = {}  # 碰撞结构
        self.cost = 0

    def __lt__(self, other):
        return self.cost < other.cost

class CBS(object):
    def __init__(self, environment):
        self.env = environment 
        self.open_set = set()   # 可达区域

    def search_a_start(self, agents):

        self.env.agents = agents
        self.env.make_agent_dict()

        start1 = time.time()

        solution = self.env.compute_solution_a_start()  # 分别进行Ａ＊搜索算法？？？

        end = time.time()

        print("A*: ", end - start1)

        return self.generate_plan(solution)

    def search(self, agents, split_solution,goal_to_goal):

        self.env.agents = agents
        self.env.make_agent_dict()

        start = HighLevelNode()
        # TODO: Initialize it in a better way
        #start.constraint_dict = {}
        # for agent in self.env.agent_dict.keys():  #遍历键
        #     start.constraint_dict[agent] = Constraints()  #

        start1 = time.time()

        start.solution = self.env.compute_solution(split_solution,goal_to_goal)# 分别进行Ａ＊搜索算法？？？

        end = time.time()

        print("A*: ", end-start1)

        #for s in start.solution:
            #print(start.solution.setdefault(s, None))
        #print("*************-------------------*******************")

        if not start.solution:
            return {}
        start.cost = self.env.compute_solution_cost(start.solution)  # 计算总体花费
        start.cost = 0

        self.open_set |= {start}

        #print("------------**********////////////55555555555")
        times = 0
        while self.open_set:
            #print("times: ", times)
            if times > 10:
                return None
            times += 1
            P = min(self.open_set)  # 应该是返回的cost 最小的解决方案
            #print(P)
            self.open_set -= {P}  #

            self.env.constraint_dict = P.constraint_dict  # 上次的环境结构

            # 1.对路径进行修正，采取靠右碰撞的原则

            #print(type(P.solution))
            #print(P.solution)

            conflict_dict = self.env.get_first_conflict(P.solution)  # 进行两两碰撞检测，顶点和边

            cishu = 0

            # print("conflict_dict: ", conflict_dict)
            #
            # print("-------------/////////////////***************")
            while conflict_dict:

                # 检测是否路口碰撞

                #print("cishu: ", cishu)

                path_1, path_2 = self.amend_path(conflict_dict, P.solution[conflict_dict.agent_1], P.solution[conflict_dict.agent_2])
                P.solution[conflict_dict.agent_1] = path_1
                P.solution[conflict_dict.agent_2] = path_2
                conflict_dict = self.env.get_first_conflict(P.solution)  # 进行两两碰撞检测，顶点和边

            # 2.对路径进行碰撞检测


            if not conflict_dict:
                print("solution found")

                return self.generate_plan(P.solution)
            # 进行路径修正，实现避障，此时区分边碰撞和点碰撞，点碰撞实现
            # 1.点碰撞，两辆同时壁障
            # 更改搜索空间的过程
            #print(conflict_dict)  # 打印碰撞情况

            constraint_dict = self.env.create_constraints_from_conflict(conflict_dict)  # 构建碰撞结构体,



            for agent in constraint_dict.keys():
                new_node = deepcopy(P)
                new_node.constraint_dict[agent].add_constraint(constraint_dict[agent])  # 更改碰撞空间
                
                self.env.constraint_dict = new_node.constraint_dict  # 添加到环境中
                new_node.solution = self.env.compute_solution()  #
                if not new_node.solution:
                    continue
                new_node.cost = self.env.compute_solution_cost(new_node.solution) # 计算整体花费

                # TODO: ending condition

        return {}

    def amend_path(self, constraint_dict, path1, path2):

        if constraint_dict.type==1:
           return self.local_palnning_1(constraint_dict.time, path1), self.local_palnning_1(constraint_dict.time, path2)
           #return path1, path2


        if constraint_dict.type==2:
           return self.local_palnning_2(constraint_dict.time, path1), self.local_palnning_2(constraint_dict.time, path2)

    def local_palnning_1(self, t, path):
        new_path = []
        index = 0

        for i,x in enumerate(path):
            #print("index: ", i + index)
            if i==t:
                index = 2
                if path[t-1].location.x == path[t].location.x:
                   if path[t - 1].location.y > path[t].location.y:
                       new_path.append(State(t, Location(path[t-1].location.x+1,path[t-1].location.y)))
                       new_path.append(State(t+1, Location(path[t - 1].location.x + 1, path[t - 1].location.y-1)))
                       new_path.append(State(t + 2, Location(path[t - 1].location.x + 1, path[t - 1].location.y - 2)))


                   elif path[t - 1].location.y <= path[t].location.y:

                        new_path.append(State(t, Location(path[t-1].location.x-1,path[t-1].location.y)))
                        new_path.append(State(t+1, Location(path[t - 1].location.x-1, path[t - 1].location.y+1)))
                        new_path.append(State(t + 2, Location(path[t - 1].location.x - 1, path[t - 1].location.y + 2)))


                elif path[t-1].location.y == path[t].location.y:
                   if path[t - 1].location.x > path[t].location.x:

                       new_path.append(State(t, Location(path[t-1].location.x,path[t-1].location.y-1)))
                       new_path.append(State(t+1, Location(path[t - 1].location.x -1, path[t - 1].location.y-1)))
                       new_path.append(State(t + 2, Location(path[t - 1].location.x - 2, path[t - 1].location.y - 1)))

                   elif path[t - 1].location.x <= path[t].location.x:

                        new_path.append(State(t, Location(path[t-1].location.x,path[t-1].location.y+1)))

                        new_path.append(State(t+1, Location(path[t - 1].location.x + 1, path[t - 1].location.y+1)))
                        new_path.append(State(t + 2, Location(path[t - 1].location.x + 2, path[t - 1].location.y + 1)))

            else:
                new_path.append(State(i+index, Location(x.location.x, x.location.y)))
        return new_path

    def local_palnning_2(self, t, path):
        new_path = []
        index = 0

        for i,x in enumerate(path):
            #print("index: ", i + index)
            if i == t:
                index = 2
                if path[t-1].location.x == path[t].location.x:
                   if path[t - 1].location.y > path[t].location.y:
                       new_path.append(State(t, Location(path[t-1].location.x+1,path[t-1].location.y)))
                       new_path.append(State(t+1, Location(path[t - 1].location.x + 1, path[t - 1].location.y-1)))

                       new_path.append(State(t + 2, Location(path[t - 1].location.x + 1, path[t - 1].location.y - 2)))


                   elif path[t - 1].location.y <= path[t].location.y:

                        new_path.append(State(t, Location(path[t-1].location.x-1,path[t-1].location.y)))
                        new_path.append(State(t+1, Location(path[t - 1].location.x-1, path[t - 1].location.y+1)))

                        new_path.append(State(t + 2, Location(path[t - 1].location.x - 1, path[t - 1].location.y + 2)))


                elif path[t-1].location.y == path[t].location.y:
                   if path[t - 1].location.x > path[t].location.x:

                       new_path.append(State(t, Location(path[t-1].location.x,path[t-1].location.y-1)))
                       new_path.append(State(t+1, Location(path[t - 1].location.x -1, path[t - 1].location.y-1)))

                       new_path.append(State(t + 2, Location(path[t - 1].location.x - 2, path[t - 1].location.y - 1)))

                   elif path[t - 1].location.x <= path[t].location.x:

                        new_path.append(State(t, Location(path[t-1].location.x,path[t-1].location.y+1)))

                        new_path.append(State(t+1, Location(path[t - 1].location.x + 1, path[t - 1].location.y+1)))

                        new_path.append(State(t + 2, Location(path[t - 1].location.x + 2, path[t - 1].location.y + 1)))

            else:
                if i!=t+1:
                    new_path.append(State(t+index, Location(x.location.x, x.location.y)))
        return new_path

    def generate_plan(self, solution):
        plan = {}
        for agent, path in solution.items():
            path_dict_list = [{'t':state.time, 'x':state.location.x, 'y':state.location.y,} for i,state in enumerate(path)]
            plan[agent] = path_dict_list
        return plan

def load_goal_list(path):
    goal_list = []
    with open(path, "r") as f:
        n = int(f.readline())
        for x in range(n):
            px, py = map(int, f.readline().split())
            goal_list.append(((px - 5) // 10, (py - 5) // 10))
    return goal_list


def random_agent(n, no_obstacles,goal_list):
    agent_xy = set()
    while len(agent_xy) < n:
        agent_xy.add(random.randint(0, len(no_obstacles)))
    agent_xy = list(agent_xy)

    agent_goal = set()
    while len(agent_goal) < n:
        agent_goal.add(random.randint(0, len(goal_list)))
    agent_goal = list(agent_goal)

    agents=[]

    for x in range(n):
        a = dict()
        p = []
        p.append(no_obstacles[agent_xy[x]][0])
        p.append(no_obstacles[agent_xy[x]][1])
        a['start'] = p.copy()
        p.clear()

        p.append(no_obstacles[agent_goal[x]][0])
        p.append(no_obstacles[agent_goal[x]][1])
        a['goal'] = p.copy()
        p.clear()

        a['name'] = "agent" + str(x)

        agents.append(a.copy())
        a.clear()

    return agents




def load_no_obstacles(path):
    no_obstacles = []
    dimension = []
    x_max = 0
    y_max = 0
    with open(path, "r") as f:
        n = int(f.readline())
        for x in range(n):
            px, py = map(int, f.readline().split())
            x_max = max((px-5)//10,x_max)
            y_max = max((py-5)//10,y_max)
            no_obstacles.append(((px-5)//10, (py-5)//10))

    no_obstacles = list(set(no_obstacles))

    dimension.append(x_max+1)
    dimension.append(y_max+1)
    return no_obstacles, dimension


def change_to_no_obstacles(no_obstacles, dimension):
    obstacles = []
    for i in range(dimension[0]):
        for j in range(dimension[1]):
            if (i,j) not in no_obstacles:
                obstacles.append((i, j))

    return obstacles

def main():  #
    # parser = argparse.ArgumentParser() # 解析参数类
    # parser.add_argument("param", help="input file containing map and obstacles")
    # parser.add_argument("output", help="output file with the schedule")
    # args = parser.parse_args()
    #
    # # Read from input file
    # with open(args.param, 'r') as param_file:
    #     try:
    #         param = yaml.load(param_file)
    #     except yaml.YAMLError as exc:
    #         print(exc)
    #
    # dimension = param["map"]["dimensions"]  # 地图尺寸
    # print(type(dimension[0]))
    # print(dimension)
    # obstacles = param["map"]["obstacles"]   # 障碍物
    # print(type(obstacles[0]))
    # print(obstacles)
    # agents = param['agents']  # agent　信息
    # print(type(agents[0]))
    # print(agents)

    no_obstacles, dimension = load_no_obstacles("./final/tongxing_cbs.txt")

    obstacles = change_to_no_obstacles(no_obstacles, dimension)

    np.random.seed(np.int32(time.time() % 1 * 1000 + int(10) * 10))  # 随机数种子

    goal_list = load_goal_list("./final/goal_list.txt")
    flag = False

    while not flag:
        with eventlet.Timeout(2,False):
            agents = random_agent(3, no_obstacles,goal_list)

    #print("dimension:", dimension)
    #print("no_obstacles", len(no_obstacles))
    #print("obstacles", len(obstacles))
    #print(agents)
            env = Environment(dimension, agents, obstacles)  # 　构造环境

    # Searching
            cbs = CBS(env)
            solution = cbs.search()   # 多远路径搜索算法
            flag = True

    print("solution: ", solution)
    if not solution:
        print(" Solution not found" ) 
        return



    # Write to output file
    with open("output.yaml", 'r') as output_yaml:
        try:
            output = yaml.load(output_yaml)
        except yaml.YAMLError as exc:
            print(exc)

    output["schedule"] = solution
    output["cost"] = env.compute_solution_cost(solution)
    with open("output.yaml", 'w') as output_yaml:
        yaml.safe_dump(output, output_yaml)  
        
    
if __name__ == "__main__":
    main()
