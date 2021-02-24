"""

AStar search

author: Ashwin Bose (@atb033)

"""

class AStar():
    def __init__(self, env):
        self.agent_dict = env.agent_dict
        self.admissible_heuristic = env.admissible_heuristic
        self.is_at_goal = env.is_at_goal
        self.get_neighbors = env.get_neighbors

    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from.keys():
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]

    def search(self, agent_name):   # ａ*搜索算法
        """
        low level search 
        """
        initial_state = self.agent_dict[agent_name]["start"]  # 获取起点
        #print(agent_name)
        #print(initial_state)

        step_cost = 1
        
        closed_set = set()
        open_set = {initial_state}

        came_from = {}

        g_score = {} 
        g_score[initial_state] = 0

        f_score = {} 

        f_score[initial_state] = self.admissible_heuristic(initial_state, agent_name)  # 当前位置得分

        while open_set:
            temp_dict = {open_item: f_score.setdefault(open_item, float("inf")) for open_item in open_set} # f_score遍历f_score
            current = min(temp_dict, key=temp_dict.get)   # 找到下一步的最小l值

            if self.is_at_goal(current, agent_name):  # 判断是否到达目标点
                return self.reconstruct_path(came_from, current)

            open_set -= {current}     # 可走区域
            closed_set |= {current}   # 路过的网格

            neighbor_list = self.get_neighbors(current)  #获取临近的网格,包括上次碰撞的情况

            for neighbor in neighbor_list:
                if neighbor in closed_set:  # 走过的路径
                    continue
                
                tentative_g_score = g_score.setdefault(current, float("inf")) + step_cost  # 步长得分

                if neighbor not in open_set:  # 可达区域
                    open_set |= {neighbor}
                elif tentative_g_score >= g_score.setdefault(neighbor, float("inf")):  # inf 无穷大 # 更新得分
                    continue

                came_from[neighbor] = current

                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + self.admissible_heuristic(neighbor, agent_name)
        return False

