import time
import _thread
import numpy as np
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from virtual.work.roadnet_unit import MNode, MLine
from virtual.monitor.monitoring_unit import Agv_Car
import threading


class Thread1(QThread):
    def __init__(self, map_label, car_list, navigation, way):
        super().__init__()
        self.navigation = navigation
        self.map_label = map_label
        self.__car_list = car_list
        self.enable_run = True
        self.ai_enable_run = True
        self.solution = dict()
        self.now_solution = False
        self.way = way
        self.new_agent = []
        self.cars_len = len(car_list)
        self.point_index = [0 for i in range(self.cars_len)]
        self.cbs_number = 0
        self.start_cbs = False

    def set_enable(self,able):
        self.enable_run = able
    def set_AIenable(self,able):
        self.ai_enable_run = able

    def set_solution(self, all_solution):
        self.new_agent.clear()
        for solution in all_solution:
            path11 = solution['path']
            name11 = solution['name']
            self.solution[name11] = path11
            self.new_agent.append(solution['name'])

        for xxx in self.new_agent:
            self.point_index[int(xxx)] = 0
        self.now_solution = True

    def check_point(self,a,x,y,old_car):
        for i,car in enumerate(old_car):
            if i==a:
                continue
            car_x = old_car[i][0]
            car_y = old_car[i][1]
            if abs(x-car_x)<1 and abs(y-car_y)<1:
               if self.__car_list[i].get_startTime() !=0.0:
                self.cbs_number += 1

                print("碰撞了！！！！--次数：", self.cbs_number)
                return False
        return True

    def check_end(self):
        for i,x in enumerate(self.point_index):
            if str(i) in self.solution and x < len(self.solution[str(i)]):
                return True
        return False

    def set_start_cbs(self,flag):
        self.start_cbs = flag
    def set_cbs(self,num):
        self.cbs_number = num

    def get_cbs(self):
        return self.cbs_number


    def run(self):

        last_status = None
        while True:
         if self.way == 0 and self.ai_enable_run:
            all_status = self.navigation.get_all_state()
            print(all_status)
            sta_length = len(all_status)
            # for i in range(sta_length):
            #     if last_status != None and (all_status[i][0] != last_status[i][0] or all_status[i][1] != last_status[i][1] or all_status[i][2] != last_status[i][2]):
            #         self.__car_list[i].set_location_angle(QPoint(all_status[i][0] * 20, 910 - all_status[i][1] * 20), 0)
            #         if self.__car_list[i].get_status != 1 :
            #             self.__car_list[i].set_status(all_status[i][2])
            #
            # self.map_label.get_car_list(self.__car_list)
            # time.sleep(0.01)
            # last_status = all_status

            for h in range(1, 11):
                for i in range(sta_length):
                    if last_status!=None and (all_status[i][0]!=last_status[i][0] or all_status[i][1]!=last_status[i][1] or all_status[i][2]!=last_status[i][2]):

                        if 915-all_status[i][1]*20 == 915-last_status[i][1]*20 and all_status[i][0]*20 > last_status[i][0]*20:
                            z = (all_status[i][0]*20 - last_status[i][0]*20)/10
                            self.__car_list[i].set_location_angle(QPoint(last_status[i][0]*20 + h*z, 915-all_status[i][1]*20), 0)
                        elif 915-all_status[i][1]*20 == 915-last_status[i][1]*20 and all_status[i][0]*20 < last_status[i][0]*20:
                            z = (all_status[i][0]*20 - last_status[i][0]*20)/10
                            self.__car_list[i].set_location_angle(QPoint(last_status[i][0]*20 + h*z, 915-all_status[i][1]*20), 0)
                        elif all_status[i][0]*20==last_status[i][0]*20 and 915-all_status[i][1]*20>915-last_status[i][1]*20:
                            z = ((915-all_status[i][1]*20) - (915-last_status[i][1]*20))/10
                            self.__car_list[i].set_location_angle(QPoint(all_status[i][0] * 20, (915-last_status[i][1]*20)+h*z), 0)
                        elif all_status[i][0]*20==last_status[i][0]*20 and 915-all_status[i][1]*20<915-last_status[i][1]*20:
                            z = ((915-all_status[i][1]*20) - (915-last_status[i][1]*20))/10
                            self.__car_list[i].set_location_angle(QPoint(all_status[i][0]*20, (915- last_status[i][1]*20)+h*z), 0)

                        if self.__car_list[i].get_status != 1 and  h == 10:
                            self.__car_list[i].set_status(all_status[i][2])
                time.sleep(0.035)
                self.map_label.get_car_list(self.__car_list)



            last_status = all_status


         else:



    # def run(self):
    #     while (self.enable_run):
            while self.enable_run and len(self.solution) > 0 and self.check_end():
                car_sta = [0 for i in range(self.cars_len)]
                old_car = np.zeros(shape=(self.cars_len, 2))
                for i in range(self.cars_len):
                    old_car[i] = [self.__car_list[i].x, self.__car_list[i].y]
                for h in range(1, 11):
                    for i in range(self.cars_len):
                        if self.enable_run and str(i) in self.solution and self.point_index[i] == len(self.solution[str(i)])-1 and self.__car_list[i].get_isbind()==1:
                            car_sta[i] = 1
                            self.solution[str(i)].clear()
                            continue
                        if self.enable_run and str(i) in self.solution and self.point_index[i]+1<len(self.solution[str(i)]):
                            x_old = self.solution[str(i)][self.point_index[i]][0]
                            y_old = self.solution[str(i)][self.point_index[i]][1]
                            x = self.solution[str(i)][self.point_index[i]+1][0]
                            y = self.solution[str(i)][self.point_index[i]+1][1]
                            if self.check_point(i,x,y,old_car):
                                old_car[i]=[x,y]
                                if y == y_old and x>x_old:
                                    self.__car_list[i].set_location_angle(QPoint(x_old+h, y), 0)
                                elif y == y_old and x<x_old:
                                    self.__car_list[i].set_location_angle(QPoint(x_old - h, y), 0)
                                elif x==x_old and y>y_old:
                                    self.__car_list[i].set_location_angle(QPoint(x, y_old+h), 0)
                                elif x==x_old and y<y_old:
                                    self.__car_list[i].set_location_angle(QPoint(x, y_old - h), 0)
                                if h==10:
                                    self.point_index[i] += 1
                    time.sleep(0.035)
                    #print("--------界面刷新------")
                    self.map_label.get_car_list(self.__car_list)
                for i in range(self.cars_len):
                    if car_sta[i] == 1:
                        self.__car_list[i].set_status(1)
            time.sleep(0.15)
