import time
import _thread
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from virtual.work.roadnet_unit import MNode, MLine
from virtual.monitor.monitoring_unit import Agv_Car
from virtual.navi.Navigator import Navigator
from virtual.cbs.Navigator_cbs import Navigator_cbs
import threading

class Thread1(QThread):

    def __init__(self,map_label):
        super().__init__()
        self.navigation = None
        self.map_label = map_label
        self.__car_list = []
        try:
            _thread.start_new_thread( self.init_navigation, ("Thread-1", 2, ) )
        except:
            print ("Error: 无法启动线程")

    def init_navigation(self,threadName, delay):
        print('init navi...')
        self.navigation = Navigator()  # 导航模块
        self.navigation.init_map_m(self)



    def run(self):

        self.__car_list = self.map_label.output_car_list()
        print("go to navigation!!!")  # TODO如果小车没有初始目标点会报错

        self.navigation.init_data(self.map_label.map_name, self.__car_list)  # 此处需要图片的路径
        while True:
            agent_dict = self.get_state()
            for (k, v) in agent_dict.items():
                print(k, v[0],v[1])
                self.__car_list[k].set_location_angle(QPoint(v[0]*20, v[1]*20), 0)
            time.sleep(0.2)
            self.map_label.get_car_list(self.__car_list)




        # self.__car_list.clear()
        # self.__car_list.append(Agv_Car())
        # self.__car_list.append(Agv_Car())
        # self.__car_list.append(Agv_Car())
        # self.__car_list[0].set_id(0)
        # self.__car_list[1].set_id(1)
        # self.__car_list[2].set_id(2)
        # i = 0
        # while i<200:
        #     self.__car_list[0].set_location_angle(QPoint(30 + 5 * i, 30 + 5 * i), 0)
        #     self.__car_list[1].set_location_angle(QPoint(60 + 5 * i, 60 + 5 * i), 0)
        #     self.__car_list[2].set_location_angle(QPoint(90 + 5 * i, 90 + 5 * i), 0)
        #     i = i+1
        #     if  i>198:
        #         i=0
        #     time.sleep(0.2)
        #     self.map_label.get_car_list(self.__car_list)

        

