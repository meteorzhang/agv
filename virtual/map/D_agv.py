import time
import _thread
from PyQt5.QtGui import *
from PyQt5.QtCore import *
class AgvRunThread(QThread):
    def __init__(self,car):
      self.car=car;
      self.enable_thread_run=True



    #设置路径点
    def setPath(self, list):
        pass

    #判断是否到达目标
    def is_don(self):
        pass

    #线程运行
    def run(self):
        while(self.enable_thread_run):
         if self.is_don()==True:
             self.car.taskStatus=Finished
             self.car.RunTastk=0
             self.enable_thread_run=False



         pass
