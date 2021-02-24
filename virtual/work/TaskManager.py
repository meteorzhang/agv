from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QPoint, pyqtSignal
import random
import json
from virtual.monitor.monitoring_unit import Agv_Car
from virtual.work.roadnet_unit import MNode, MLine
import math
from threading import Thread
import time
import threading
from PyQt5.QtCore import *
# mnode1 = MNode(QPoint(555, 555),"red", 5,7, 0, None)
# mnode2 = MNode(QPoint(777, 777), "red", 5, 7, 0, None)
# self.map_label.set_light(mnode1,mnode2)
# self.map_label.clear_light()

from virtual.navi.Navigator import Navigator1
from virtual.GoNavigation.Navigator_cbs import Navigator



class TaskManager(object):

    def __init__(self, map_label,map_name):
        self.goal_point = []

        self.txt_name=''
        self.tasks = []
        self.powers = []
        self.agvs=None

        self.dialog = Dialog()
        self.look_dialog = LookDialog()
        self.dialog.dialogSignel.connect(self.slot_emit)
        self.look_dialog.dialogSignel.connect(self.look_emit)
        self.map_label = map_label

        self.navigation = None
        self.task_thread=None
        self.thread1 = None
        self.way = 1
        self.navi_astar = None
        self.total_data = []
        self.tasks_old = []
        self.isagain = 0
        self.task_dispatchThread = None
        self.filename = map_name[0:-4] + ".txt"
        print(self.filename)
        # try:
        #     # threads_navi = threading.Thread(target=self.init_naviga tion, args=("map", 2,))
        #     # threads_navi.start()
        #     self.init_navigation()
        #
        # except:
        #     print("Error: 无法启动线程")

    def init_navigation(self):


            self.navigation = Navigator1()  # 导航模块
            init_thread = Navi_thread(self.navigation, self.agvs, self.filename)
            init_thread.start()
            init_thread.wait()



            print("navi init----------")
            self.navi_astar = Navigator(self.filename)
            # self.navigation.load_goal_list("cbs/final/goal_list.txt")

    def init(self,dict, agvs,actionExecuteTask,actionCancelTask):



            self.actionExecuteTask = actionExecuteTask
            self.actionCancelTask = actionCancelTask
            load_dict = dict
            self.agvs = agvs

            self.init_navigation()

            #print(load_dict)
            #self.__road_json_path = self.filename
            #number_p = load_dict.get("number_p").get("number_p")
            number_goal = load_dict.get("number_goal").get("number_goal")
            number_power = load_dict.get("number_power").get("number_power")
            number_ku = load_dict.get("number_ku").get("number_ku")


            for i in range(0, number_goal):
                index = 'goal' + str(i)

                if i < 78:
                    self.goal_point.append(
                    MNode(QPoint(load_dict.get(index).get("point.x"), load_dict.get(index).get("point.y")),
                          load_dict.get(index).get("colour"), load_dict.get(index).get("thickness"),
                          load_dict.get(index).get("radius"), i, None))


                    self.dialog.goal_box.addItem("站"+str(i))
                    self.dialog.start_box.addItem("站" + str(i))
                if i == 98:
                    self.goal_point.append(
                    MNode(QPoint(load_dict.get(index).get("point.x"), load_dict.get(index).get("point.y")),
                          load_dict.get(index).get("colour"), load_dict.get(index).get("thickness"),
                          load_dict.get(index).get("radius"), 78, None))
                    self.dialog.goal_box.addItem("入库点")
                    self.dialog.start_box.addItem("入库点")
                if i == 99:
                    self.goal_point.append(
                    MNode(QPoint(load_dict.get(index).get("point.x"), load_dict.get(index).get("point.y")),
                          load_dict.get(index).get("colour"), load_dict.get(index).get("thickness"),
                          load_dict.get(index).get("radius"), 79, None))
                    self.dialog.goal_box.addItem("出库点")
                    self.dialog.start_box.addItem("出库点")

            self.power_point = []
            for i in range(0,number_power):
                index = 'power' + str(i)
                self.power_point.append(load_dict.get(index).get("point.x"))
                self.power_point.append(load_dict.get(index).get("point.y"))
                self.powers.append(self.power_point)

                self.power_point = []

            self.ku_point = []
            for i in range(0, number_ku):
                index = 'ku' + str(i)
                self.ku_point.append([load_dict.get(index).get("point.x"), load_dict.get(index).get("point.y")])

            self.map_label.get_car_list(self.agvs)
            self.update_task = udpate_task_thread(self.look_dialog,self.agvs)
            self.update_task.start()

    def Again(self):
        for m in self.tasks_old:
            task = Task()
            task.set_start(m.get_start())
            task.set_goal(m.get_goal())
            task.set_status(m.get_id())
            self.tasks.append(task)

        print("old_tasks_len:",len(self.tasks_old),len(self.tasks))
        print("again")
        self.isagain = 1
        self.tasks_old.clear()
        self.look_dialog.show_list(self.tasks)
    def Total(self):
        self.TotalDialog = TotalDialog()
        self.TotalDialog.show_data(self.total_data)
        self.TotalDialog.show()
        print("total",self.total_data)

    def selectMode(self,way):
        self.way = way
        #if way ==0:
            #self.thread1.set_enable(False)


    def get_collision(self):
        self.navigation.get_all_collision_number()


    def NewTask(self):
        self.dialog.show()

    def goal_txt(self):
        with open("goal_list.txt", 'w', encoding='utf-8') as f:
            f.write(str(len(self.goal_point)))
            f.write("\n")
            for p in self.goal_point:
                f.write(str(p.get_location().x()) + " " + str(p.get_location().y())+"\n")



    def slot_emit(self,resut,start,goal):
        #预览
        if resut == 0:
            self.map_label.clear_light()
            task = Task()
            task.set_start(self.goal_point[start])
            task.set_goal(self.goal_point[goal])
            mnode1 = self.goal_point[start]
            mnode2 = self.goal_point[goal]
            self.map_label.set_light(mnode1,mnode2)


        #取消预览
        elif resut ==1:
            self.map_label.clear_light()


        #生成任务
        elif resut ==2:
            self.tasks_old.clear()
            task = Task()
            task.set_start(self.goal_point[start])
            task.set_goal(self.goal_point[goal])
            print("任务设置的目标点：",self.goal_point[goal].get_point().x(),self.goal_point[goal].get_point().y())
            task.set_status(1)
            if len(self.tasks) == 0:
                task.set_id(0)
            else:
                id = self.tasks[len(self.tasks)-1].get_id()+1
                task.set_id(id)
            self.tasks.append(task)
            self.LookTask()
            self.map_label.clear_light()


    def look_emit(self,a,b,c):
        #删除某个任务
        if a ==0 :
            row_num = b
            del self.tasks[row_num]
            self.map_label.clear_light()
        #切换任务item
        if a ==1 and len(self.tasks)>0 :
            item = b
            self.map_label.clear_light()
            self.map_label.set_light(self.tasks[item].get_start(), self.tasks[item].get_goal())

        #退出任务窗口
        if a == 2 :
            self.map_label.clear_light()
    def LookTask(self):
        self.look_dialog.show_list(self.tasks)
        self.look_dialog.show()


    #执行任务
    def Execute_Task(self):
        print("线程开始执行")
        self.thread1 = Thread1(self.map_label, self.agvs, self.navigation, self.way)
        self.thread1.start()
        #self.thread1.exec_()
        time.sleep(0.5)
        # if self.task_thread != None:
        #
        #     self.task_thread.enable = False
        #     time.sleep(1)
        #     self.task_thread = None
        self.task_dispatchThread = Task_dispatchThread(self.tasks, self.agvs)
        self.task_dispatchThread.start()
        time.sleep(0.5)
        self.task_thread = TaskThread(self.tasks, self.agvs, self.map_label, self.thread1,self.navigation,self.navi_astar,self.way)
        self.task_thread.start()
        print("tasklen",len(self.tasks))
        for m in  self.tasks:

         self.tasks_old.append(m)
        print("oldtasklen",len(self.tasks_old))


    #加载配置
    def Pause_Task(self):
        self.tasks_old.clear()
        m = 77

        for i in range(20,0,-1):

            task = Task()
            task.set_start(self.goal_point[74])
            m-=1
            task.set_goal(self.goal_point[75])
            m-=1
            task.set_status(1)
            if len(self.tasks) == 0:
                task.set_id(0)
            else:
                id = self.tasks[len(self.tasks)-1].get_id()+1
                task.set_id(id)
            self.tasks.append(task)
            #self.LookTask()
            self.map_label.clear_light()


        # n = 42
        #
        # for i in range(3,0,-1):
        #
        #     task = Task()
        #     task.set_start(self.goal_point[n])
        #     n-=2
        #     task.set_goal(self.goal_point[n])
        #     n-=2
        #     task.set_status(1)
        #     if len(self.tasks) == 0:
        #         task.set_id(0)
        #     else:
        #         id = self.tasks[len(self.tasks)-1].get_id()+1
        #         task.set_id(id)
        #     self.tasks.append(task)
        #
        #
        #
        #     task = Task()
        #     task.set_start(self.goal_point[n])
        #     n-=2
        #     task.set_goal(self.goal_point[n])
        #     n-=1
        #     task.set_status(1)
        #     if len(self.tasks) == 0:
        #         task.set_id(0)
        #     else:
        #         id = self.tasks[len(self.tasks)-1].get_id()+1
        #         task.set_id(id)
        #     self.tasks.append(task)


            self.LookTask()
            self.map_label.clear_light()






        #print("暂停任务")

    def Conti_Task(self):
        print("继续任务")

    def clear_Task(self):
        self.tasks.clear()
        #self.look_dialog.show_list(self.tasks)

    def cancel_Task(self):
        if self.thread1 is not None:

         self.thread1.set_AIenable(False)
         self.thread1.set_AIenable(False)
         self.thread1.set_enable(False)
         self.thread1.set_start_cbs(False)
        self.navigation.delete_all_agent()

        #self.tasks.clear()
        if self.task_dispatchThread is not None:

         self.task_dispatchThread.status = False
         self.task_dispatchThread = None
        time.sleep(0.1)
        if self.task_thread is not None:
         self.task_thread.enable = False
        time.sleep(0.1)

        self.task_thread = None

        time.sleep(0.1)
        #把所有小车设为原点



        self.statist()


        self.reset_car()

        #
        #
        self.actionExecuteTask.setEnabled(True)
        self.actionCancelTask.setEnabled(False)
        self.look_dialog.show_list(self.tasks)
        self.look_dialog.updateSignel.emit(2)

        self.isagain = 0

        print("cancle-task:",len(self.tasks_old))

        self.tasks.clear()
    def statist(self):
        totalTime = 0.0
        pl_totalTime = 0.0
        collision_all = 0
        astar_cbs=0
        self.navigation.get_all_collision_number()

        for i in self.agvs:
            num = i.get_collision_number()
            startTime = i.get_startTime()
            endTime = i.get_endTime()
            pl_time = i.get_planning_endTime()-i.get_planning_startTime()
            Ti = endTime-startTime
            #print("ID：", i, "time:", Ti)
            totalTime+=Ti
            pl_totalTime+=pl_time
            collision_all+=num

        if self.thread1 is not None:

          astar_cbs = self.thread1.get_cbs()

        time.sleep(0.1)
        self.thread1 = None


        if self.isagain == 1:
         if self.way ==0:

          self.total_data.append([1,len(self.tasks),self.way,totalTime,collision_all])
         else:
             self.total_data.append([1, len(self.tasks), self.way, totalTime, astar_cbs])
        else:
           if self.way ==0:
            self.total_data.append([0,len(self.tasks), self.way, totalTime,collision_all])
           else:
               self.total_data.append([0, len(self.tasks), self.way, totalTime, astar_cbs])

        #print("规划时长：",pl_totalTime)
        if self.thread1 is not None:
         self.thread1.set_cbs(0)
        self.navigation.end_collision()

        print("运行时长：",totalTime)

    def reset_car(self):
        for car in self.agvs:
            home_x = car.get_home().x()
            home_y = car.get_home().y()
            car.set_location_angle(
                QPoint(home_x, home_y), 0)
            car.set_goal(QPoint(home_x, home_y))
            car.set_is_package(0)
            car.set_isbind(0)
            car.set_isstart(0)
            car.set_status(0)
            car.set_startTime(0)
            car.set_endTime(0)
            car.set_collision_number(0)


        self.map_label.get_car_list(self.agvs)


from virtual.map.qthread import Thread1


class Task_dispatchThread(QThread):
    def __init__(self,tasks,agvs):
        super().__init__()
        self.tasks = tasks
        self.agvs = agvs
        self.status = True


    def run(self):
        idle_agvs = dict()
        index = 0

        while(self.status):
            idle_agvs.clear()

            for t in self.tasks:

                if t.get_excute_status() == 0:

                    for m in self.agvs:

                        if m.get_isbind() == 0:
                            idle_agvs[m.get_id()] = m
                            #print("执行了一次")

                    if len(idle_agvs) == 0:
                        time.sleep(0.1)
                        continue

                    min_id = -1
                    start_x = t.get_start().get_point().x()
                    start_y = t.get_start().get_point().y()
                    min = 10000
                    for k, m in idle_agvs.items():
                        x = m.get_location().x() - start_x
                        y = m.get_location().y() - start_y
                        distance = math.sqrt(x ** 2 + y ** 2)

                        if min > distance and m.get_isbind() == 0:
                            min = distance
                            min_id = m.get_id()
                    if min_id>-1:
                        idle_agvs[min_id].set_status(-1)
                        idle_agvs[min_id].set_goal(QPoint(start_x, start_y))
                        t.set_bindAGV(min_id)
                        print("我绑定了一个小车，ID:",min_id,"任务ID：",t.get_id())
                        idle_agvs[min_id].set_isbind(1)
                        idle_agvs[min_id].set_isstart(0)
                        t.set_excute_status(1)



                    # idle_agvs[min_id].set_status(-1)
                    # idle_agvs[min_id].set_goal(QPoint(start_x, start_y))
                    # t.set_bindAGV(min_id)
                    # print("我绑定了一个小车，ID:",min_id,"任务ID：",t.get_id())
                    # idle_agvs[min_id].set_isbind(1)
                    # idle_agvs[min_id].set_isstart(0)
                    # t.set_excute_status(1)



            time.sleep(0.5)



    def enable_run(self,enable):

        self.status = enable






class TaskThread(QThread):
    def __init__(self,tasks,agvs,map_label,thread,navigation,navi_astar,way):
        super().__init__()
        self.enable = True
        self.tasks = tasks
        self.agvs = agvs
        self.map_label = map_label
        self.thread1 = thread
        self.navigation = navigation
        self.navi_astar = navi_astar
        self.way = way



    def run(self):

        while (self.enable):

            time.sleep(0.2)




            for i,task in enumerate(self.tasks):
              #time.sleep(0.3)

              if task.get_bindAGV() is not None:
                #print("---------------------------ID为", task.get_id(), "的任务,绑定的小车为：", self.agvs[task.get_bindAGV()].get_id(),"任务状态为：",task.get_excute_status())





                agv_status = self.agvs[task.get_bindAGV()].get_status()

                #print("ID为", task.get_id(), "的任务,绑定的小车为：", self.agvs[task.get_bindAGV()].get_id(),"agv_status:",agv_status)

                if self.agvs[task.get_bindAGV()].get_isbind() ==1 and self.agvs[task.get_bindAGV()].get_isstart() == 0:
                    print("----------------开始执行第一次规划,",len(self.tasks))
                    ##print("ID为", task.get_id(), "的任务开始执行,绑定的小车为：", self.agvs[task.get_bindAGV()].get_id())
                    ##print("----------------起始点：", task.get_start().get_point().x(), task.get_start().get_point().y())
                    #print("----------------目标点：", task.get_goal().get_point().x(), task.get_goal().get_point().y())
                    if self.way==0:

                     self.navigation.set_goal(self.agvs[task.get_bindAGV()])
                    else:
                        print("start astar ")
                        solution = self.navi_astar.navigation_simulation(self.agvs[task.get_bindAGV()],self.way)
                        self.thread1.set_solution([solution])

                    #self.thread1.enable_run = True

                    #self.thread1.set_solution([solution])

                    pl_startTime = time.time()
                    self.agvs[task.get_bindAGV()].set_planning_startTime(pl_startTime)

                    self.agvs[task.get_bindAGV()].set_isstart(1)
                    task.set_excute_status(1)
                    task.set_status(1)

                if agv_status == 1 and task.get_status() == 1:
                     pl_endTime = time.time()
                     self.agvs[task.get_bindAGV()].set_planning_endTime(pl_endTime)

                     goal_x = task.get_goal().get_point().x()
                     goal_y = task.get_goal().get_point().y()
                     print("第二次规划")
                     #start collision
                     self.agvs[task.get_bindAGV()].set_goal(QPoint(goal_x, goal_y))

                     #print("----------------起始点：", task.get_start().get_point().x(), task.get_start().get_point().y())
                     #print("----------------目标点：", task.get_goal().get_point().x(), task.get_goal().get_point().y())
                     if self.way == 0:
                         self.navigation.set_second()
                         self.navigation.set_goal(self.agvs[task.get_bindAGV()])

                     else:
                         self.thread1.set_start_cbs(True)
                         print("start astar ")
                         solution = self.navi_astar.navigation_simulation(self.agvs[task.get_bindAGV()],self.way)
                         self.thread1.set_solution([solution])
                     self.agvs[task.get_bindAGV()].set_status(-1)
                     task.set_status(2)

                     #self.thread1.enable_run = False
                     startTime = time.time()
                     self.agvs[task.get_bindAGV()].set_startTime(startTime)
                     #solution = self.navigation.set_goal(self.agvs[task.get_bindAGV()])

                     #self.thread1.enable_run = True
                     # print("----------------开始执行第二次规划,规划用时：　", end_time-start_time)
                     #self.thread1.set_solution([solution])
                     self.agvs[task.get_bindAGV()].set_is_package(1)
                     # print("ID为",str(task.get_bindAGV()),"的AGV到达起始点，开始执行任务：",task.get_id())
                     # print("起始点为：",self.agvs[task.get_bindAGV()].get_location().x(),self.agvs[task.get_bindAGV()].get_location().y())

                elif agv_status == 1 and task.get_status() == 2:
                    print("第三次规划")
                    goal_x = self.agvs[task.get_bindAGV()].get_home().x()
                    goal_y = self.agvs[task.get_bindAGV()].get_home().y()
                    self.agvs[task.get_bindAGV()].set_goal(QPoint(goal_x, goal_y))
                    self.agvs[task.get_bindAGV()].set_status(-1)
                    task.set_status(3)

                    #self.thread1.enable_run = False
                    start_time = time.time()
                    if self.way == 0:

                        self.navigation.set_goal(self.agvs[task.get_bindAGV()])
                    else:
                        print("start astar ")
                        solution = self.navi_astar.navigation_simulation(self.agvs[task.get_bindAGV()],self.way)
                        self.thread1.set_solution([solution])
                    #solution = self.navigation.set_goal(self.agvs[task.get_bindAGV()])
                    end_time = time.time()
                    #self.thread1.enable_run = True
                    # print("----------------开始执行第三次规划,规划用时：　", end_time - start_time)
                    #self.thread1.set_solution([solution])

                    # print("ID为", str(task.get_bindAGV()), "的AGV到达目标点，开始回停车点：", task.get_id())
                    # print("起始点为：", self.agvs[task.get_bindAGV()].get_location().x(),self.agvs[task.get_bindAGV()].get_location().y())
                    self.agvs[task.get_bindAGV()].set_is_package(0)
                # AGV完成任务
                elif agv_status == 1 and task.get_status() == 3:
                    print("完成任务")
                    endTime = time.time()
                    self.agvs[task.get_bindAGV()].set_endTime(endTime)
                    # print("ID为", str(task.get_bindAGV()), "的AGV完成任务")
                    self.agvs[task.get_bindAGV()].set_status(-1)
                    self.agvs[task.get_bindAGV()].set_isbind(0)
                    task.set_bindAGV(None)
                    task.set_status(4)
                    task.set_excute_status(2)






class  udpate_task_thread(QThread):


    def __init__(self,dialog,agvs):
        super().__init__()
        self.dialog = dialog
        self.agvs = agvs


    def run(self):

      while(1):
          self.dialog.updateSignel.emit(2)

          time.sleep(1)


class Dialog(QDialog):
    dialogSignel = pyqtSignal(int, int,int)
    def __init__(self,parent=None):
        super(Dialog, self).__init__(parent)

        self.status = 0  # 1为预览

        self.setWindowTitle("生成任务")
        self.layout=QGridLayout(self)
        #agv模拟数量控件
        self.agv_number_label=QLabel()
        self.agv_number_label.setText("起始点:")
        self.start_box = QComboBox(self, minimumWidth=5)
        self.goal_box = QComboBox(self, minimumWidth=5)
        #self.agv_number_edit = QLineEdit()
        #self.agv_max_label = QLabel()
        #self.agv_max_label.setStyleSheet("color:red")
        #站点数量

        self.goal_number_label = QLabel()
        #self.goal_number_label.setText("请输入站点数量:")
        self.goal_number_label.setText("目标点:")
        #self.goal_number_edit = QLineEdit()
        #self.goal_max_label = QLabel()
        #self.goal_max_label.setStyleSheet("color:red")

        #功能按钮
        self.preview_label = QPushButton()
        self.preview_label.setText("预览")
        self.cancel_label = QPushButton()
        self.cancel_label.setText("清空")
        self.accept_label = QPushButton()
        self.accept_label.setText("确定")

        self.layout.addWidget(self.agv_number_label,0,0)
        self.layout.addWidget(self.start_box,0,1)
        #self.layout.addWidget(self.goal_box,0,2)
        self.layout.addWidget(self.goal_number_label,1,0)
        self.layout.addWidget(self.goal_box,1,1)
        # self.layout.addWidget(self.goal_max_label,1,2)
        test_label = QLabel()
        test_label.setText("")
        #self.layout.addWidget(test_label, 2, 0)
        self.layout.addWidget(self.preview_label, 3, 0)
        self.layout.addWidget(self.preview_label,3,0)
        self.layout.addWidget(self.cancel_label,3,1)
        self.layout.addWidget(self.accept_label,3,2)
        self.preview_label.clicked.connect(self.preview)
        self.cancel_label.clicked.connect(self.cancel)
        self.accept_label.clicked.connect(self.accept)




    #该方法在父类方法中调用，直接打开了子窗体，返回值则用于向父窗体数据的传递

    def preview(self):

        print("预览")
        self.status = 1
        self.start = self.start_box.currentIndex()
        self.goal = self.goal_box.currentIndex()

        # self.agv_number = self.agv_number_edit.text()
        # self.goal_number = self.goal_number_edit.text()
        # #print(self.agv_number)
        # if self.agv_number is not "" and self.goal_number is not "":
        #
        self.dialogSignel.emit(0,int(self.start),int(self.goal))

    def cancel(self):
        self.status = 0
        # self.agv_number_edit.setText("")
        # self.goal_number_edit.setText("")
        self.dialogSignel.emit(1,0,0)
        print("清空")

    def accept(self):
        self.start = self.start_box.currentIndex()
        self.goal = self.goal_box.currentIndex()

        self.dialogSignel.emit(2,int(self.start),int(self.goal))
        self.close()
        print("生成")


class LookDialog(QDialog):
    dialogSignel = pyqtSignal(int, int, int)

    updateSignel = pyqtSignal(int)

    def __init__(self, parent=None):
            super(LookDialog, self).__init__(parent)
            self.setWindowTitle("任务列表")

            self.updateSignel.connect(self.update)

            self.row = 0
            self.task_list = []
            self.resize(525, 300)
            self.layout = QHBoxLayout()
            self.TableWidget = QTableWidget()
            QTableWidget.resizeColumnsToContents(self.TableWidget)
            QTableWidget.resizeRowsToContents(self.TableWidget)
            self.TableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
            self.TableWidget.setColumnCount(5)
            self.TableWidget.setContextMenuPolicy(Qt.CustomContextMenu)
            self.TableWidget.customContextMenuRequested.connect(self.generateMenu)
            self.TableWidget.setHorizontalHeaderLabels(['任务ID', '起始点', '目标点','状态',"绑定AGV"])
            self.TableWidget.verticalHeader().setVisible(False)
            self.TableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.TableWidget.clicked.connect(self.click_item)
            #删除某一行
            #self.TableWidget.removeRow()
    def show_list(self, task_list):
            self.task_list = task_list
            self.TableWidget.setRowCount(len(task_list))

            for i, m in enumerate(self.task_list):

                agv_label = QTableWidgetItem(str(i))
                self.TableWidget.setItem(i, 0, agv_label)
                #self.task_list[i].set_id(i)
                item1 = m.get_start().get_id()
                if item1 ==78:
                    start_str = "入库点"
                elif item1 == 79:
                    start_str = "出库点"
                else:
                    start_str = "站" + str(m.get_start().get_id())
                start_label = QTableWidgetItem(start_str)
                self.TableWidget.setItem(i, 1, start_label)
                item2 = m.get_goal().get_id()
                if item2 == 78:
                    start_str2 = "入库点"
                elif item2 == 79:
                    start_str2 = "出库点"
                else:
                    start_str2 = "站" + str(m.get_goal().get_id())
                #goal = "站" + str(m.get_goal().get_id())
                goal_label =  QTableWidgetItem(start_str2)
                self.TableWidget.setItem(i, 2, goal_label)
                status = task_list[i].get_excute_status()
                if status == 0 :
                    status_str = "未执行"
                    status_label = QTableWidgetItem(status_str)
                elif status == 1:
                    status_str = "执行中"
                    status_label = QTableWidgetItem(status_str)
                    status_label.setForeground(QBrush(QColor(255,0,0)))
                elif status ==2:
                    status_str = "已完成"
                    status_label = QTableWidgetItem(status_str)
                    status_label.setForeground(QBrush(QColor(44,18,239)))
                self.TableWidget.setItem(i, 3, status_label)
                bind_agv = task_list[i].get_bindAGV()
                if bind_agv == None:
                    str1 = "无"
                else:
                    str1 = str(bind_agv)
                bind_agv_label =  QTableWidgetItem(str1)
                self.TableWidget.setItem(i, 4, bind_agv_label)
            self.layout.addWidget(self.TableWidget)
            self.setLayout(self.layout)


    def update(self,item):
        if item == 2 and len(self.task_list)>0:

            for i, m in enumerate(self.task_list):

                agv_label = QTableWidgetItem(str(i))
                self.TableWidget.setItem(i, 0, agv_label)
                item1 = m.get_start().get_id()
                if item1 == 78:
                    start_str = "入库点"
                elif item1 == 79:
                    start_str = "出库点"
                else:
                    start_str = "站" + str(m.get_start().get_id())
                start_label = QTableWidgetItem(start_str)
                self.TableWidget.setItem(i, 1, start_label)
                item2 = m.get_goal().get_id()
                if item2 == 78:
                    start_str2 = "入库点"
                elif item2 == 79:
                    start_str2 = "出库点"
                else:
                    start_str2 = "站" + str(m.get_goal().get_id())
                #goal = "站" + str(m.get_goal().get_id())
                goal_label =  QTableWidgetItem(start_str2)
                self.TableWidget.setItem(i, 2, goal_label)
                status = self.task_list[i].get_excute_status()

                if status == 0 :
                    status_str = "未执行"
                    status_label = QTableWidgetItem(status_str)
                elif status == 1:
                    status_str = "执行中"
                    status_label = QTableWidgetItem(status_str)
                    status_label.setForeground(QBrush(QColor(255,0,0)))
                elif status ==2:
                    status_str = "已完成"

                    status_label = QTableWidgetItem(status_str)
                    status_label.setForeground(QBrush(QColor(44,18,239)))
                self.TableWidget.setItem(i, 3, status_label)
                bind_agv = self.task_list[i].get_bindAGV()
                if bind_agv == None:
                    str1 = "无"
                else:
                    str1 = str(bind_agv)
                bind_agv_label =  QTableWidgetItem(str1)
                self.TableWidget.setItem(i, 4, bind_agv_label)


    def click_item(self):
        for i in self.TableWidget.selectionModel().selection().indexes():
            self.row = i.row()
        self.dialogSignel.emit(1, self.row, 0)

    def delete_item(self,row):
        self.TableWidget.removeRow(row)


    def generateMenu(self, pos):
            # 计算有多少条数据，默认-1,
            row_num = -1
            for i in self.TableWidget.selectionModel().selection().indexes():
                row_num = i.row()


                menu = QMenu()
                item1 = menu.addAction(u'删除')

                action = menu.exec_(self.TableWidget.mapToGlobal(pos))

                # 显示选中行的数据文本
                if action == item1:
                    self.delete_item(row_num)
                    self.dialogSignel.emit(0, row_num, 0)
                    menu.close()
                    break

    def closeEvent(self, QCloseEvent):
        self.dialogSignel.emit(2, 0, 0)



class TotalDialog(QDialog):
    dialogSignel = pyqtSignal(int)

    def __init__(self, parent=None):
        super(TotalDialog, self).__init__(parent)
        self.setWindowTitle("数据统计")

        self.row = -1

        self.resize(430, 300)
        self.layout = QHBoxLayout()
        # self.task_list = task_list
        self.TableWidget = QTableWidget()
        QTableWidget.resizeColumnsToContents(self.TableWidget)
        QTableWidget.resizeRowsToContents(self.TableWidget)
        self.TableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.TableWidget.setColumnCount(4)
        self.TableWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.TableWidget.setHorizontalHeaderLabels(['测试ID','任务数量', '算法类型', '运行时长'])
        self.TableWidget.verticalHeader().setVisible(False)
        self.TableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)
        # 删除某一行
        # self.TableWidget.removeRow()
        #self.dialogSignel.connect(self.update_car)

        self.TableWidget.setRowCount(1)
        agv_label = QTableWidgetItem(0)
        self.TableWidget.setItem(1, 0, agv_label)
        self.layout.addWidget(self.TableWidget)
        self.setLayout(self.layout)

    def show_data(self,data):
        index = 0
        self.TableWidget.setRowCount(len(data))

        for i, m in enumerate(data):
            if m[0] == 1:

              agv_label = QTableWidgetItem(str(index))


            else:
                index+=1
                agv_label = QTableWidgetItem(str(index))
            self.TableWidget.setItem(i, 0, agv_label)

            #task number
            task_number = m[1]
            taskNumber_lable = QTableWidgetItem(str(task_number))
            self.TableWidget.setItem(i, 1, taskNumber_lable)

            if m[2] == 0:
             type_lable = QTableWidgetItem("DRL")
            elif m[2] ==1:
             type_lable = QTableWidgetItem("A*")
            elif m[2] ==2:
             type_lable = QTableWidgetItem("Dijkstra")
            elif m[2] ==3:
             type_lable = QTableWidgetItem("BellmanFord")
            self.TableWidget.setItem(i, 2, type_lable)
            #set run time
            times = int(m[3])
            times_lable = QTableWidgetItem(str(times))
            self.TableWidget.setItem(i, 3, times_lable)

            #set collision number
            collosion_number = m[4]
            collision_lable = QTableWidgetItem(str(collosion_number))
            self.TableWidget.setItem(i, 4, collision_lable)




        self.layout.addWidget(self.TableWidget)
        self.setLayout(self.layout)


class Task():
    def __init__(self):
        super(Task, self).__init__()

        self.__id = 0
        self.__start = None
        self.__goal = None
        self.__status = -1
        self.__bindAGV = None
        self.__excute_status = 0
        self.__home = None
    def set_id(self,id):
        self.__id = id

    def set_start(self,start):
        self.__start = start

    def set_goal(self,goal):
        self.__goal = goal

    def set_status(self,status):
        self.__status = status

    def get_id(self):
        return  self.__id

    def get_start(self):
        return self.__start

    def get_goal(self):
        return self.__goal

    def get_status(self):
        return self.__status

    def set_bindAGV(self,agv_id):
        self.__bindAGV = agv_id

    def get_bindAGV(self):
        return self.__bindAGV


    def set_excute_status(self,status):
        self.__excute_status = status

    def get_excute_status(self):
        return self.__excute_status

    def set_home(self,home):
        self.__home = home

    def get_home(self):
        return  self.__home





class Navi_thread(QThread):
    def __init__(self, navi, car_list, map_name):
        super().__init__()
        self.navigation = navi
        self.car_list = car_list
        self.map_name = map_name

    def run(self):
        print("navi init----------////////")
        # self.navigation = Navigator1()  # 导航模块
        self.navigation.init(self.map_name)
        self.navigation.init_map_m(self.map_name)
        print("len(car_list): ", len(self.car_list))
        self.navigation.init_data(self.map_name, self.car_list)
