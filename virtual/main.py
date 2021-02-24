import json
import sys
import time

from PyQt5 import QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import *

from virtual.map.mapManager import Map_Layer
from virtual.monitor.CoordinateTransition import Transition
from virtual.monitor.monitoring_unit import Agv_Car
from virtual.views import mainUI
from virtual.work.TaskManager import TaskManager
from PyQt5 import QtCore, QtGui

class MainCode(QMainWindow, mainUI.Ui_MainWindow):
    """
            editroadnet_status
             0 : 无效
             1 ：有效
    """
    def __init__(self):
        QMainWindow.__init__(self)
        self.__mapStatus = 0   # 地图加载标识
        self.__roadStatus = 0   # 路网加载标识
        self.road_path = None
        self.monitoring_status = 0
        # 用于像素转换的类
        self.pixel_translation = Transition()
        self.road_label = None
        self.monitoring_label = None
        self.img = None
        self.__map_name = ""
        self.map_name = ""
        self.__img__ = None
        self.gray_map = None
        self.buffer_road_status = None
        self.map_label = None
        self.agvs=[]
        self.select_map = 0 #是否是第一次打开地图


        mainUI.Ui_MainWindow.__init__(self)
        self.setupUi(self)                         #设置控件
        self.setWindowTitle("AGV调度及监控系统")
        self.hbox = QtWidgets.QHBoxLayout()        #创建布局容器
        self.setStyleSheet("#MainWindow{background-color: rgb(128,128,128)}")   #设置背景颜色
        self.showMaximized()               #最大化窗口

        #地图图层
        self.map_label = Map_Layer(self)
        self.map_label.setMouseTracking(True)
        self.map_label.init_data(self.pixel_translation)
        self.hbox.addWidget(self.map_label)
        self.map_label.hide()

        # 导入地图监听
        self.loadmap.triggered.connect(self.loadMap)
        # 调度监听
        self.actionNewTask.triggered.connect(self.new_task)
        self.actionLookTask.triggered.connect(self.look_task)
        self.actionExecuteTask.triggered.connect(self.execute_task)
        self.actionAgainTask.triggered.connect(self.again_task)
        self.actionPauseTask.triggered.connect(self.pause_task)
        self.actionCancelTask.triggered.connect(self.cancel_task)
        #self.actionClearTask.triggered.connect(self.clear_task)
        #self.actionExecuteTask.setEnabled(False)
        self.actionCancelTask.setEnabled(False)
        #self.actionTime.triggered.connect(self.total_time)
        self.actionAI.triggered.connect(self.AI)
        self.actionAstar.triggered.connect(self.astar)
        self.actionDijkstra.triggered.connect(self.dijkstra)
        self.actionBellman.triggered.connect(self.bellman_ford)
        self.actionData.triggered.connect(self.Total)

        #self.test_thread = Thread1(self.map_label,self.agvs)
        self.icon_red = QtGui.QIcon()
        self.icon_red.addPixmap(QtGui.QPixmap("res/image/state_red.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.icon_gray = QtGui.QIcon()
        self.icon_gray.addPixmap(QtGui.QPixmap("res/image/state_gray.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        #self.TaskThread = TaskThread()

    # def clear_task(self):
    #     self.task_manager.clear_Task()
    #     print("clear")

    def again_task(self):
        self.task_manager.Again()

    def Total(self):
        self.task_manager.Total()
        print("total")

    def AI(self):
        self.task_manager.selectMode(0)

        self.actionAI.setIcon(self.icon_red)
        self.actionAstar.setIcon(self.icon_gray)
        self.actionDijkstra.setIcon(self.icon_gray)
        self.actionBellman.setIcon(self.icon_gray)


        #self.actionAI.setIcon()
        print("AI")
    def astar(self):
        self.task_manager.selectMode(1)
        self.actionAI.setIcon(self.icon_gray)
        self.actionAstar.setIcon(self.icon_red)
        self.actionDijkstra.setIcon(self.icon_gray)
        self.actionBellman.setIcon(self.icon_gray)
        print("astar")
    def dijkstra(self):
        self.task_manager.selectMode(2)
        self.actionAI.setIcon(self.icon_gray)
        self.actionAstar.setIcon(self.icon_gray)
        self.actionDijkstra.setIcon(self.icon_red)
        self.actionBellman.setIcon(self.icon_gray)
        print("dijkstra")
    def bellman_ford(self):
        self.task_manager.selectMode(3)
        self.actionAI.setIcon(self.icon_gray)
        self.actionAstar.setIcon(self.icon_gray)
        self.actionDijkstra.setIcon(self.icon_gray)
        self.actionBellman.setIcon(self.icon_red)
        print("bellman_ford")




    #计算时长
    def total_time(self):
        self.task_manager.get_collision()
        totalTime = 0.0
        pl_totalTime = 0.0
        collision_num = 0

        for i in self.agvs:
            num = i.get_collision_number()
            startTime = i.get_startTime()
            endTime = i.get_endTime()
            pl_time = i.get_planning_endTime()-i.get_planning_startTime()
            Ti = endTime-startTime
            #print("ID：", i, "time:", Ti)
            totalTime+=Ti
            pl_totalTime+=pl_time
            collision_num+=num

        #print("规划时长：",pl_totalTime)
        print("运行时长：",totalTime)
        print("collistion----",collision_num)


    #导入地图
    def loadMap(self):
        self.map_label.clear_all()
        self.__map_name,self.__img__ = self.map_label.LoadMap()
        if self.__map_name is not None:

            self.__mapStatus =1
            # 节点和站点信息
            json_name = self.__map_name[0:-4] + ".json"

            if self.select_map ==0: #第一次打开地图
              self.task_manager = TaskManager(self.map_label,self.__map_name)
              self.init_data(json_name)
              self.select_map =1
            else:
              self.clear_all()
              self.task_manager = TaskManager(self.map_label,self.__map_name)
              self.init_data(json_name)


        else:
            self.__mapStatus = 0

    def clear_all(self):
        self.agvs = []
        self.task_manager.cancel_Task()
        self.actionAI.setIcon(self.icon_gray)
        self.actionAstar.setIcon(self.icon_gray)
        self.actionDijkstra.setIcon(self.icon_gray)
        self.actionBellman.setIcon(self.icon_gray)
    def init_agv(self,agvs):
        self.agvs = agvs
        self.agv_dilog = AGVDialog()
        self.agv_dilog.show_list(self.agvs)

        self.UpdateAGVthread = TaskThread(self.agv_dilog,self.agvs)
        self.UpdateAGVthread.start()
    def init_data(self,json_name):
        self.filename = json_name
        with open(self.filename, 'r+', encoding='utf-8') as f:
            load_dict = json.load(f)

            number_power = load_dict.get("number_power").get("number_power")
            print("========xiaoche======",number_power)
            for i in range(0, number_power):
            #for i in range(0, 20):
                self.agvs.append(Agv_Car())
                index = 'power' + str(i)

                self.agvs[i].set_location_angle(QPoint(load_dict.get(index).get("point.x"), load_dict.get(index).get("point.y")), 0)
                self.agvs[i].set_goal(QPoint(load_dict.get(index).get("point.x"), load_dict.get(index).get("point.y")))
                self.agvs[i].set_id(i)
                self.agvs[i].set_battery(100)
                self.agvs[i].set_status(-1)
                self.agvs[i].set_home(QPoint(load_dict.get(index).get("point.x"), load_dict.get(index).get("point.y")))

            self.task_manager.init(load_dict,self.agvs,self.actionExecuteTask,self.actionCancelTask)
            self.init_agv(self.agvs)

    #   生成任务
    def new_task(self):
        if self.__mapStatus != 1:
            QMessageBox.information(self, "提示", "请打开地图！", QMessageBox.Yes, QMessageBox.Yes)
            return
        print("开始生成任务")


        self.task_manager.NewTask()
        # print("agv_number:",self.agv_number)
        # print("goal_number:", self.goal_number)
    # 查看任务
    def look_task(self):
        print("查看任务")
        #self.monitoring_label.set_process(3)
        self.task_manager.LookTask()


    def execute_task(self):
        if self.__mapStatus != 1:
            QMessageBox.information(self, "提示", "请打开地图！", QMessageBox.Yes, QMessageBox.Yes)
            return
        print("执行任务")

        self.task_manager.Execute_Task()
        self.map_label.get_car_list(self.agvs)
        self.map_label.execute_task()
        self.actionExecuteTask.setEnabled(False)

        self.actionCancelTask.setEnabled(True)


        #self.test_thread.start()
        # zmq_recv_thread = threading.Thread(target=self.execute_task_thread)
        # zmq_recv_thread.start()

    def pause_task(self):

        self.task_manager.Pause_Task()
        print("加载配置")

    def cancel_task(self):

        #self.TaskThread.start()
        print("停止任务")
        #self.map_label.clear_carlist()
        self.task_manager.cancel_Task()

    def closeEvent(self, event):
        if self.buffer_road_status == 1 and self.road_label.get_movement_status():
            reply = QMessageBox.question(self, self.windowTitle(), "路网未保存，是否要退出程序？", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.road_label.clear_buffer_road()
                event.accept()
            else:
                self.road_label.clear_buffer_road()
                event.ignore()
        else:
            event.accept()



    #内部处理
    def sendMsgWindow(self, s):
        QMessageBox.information(self,"提示", s, QMessageBox.Yes, QMessageBox.Yes)

    def resizeEvent(self, event):            # 改变窗口和lable的大小
         super(MainCode, self).resizeEvent(event)
         self.pixel_translation.setMainWindowSize(self.geometry())
         if self.map_label is not None:
              self.map_label.setFixedSize(self.pixel_translation.getDisplayLabelSize().width(),self.pixel_translation.getDisplayLabelSize().height())
              self.map_label.move(self.pixel_translation.getDisplayLabelSize().x(),self.pixel_translation.getDisplayLabelSize().y())


class TaskThread(QThread):

    def __init__(self,dialog,agvs):
        super().__init__()
        self.dialog = dialog
        self.agvs = agvs


    def run(self):

      while(1):
          self.dialog.dialogSignel.emit(2)

          time.sleep(0.5)
          #print("============刷新小车==========")


class AGVDialog(QDialog):
        dialogSignel = pyqtSignal(int)

        def __init__(self, parent=None):
            super(AGVDialog, self).__init__(parent)
            self.setWindowTitle("AGV信息")


class AGVDialog(QDialog):
        dialogSignel = pyqtSignal(int)

        def __init__(self, parent=None):
            super(AGVDialog, self).__init__(parent)
            self.setWindowTitle("AGV信息")

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
            self.TableWidget.setHorizontalHeaderLabels(['车辆ID', '坐标', '电量', '状态'])
            self.TableWidget.verticalHeader().setVisible(False)
            self.TableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)
            # 删除某一行
            # self.TableWidget.removeRow()
            self.dialogSignel.connect(self.update_car)

        def show_list(self, agv_list):
            self.agv_list = agv_list
            self.TableWidget.setRowCount(len(agv_list))

            for i, m in enumerate(self.agv_list):
                agv_label = QTableWidgetItem(str(m.get_id()))
                self.TableWidget.setItem(i, 0, agv_label)
                start = "("+str(m.get_location().x())+","+str(m.get_location().y())+")"
                self.xy_label = QTableWidgetItem(start)
                self.TableWidget.setItem(i, 1, self.xy_label)
                power = str(m.get_battery())+"%"
                self.power_label = QTableWidgetItem(power)
                self.TableWidget.setItem(i, 2, self.power_label)
                #status = m.get_status()
                status = m.get_isbind()
                if status == 0 :
                    status_str = "空闲"
                    self.status_label = QTableWidgetItem(status_str)
                    self.status_label.setForeground(QBrush(QColor(0, 0, 0)))
                elif status == 1:
                    status_str = "繁忙"
                    self.status_label = QTableWidgetItem(status_str)
                    self.status_label.setForeground(QBrush(QColor(255, 0, 0)))

                self.TableWidget.setItem(i, 3, self.status_label)
            self.layout.addWidget(self.TableWidget)
            self.setLayout(self.layout)
            self.show()

        def update_car(self, item):
            if item == 2:
             for i,m in enumerate(self.agv_list):




                    start = "(" + str(m.get_location().x()) + "," + str(m.get_location().y()) + ")"
                    xy_label = QTableWidgetItem(start)
                    self.TableWidget.setItem(i, 1, xy_label)
                    power = str(m.get_battery())+"%"
                    self.power_label = QTableWidgetItem(power)
                    self.TableWidget.setItem(i, 2, self.power_label)
                    #status = m.get_status()
                    status = m.get_isbind()
                    if status == 0:
                        status_str = "空闲"
                        self.status_label = QTableWidgetItem(status_str)
                        self.status_label.setForeground(QBrush(QColor(0, 0, 0)))
                    elif status == 1:
                        status_str = "繁忙"
                        self.status_label = QTableWidgetItem(status_str)
                        self.status_label.setForeground(QBrush(QColor(255, 0, 0)))
                    self.TableWidget.setItem(i, 3, self.status_label)




             # self.update()
             # self.TableWidget.update()
             # self.layout.update()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    md = MainCode()
    md.show()
    sys.exit(app.exec_())
