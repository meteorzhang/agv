from PyQt5.Qt import *
from PyQt5.QtWidgets import *
import time
from queue import Queue
import math
import json


class Agv_Car(QPoint):
    def __init__(self):
        super(Agv_Car, self).__init__()
        self.__image = None
        self.__package = None
        self.__id = 0
        self.__location = None
        self.angle = 90
        self.__goal = None
        self.__path = None
        self.__show_status = None
        self.__select_status = None
        self.__line_speed = 0.0
        self.__angle_speed = 0.0
        self.__carry_cargo = None

        self.show_location_x = None
        self.show_location_y = None
        self.show_angle = None
        self.show_goal_x = None
        self.show_goal_y = None
        self.__home = None
        self.x = None
        self.y = None
        self.goal_x = None
        self.goal_y = None
        self.__battery = 0
        self.__status = None
        self.__isbind = 0
        self.__is_package = 0
        self.__is_start = 0
        self.__startTime = 0.0
        self.__endTime = 0.0
        self.__planning_startTime = 0.0
        self.__planning_endTime = 0.0
        self.__collision_number = 0

    def set_collision_number(self,number):
        self.__collision_number = number

    def get_collision_number(self):
        return self.__collision_number

    def set_planning_startTime(self,pl_startTime):
        self.__planning_startTime = pl_startTime
    def get_planning_startTime(self):
        return self.__planning_startTime
    def set_planning_endTime(self,pl_endTime):
        self.__planning_endTime = pl_endTime
    def get_planning_endTime(self):
        return self.__planning_endTime
    def set_startTime(self,startTime):
        self.__startTime = startTime
    def get_startTime(self):
        return self.__startTime
    def set_endTime(self,endTime):
        self.__endTime = endTime
    def get_endTime(self):
        return self.__endTime

    def set_home(self,home):
        self.__home = home
    def get_home(self):
        return self.__home
    def set_isstart(self,start):
        self.__is_start = start
    def get_isstart(self):
        return self.__is_start

    def get_package_image(self):
        return self.__package
    def get_is_package(self):
        return self.__is_package
    def set_is_package(self,a):
        self.__is_package = a


    def set_location_angle(self, location, angle):
        self.__location = location
        self.x = location.x()
        self.y = location.y()

        self.angle = angle
        self.__show_status = 1
        self._create_car_image()
        self._create_package_image()
        self.__draw_ID()

    def set_line_and_angle_speed(self, x, y):
        self.__angle_speed = x
        self.__line_speed = y

    def set_show_location_angle(self,x,y,angle):
        self.show_location_x = x
        self.show_location_y = y
        self.show_angle = angle

    def set_show_goal(self, x, y):
        self.show_goal_x = x
        self.show_goal_y = y

    def set_angle(self, angle):
        self.angle = angle
        self.__show_status = 1
        self._create_car_image()

    def set_select_status(self, s=None):        # 修改选择的状态，更新小车的图片
        self.__select_status = s
        self._create_car_image()

    def get_select_status(self):
        return self.__select_status

    def get_show(self):
        return self.__show_status

    def get_location(self):
        return self.__location

    def set_goal(self, goal):
        self.__goal = goal
        self.goal_x = goal.x()
        self.goal_y = goal.y()

    def get_goal(self):
        return self.__goal

    def pop_task(self):
        self.__goal.get()

    def set_path(self, path):
        self.__path = path

    def get_path(self):
        return self.__path

    def set_id(self, id):
        self.__id = id
        self.__draw_ID()

    def get_id(self):
        return self.__id

    def get_image(self):
        return self.__image

    def get_id_image(self):
        return self.__ID_image

    def get_image_size(self):
        return self.__image.rect().width(),self.__image.rect().height()

    def set_battery(self,battery):
        self.__battery = battery

    def get_battery(self):
        return self.__battery

    def set_status(self,status):
        self.__status = status

    def get_status(self):
        return self.__status

    def set_isbind(self,isbind):
        self.__isbind = isbind

    def get_isbind(self):
        return self.__isbind


    def _create_car_image(self):
        # 绘制一个小车
        self.__image = QImage(30, 30, QImage.Format_ARGB32)
        if self.__select_status == 1:  # 被选中的状态
            self.__image.fill(Qt.darkRed)
        else :
            self.__image.fill(Qt.transparent)
        painter = QPainter()
        painter.begin(self.__image)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.setPen(QPen(Qt.black, 2))
        painter.setBrush(Qt.black)
        painter.resetTransform()
        painter.translate(self.__image.width() / 2, self.__image.height() / 2)
        painter.rotate(self.angle)
        painter.drawEllipse(-5, -5, 8, 8)
        #painter.drawRect(-8, -14, 16, 28)
        # painter.drawRect(-12, -10, 4, 8)
        # painter.drawRect(8, -10, 4, 8)
        # painter.drawRect(-12, 2, 4, 8)
        # painter.drawRect(8, 2, 4, 8)
        painter.setPen(QPen(Qt.white, 2))
        painter.setBrush(Qt.white)
        painter.drawEllipse(-2, -4, 2, 2)
        painter.end()
        self.__image = self.__image.scaled(30, 30)
    def _create_package_image(self):
        # 绘制一个小车
        self.__package = QImage(30, 30, QImage.Format_ARGB32)
        if self.__select_status == 1:  # 被选中的状态
            self.__package.fill(Qt.darkRed)
        else :
            self.__package.fill(Qt.transparent)
        painter = QPainter()
        painter.begin(self.__package)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.setPen(QPen(Qt.black, 2))
        painter.setBrush(Qt.black)
        painter.resetTransform()
        painter.translate(self.__image.width() / 2, self.__image.height() / 2)
        painter.rotate(self.angle)
        rect = QRect(-5, -5, 8, 8)
        painter.drawRect(rect)

        # painter.drawEllipse(-5, -5, 10, 10)
        painter.setPen(QPen(Qt.white, 2))
        painter.setBrush(Qt.white)
        painter.drawEllipse(-2, -4, 2, 2)
        painter.end()
        self.__package = self.__package.scaled(30, 30)

    def __draw_ID(self):
        s = "ID:" + str(self.get_id())
        self.__ID_image = QImage(40, 16, QImage.Format_ARGB32)
        self.__ID_image.fill(Qt.transparent)

        painter = QPainter()
        painter.begin(self.__ID_image)
        painter.setPen(QPen(Qt.black, 5))
        F = QFont('SimSun', 8)
        F.setBold(True)
        painter.setFont(F)  # SimSun 宋体 SimHei 黑体
        painter.drawText(QRect(0, 0, 40, 20), Qt.AlignCenter, s)
        painter.end()


class Show_Car_UI(QThread):
    sinOut = pyqtSignal(str)  # 自定义信号，执行run()函数时，从相关线程发射此信号
    sinOut_demo = pyqtSignal(str)

    def __init__(self, parent=None):
        super(Show_Car_UI, self).__init__(parent)
        self.__working = None

    def enable_work(self):    # 关闭
        self.__working = True
        self.start()

    def disable_work(self):  # 打开
        self.__working = False
        self.wait()

    def run(self):
        while self.__working is True:
            self.sinOut.emit("refresh")
            self.sinOut_demo.emit("hello")
            time.sleep(0.2)


class Agv_Information_Window(QDockWidget):
    def __init__(self, parent=None):
        super(Agv_Information_Window, self).__init__(parent)
        self.__init_ui()

    def __init_ui(self):
        self.__enable = False
        self.setWindowTitle("在线AGV信息列表")
        self.layout = QHBoxLayout()

        self.tableWidget = QTableWidget()
        self.setFont(QFont('SimSun', 10))   # 宋体，10号字
        #self.tableWidget.horizontalHeader().setClickable(False)             # 设置表头不可点击
        font = QFont('SimSun', 10)
        font.setBold(True)
        self.tableWidget.horizontalHeader().setFont(font)

        #self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  #

        #self.tableWidget.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)  #
        self.tableWidget.setSelectionMode(QAbstractItemView.SingleSelection)  # 设置只能选中一行
        self.tableWidget.setEditTriggers(QTableView.NoEditTriggers)  # 不可编辑
        self.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows);  # 设置

        self.tableWidget.verticalHeader().setVisible(False)
        self.setFloating(False)
        self.setWidget(self.tableWidget)
        # 实例化列表窗口，添加几个条目
        # 设置dock窗口是否可以浮动，True，运行浮动在外面，自动与主界面脱离，False，默认浮动主窗口内，可以手动脱离
        self.setFloating(False)
        self.hide()

    def set_enable(self, s):
        self.__enable = s
        if s == False:
            self.close()

    def get_enable(self):
        return self.__enable

    def update_ui(self, car_list):
        self.tableWidget.setRowCount(len(car_list))
        self.tableWidget.setColumnCount(5)
        self.tableWidget.setHorizontalHeaderLabels(['ID', '位置(x,y)', '角度', '任务', '电量'])
        self.setFixedSize(220,(len(car_list)+1)*35)   # 长，宽
        #self.tableWidget.setFont(QFont("SimSun", 12))  # 字体
        # 在窗口区域设置QWidget，添加列表控件
        for i, x in enumerate(car_list):
            information = x.get_information_json()
            self.tableWidget.setItem(i, 0, QTableWidgetItem(str(information.get("id"))))
            location = "("+information.get("show_location_x")+","+information.get("show_location_y")+")"
            self.tableWidget.setItem(i, 1, QTableWidgetItem(location))
            self.tableWidget.setItem(i, 2, QTableWidgetItem(str(information.get("show_angle"))))
            if information.get("show_goal_x") is not None:
                self.tableWidget.setItem(i, 3, QTableWidgetItem("True"))
            else:
                self.tableWidget.setItem(i, 3, QTableWidgetItem("False"))
            self.tableWidget.setItem(i, 4, QTableWidgetItem(str(information.get("power"))))
        #self.tableWidget.horizontalHeader().resizeColumnsToContents()
        self.tableWidget.resizeColumnsToContents()
        # self.tableWidget.resizeRowsToContents()
        #self.setFixedSize(self.tableWidget.rect())
        self.show()

    def mousePressEvent(self, event):
        super(Agv_Information_Window, self).mouseMoveEvent(event)
        item_list = self.tableWidget.selectedItems();
        if item_list is None:
           return

    def closeEvent(self, event):
        self.__enable = False
        event.accept()

class single_Information_Window(QDockWidget):
    def __init__(self, parent=None):
        super(single_Information_Window, self).__init__(parent)
        self.__init_ui()

    def __init_ui(self):

        self.__enable = False

        self.__id = None

        self.setWindowTitle("AGV详细信息")
        self.layout = QHBoxLayout()

        self.tableWidget = QTableWidget()
        self.setFont(QFont('SimSun', 10))  # 宋体，10号字
        font = QFont('SimSun', 10)
        font.setBold(True)
        self.tableWidget.horizontalHeader().setFont(font)
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  #
        self.tableWidget.setSelectionMode(QAbstractItemView.SingleSelection)  # 设置只能选中一行
        self.tableWidget.setEditTriggers(QTableView.NoEditTriggers)  # 不可编辑
        self.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows);  # 设置
        self.tableWidget.verticalHeader().setVisible(False)
        self.setFloating(False)
        self.setWidget(self.tableWidget)
        # 实例化列表窗口，添加几个条目
        # 设置dock窗口是否可以浮动，True，运行浮动在外面，自动与主界面脱离，False，默认浮动主窗口内，可以手动脱离
        self.setFloating(False)
        self.hide()

    def set_enable(self, s, id = None):
        if s == False:
            self.close()
        else:
            self.__id = id
        self.__enable = s

    def get_enable(self):
        return self.__enable

    def get_id(self):
        return self.__id

    # ＩＤ，位置，角度，线速度，角速度，任务中，载货情况
    def update_ui(self, car):
        self.tableWidget.setRowCount(7)
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setHorizontalHeaderLabels(['属性', '值'])
        self.setFixedSize(220,8*35)   # 长，宽
        #self.tableWidget.setFont(QFont("SimSun", 12))  # 字体
        # 在窗口区域设置QWidget，添加列表控件
        information = car.get_information_json()
        self.tableWidget.setItem(0, 0, QTableWidgetItem("ID"))
        self.tableWidget.setItem(0, 1, QTableWidgetItem(str(information.get("id"))))
        self.tableWidget.setItem(1, 0, QTableWidgetItem("位置"))
        location = "(" + information.get("show_location_x") + "," + information.get("show_location_y") + ")"
        self.tableWidget.setItem(1, 1, QTableWidgetItem(location))
        self.tableWidget.setItem(2, 0, QTableWidgetItem("朝向"))
        self.tableWidget.setItem(2, 1, QTableWidgetItem(str(information.get("show_angle"))))
        self.tableWidget.setItem(3, 0, QTableWidgetItem("线速度"))
        self.tableWidget.setItem(3, 1, QTableWidgetItem(str(information.get("line_speed"))))
        self.tableWidget.setItem(4, 0, QTableWidgetItem("角速度"))
        self.tableWidget.setItem(4, 1, QTableWidgetItem(str(information.get("angle_speed"))))
        self.tableWidget.setItem(5, 0, QTableWidgetItem("目标点"))
        if information.get("show_goal_x") is not None:
            goal = "(" + str(information.get("show_goal_x")) + "," + str(information.get("show_goal_y")) + ")"
            self.tableWidget.setItem(5, 1, QTableWidgetItem(goal))
        else:
            self.tableWidget.setItem(5, 1, QTableWidgetItem("None"))
        self.tableWidget.setItem(6, 0, QTableWidgetItem("载货情况"))
        self.tableWidget.setItem(6, 1, QTableWidgetItem(str(information.get("carry_cargo"))))

        # self.tableWidget.resizeColumnsToContents()
        # self.tableWidget.resizeRowsToContents()
        self.show()

    def closeEvent(self, event):
        self.__enable = False
        event.accept()

class point():
    def __init__(self, p, id):
        super(point, self).__init__()
        self.x = p.x()
        self.y = p.y()
        self.id = id
        self.lock = 0
        self.next_list = []
        self.cost_p = []
        self.next_path = []
        self.lock_list = []


    def set_next(self, id):
        self.next_list.append(id)
        self.lock_list.append(False)

    def get_next(self):
        return self.next_list

    def set_lock(self, id):
        for i, x in enumerate(self.next_list):
            if x == id:
                self.lock_list[i] = True
                break

    def get_lock(self, id):
        for i, x in enumerate(self.next_list):
            if x == id:
                return self.lock_list[i]

    def set_path(self, id , path):
        for i, x in enumerate(self.next_list):
            if x == id:
                self.next_path[i] = path
                break

    def get_path(self, id):
        for i, x in enumerate(self.next_list):
            if x == id:
                return self.next_path[i]

    def compu_cost(self):
        for p in self.next_list:
            cost = math.sqrt(pow((self.x-p.x), 2)+pow((self.x-p.x), 2))
            self.cost_p.append(cost)

class Graph():
    def __init__(self, parent=None):
        super(Graph, self).__init__(parent)
        self.node_list = []  # id列表

    def init_data(self, data):  # data 列表
        for i, x in enumerate(data):
            if x.get_type() == 0:        # 节点
                self.node_list.append(point(x.get_point(), x.get_id()))
            if x.get_type() == 1:  # 边
                self.node_list[x.get_start()].set_next[self.node_list[x.get_end()]]
                self.node_list[x.get_end()].set_next[self.node_list[x.get_start()]]
        for x in self.node_list:
            x.compu_cost()

    def init_json(self, data):  # data 列表

        with open(filename, 'r+', encoding='utf-8') as f:
            load_dict = json.load(f)
            number = load_dict.get("number").get("number")
            print(number)
            for i in range(0, number):
                index = str(i)
                if load_dict.get(index).get("type") == 0:  # 节点
                    self.__movement.append(
                        MNode(QPoint(load_dict.get(index).get("point.x"), load_dict.get(index).get("point.y")),
                              load_dict.get(index).get("colour"), load_dict.get(index).get("thickness"),
                              load_dict.get(index).get("radius"), load_dict.get(index).get("id"), None))
                if load_dict.get(index).get("type") == 1:  # 边
                    self.__movement.append(MLine(self.__get_node_by_index(load_dict.get(index).get("start.id")),
                                                 self.__get_node_by_index(load_dict.get(index).get("end.id")),
                                                 load_dict.get(index).get("colour"),
                                                 load_dict.get(index).get("thickness"),
                                                 load_dict.get(index).get("id")))

        for i, x in enumerate(data):
            if x.get_type() == 0:        # 节点
                self.node_list.append(point(x.get_point(), x.get_id()))
            if x.get_type() == 1:  # 边
                self.node_list[x.get_start()].set_next[self.node_list[x.get_end()]]
                self.node_list[x.get_end()].set_next[self.node_list[x.get_start()]]
        for x in self.node_list:
            x.compu_cost()

    def search_path(self, s, t):
        que = Queue()
        dis = []
        pre = []
        vis = []
        for x in range(0, len(self.node_list)):
            dis.append(10000)
            vis.append(0)
            pre.append(0)
        que.put(s)
        vis[s] = 1
        dis[s] = 0

        while not que.empty():
            ttt = que.get()
            vis[ttt] = 0
            for x, ne in enumerate(self.node_list[ttt].get_next()):
                if self.node_list[ne].lock == 0 and self.node_list[ttt].lock_list[x] > 0:
                        if dis[ne] > dis[ttt] + self.list[ttt].cost_p[x]:
                            dis[ne] = dis[ttt] + self.list[ttt].cost_p[x]
                            pre[ne] = ttt
                            if vis[ne] == 0:
                                que.put(ne)
                                vis[ne] = 1
                else:
                    if self.node_list[ne].lock == 0 and self.node_list[ttt].lock_list[x] > 0 and dis[ne] > dis[ttt] + \
                            self.node_list[ttt].cost_p[x]:
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
        for x in path1:
            self.path.append(self.dy[x])
        if s == t:
            self.path = []
        dist = dis[t]
        if dist >= 1000:
            return False
        return True









