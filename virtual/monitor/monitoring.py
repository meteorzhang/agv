from PyQt5.Qt import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from work.pathv3 import Graph
import math

class Monitoring_Layer(QLabel):
    sinOut = pyqtSignal(dict)

    def __init__(self, parent=None):
        super(Monitoring_Layer, self).__init__(parent)
        self.__board = None
        self.__car_Number = None
        self.__car_list = []
        self.__draw_car = None
        self.__process = None          #导航模块的状态
        self.__now_select_ = None    # 是否有小车被选择，被选择小车的索引
        self.__last_select_ = None
        self.__overall_information = None   # 是否显示全部AGV信息表
        self.__line_start = None
        self.__line_end = None
        self.__draw_line = None
        self.test_n = 0
        self.__mouse_enter_pos = None
        self.__popMenu = None            # AGV右键菜单
        self.__rightMenu_create()        # 空白处右键菜单
        self.__blank_rightMenu = None
        self.__blank_rightMenu_create()
        self.__show_agv_window = False    # 整体表
        self.__show_single_window = False  # 单个表
        self._translate = QCoreApplication.translate
        self.__show_ID = False
        self.__show_goal_id = False
        self.__goal_img = QImage("res/image/goal_22_27.png")


    def init_data(self, pgm, location):
        self.__board = pgm  # 填充一张空地图
        self.__board.fill(Qt.transparent)
        self.setFixedSize(location.width(), location.height())
        self.move(location.x(), location.y())
        self.setPixmap(self.__board)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.rightMenuShow)  # 开放右键策略
        self.tableWidget = QTableWidget()

    def init_graph(self, s):
        print(s)
        self.g = Graph(s)

    def clear_data(self):
        self.__board.fill(Qt.transparent)
        self.setPixmap(self.__board)

    def update_size(self):
        self.__board = QPixmap.fromImage(self.__board.toImage().scaled(self.width(), self.height(), Qt.IgnoreAspectRatio))  # 快速缩放
        self.setPixmap(self.__board)
        self.show()

    def set_process(self, process, parameter1=None, parameter2 = None):
        self.__process = process
        if self.__process == 0:               # 等待状态
            self.__now_select_ = None
        if self.__process == 1:               # 监控
            self.__get_car_list(parameter1)
        if self.__process == 2:               # 位姿校正
            print("位姿校正！")
            self.setMouseTracking(False)
        if self.__process == 3:               # 发布任务
            print("发布任务！")


    def refresh_UI(self, carlist):         # 刷新窗口，并刷新列表
        self.__get_car_list(carlist)
    def set_car_list(self, car_list):
        self.__car_list.clear()
        for x in car_list:
            self.__car_list.append(x)

    def execute_task(self):
        paths = []
        print("car_number:", len(self.__car_list))
        for x in self.__car_list:
            print(x)
            print(x.get_location().x())
            print(x.get_location().y())
            print(x.get_goal().x())
            print(x.get_goal().y())
            path = self.g.navigation_to_goal(x.get_location().x(), x.get_location().y(), x.get_goal().x(),x.get_goal().y())

            paths.append(path)
        return paths

    def __get_car_list(self, car_list):
        self.__car_Number = len(car_list)
        self.__car_list.clear()
        for x in car_list:
            self.__car_list.append(x)
        self.__draw_car = 1
        self.update()
        self.show()

    def mousePressEvent(self, event):  # 左键释放时进行选择判断，长按是设置，右键按下菜单选择
        super(Monitoring_Layer, self).mousePressEvent(event)
        if event.button() == Qt.RightButton:
            return
        self.__mouse_enter_pos = event.pos()
        if self.__process == 2 and self.__now_select_ is not None and self.__line_start is None:   # 位姿校正
            self.__line_start = event.pos()
        if self.__process == 3 and self.__now_select_ is not None:  # 发布任务
            if self.__check_pos(event.pos()):      # 当已有小车选中，则放弃此类操作
                return
            dict_out = {}
            dict_out["type"] = "set_goal"
            dict_out["id"] = self.__now_select_
            dict_out["goal_x"] = event.pos().x()
            dict_out["goal_y"] = event.pos().y()
            self.sinOut.emit(dict_out)
            print("发送成功！")
        if self.__process == 4:  # 发布任务
            # 鼠标按下时，获取鼠标的当前位置保存为上一次位置
            if self.__select_car(event.pos()):
                self.__draw_car = 1  # 画车体
                self.update()
                self.show()
                self.__line_start = event.pos(self.__car_list[self.__select_car].get_location())

    def mouseMoveEvent(self, event):
        super(Monitoring_Layer, self).mouseMoveEvent(event)
        if self.__process == 2 and self.__now_select_ is not None and self.__line_start is not None:
            self.__line_end = event.pos()
            self.__draw_car = 1
            self.__draw_line = 1
            self.update()
            self.show()

    def mouseReleaseEvent(self, event):  # 鼠标释放！
        if event.button() == Qt.RightButton:
            return
        if self.__select_car(event.pos()):           # 鼠标释放时进行位置判断，是否选择小车。
            self.__draw_car = 1  # 画车体
            self.update()
            self.show()

        if self.__process != 2:
            return
        if self.__now_select_ is not None and self.__line_start is not None and math.sqrt(
                pow(event.x() - self.__line_start.x(), 2) + pow(event.y() - self.__line_start.y(), 2)) < 5:
            print("设置不合理")
            self.__line_start = None
            self.__line_end = None
            self.__draw_car = 0
            self.__draw_line = 0
        if self.__now_select_ is not None and self.__line_start is not None:
            dict_out = {}
            dict_out["type"] = "jiaozheng"
            dict_out["id"] = self.__now_select_
            dict_out["start_x"] = self.__line_start.x()
            dict_out["start_y"] = self.__line_start.y()
            if event.y() == self.__line_start.y():
              jiao = math.degrees(

                math.atan((event.x() - self.__line_start.x()) / 0.01)) * (-1)
            else:
                jiao = math.degrees(

                    math.atan((event.x() - self.__line_start.x()) / (event.y() - self.__line_start.y()))) * (-1)

                #math.atan2((event.x() - self.__line_start.x()),(event.y() - self.__line_start.y()))) * (-1)
            if event.y() - self.__line_start.y() > 0:
                jiao += 180
            dict_out["jiao"] = jiao
            self.sinOut.emit(dict_out)
            for x in self.__car_list:
                if x.get_id() == self.__now_select_:
                    x.set_select_status()
                    break
            self.__last_select_ = self.__now_select_
            self.__now_select_ = None
            self.__line_start = None
            self.__draw_car = 1
            self.__draw_line = 0
            self.update()
            self.show()

    def rightMenuShow(self, evevt):
        # 添加右键菜单
        if not self.__check_pos(evevt):     # 点击空白处
            self.show_blank_ContextMenu(QCursor.pos())
        else:                               # 点击目标点
            self.showContextMenu(QCursor.pos())

    def show_blank_ContextMenu(self, enevt):
        self.__blank_rightMenu.move(enevt)
        self.__blank_rightMenu.show()

    def showContextMenu(self, enevt):
        self.__popMenu.move(enevt)
        self.__popMenu.show()

    def __blank_rightMenu_create(self):
        self.__blank_rightMenu = QMenu()
        self.__blank_rightMenu_tj = QAction('显示在线AGV_List', self)
        self.__blank_rightMenu_sc = QAction('显示在线AGV_ID', self)
        self.__blank_rightMenu_sc1 = QAction('显示目标点ID', self)
        self.__blank_rightMenu_xg = QAction('显示规划路径', self)
        self.__blank_rightMenu.addAction(self.__blank_rightMenu_tj)
        self.__blank_rightMenu.addAction(self.__blank_rightMenu_sc)
        self.__blank_rightMenu.addAction(self.__blank_rightMenu_sc1)
        self.__blank_rightMenu.addAction(self.__blank_rightMenu_xg)
        # 绑定事件
        self.__blank_rightMenu_tj.triggered.connect(self.__show_online_information)
        self.__blank_rightMenu_sc.triggered.connect(self.__show_AGV_ID)
        self.__blank_rightMenu_sc1.triggered.connect(self.__show_GOAL_ID)
        self.__blank_rightMenu_xg.triggered.connect(self.__show_planning_path)

    def __show_online_information(self):
        dict_out = {}
        if not self.__show_agv_window:
            self.__show_agv_window = True
            dict_out["type"] = "show_online_information"
            self.sinOut.emit(dict_out)
            self.__blank_rightMenu_tj.setText(self._translate("MainWindow", "关闭在线AGV列表"))
        else:
            self.__show_agv_window = False
            dict_out["type"] = "shutdown_online_information"
            self.sinOut.emit(dict_out)
            self.__blank_rightMenu_tj.setText(self._translate("MainWindow", "显示在线AGV列表"))

    def __show_AGV_ID(self):
        if self.__show_ID:
            self.__show_ID = False
            self.__blank_rightMenu_sc.setText(self._translate("MainWindow", "显示在线AGV_ID"))
        else:
            self.__show_ID = True
            self.__blank_rightMenu_sc.setText(self._translate("MainWindow", "隐藏在线AGV_ID"))

    def __show_GOAL_ID(self):
        if self.__show_goal_id:
            self.__show_goal_id = False
            self.__blank_rightMenu_sc1.setText(self._translate("MainWindow", "显示目标点ID"))
        else:
            self.__show_goal_id = True
            self.__blank_rightMenu_sc1.setText(self._translate("MainWindow", "隐藏目标点ID"))

    def __show_planning_path(self):
        print("__show_planning_path")

    def __rightMenu_create(self):
        self.__popMenu = QMenu()
        tj = QAction('显示详细信息', self)
        jz = QAction('校正位姿', self)
        sc = QAction('暂停当前任务', self)
        xg = QAction('清空任务队列', self)
        yg = QAction('设置AGV属性', self)
        self.__popMenu.addAction(tj)
        self.__popMenu.addAction(jz)
        self.__popMenu.addAction(sc)
        self.__popMenu.addAction(xg)
        self.__popMenu.addAction(yg)
        # 绑定事件
        tj.triggered.connect(self.__show_information)
        jz.triggered.connect(self.__jiaozheng_weizihechaoxiang)
        sc.triggered.connect(self.__stop_task)
        xg.triggered.connect(self.__clear_task)
        yg.triggered.connect(self.__test)

    def __show_information(self):
        if self.__now_select_ is None:
            return False
        dict_out = {}
        dict_out["type"] = "show_information"
        dict_out["id"] = self.__now_select_
        self.sinOut.emit(dict_out)

    def __jiaozheng_weizihechaoxiang(self):
        print("__jiaozheng_weizihechaoxiang")
        self.set_process(2)

    def __stop_task(self):
        if self.__now_select_ is None:
            return False
        dict_out = {}
        dict_out["type"] = "stop_task"
        dict_out["id"] = self.__now_select_
        self.sinOut.emit(dict_out)

    def __clear_task(self):
        if self.__now_select_ is None:
            return False
        dict_out = {}
        dict_out["type"] = "stop_task"
        dict_out["id"] = self.__now_select_
        self.sinOut.emit(dict_out)

    def __test(self):
        print("test!")

    def __check_pos(self, event):
        if self.__now_select_ is None:
            return False
        for x in self.__car_list:
            if self.__now_select_ == x.get_id():
                if math.sqrt(pow(x.get_location().x()-event.x(), 2)+pow(x.get_location().y()-event.y(), 2)) < 20:
                    return True
        return False

    def paintEvent(self, event):
        super(Monitoring_Layer, self).paintEvent(event)
        painter = QPainter(self)
        painter.begin(self.__board)  # 在Image画图
        # painter.setRenderHint(QPainter.Antialiasing, True)
        # painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        if self.__draw_car == 1:
            painter.setPen(Qt.NoPen)
            for x in self.__car_list:
                if x.get_show() == 1:
                    w, h = x.get_image_size()
                    start_pixel = QPoint(x.get_location().x()-w/2, x.get_location().y()-h/2)
                    painter.drawImage(start_pixel, x.get_image())
                    if self.__show_ID:
                        text_pixel = QPoint(x.get_location().x()-20, x.get_location().y()-30)
                        painter.drawImage(text_pixel, x.get_id_image())
                    if x.get_goal() is not None:
                        goal_pixel = QPoint(x.get_goal().x() - 4, x.get_goal().y() - 27)
                        painter.drawImage(goal_pixel, self.__goal_img)
                        if self.__show_goal_id:
                           goal_id_pixel = QPoint(x.get_goal().x()-12, x.get_goal().y()-42)
                           painter.drawImage(goal_id_pixel, x.get_id_image())
            self.__draw_car = 0
        if self.__draw_line == 1:
            painter.setPen(QPen(Qt.black, 5))
            painter.drawLine(self.__line_start, self.__line_end)
        painter.end()

    def __check_pos(self, pos):
        for i, x in enumerate(self.__car_list):
            if math.sqrt(pow(x.get_location().x()-pos.x(), 2)+pow(x.get_location().y()-pos.y(), 2)) < 20:
                return True
        return False

    def __select_car(self, pos):
        flag = False
        for i, x in enumerate(self.__car_list):
            if math.sqrt(pow(x.get_location().x()-pos.x(), 2)+pow(x.get_location().y()-pos.y(), 2)) < 20:
                flag = True
                if x.get_id() == self.__now_select_:  # 取消选中
                    self.__last_select_ = self.__now_select_
                    self.__now_select_ = None
                else:
                    self.__last_select_ = self.__now_select_
                    self.__now_select_ = x.get_id()
                break
        for x in self.__car_list:

            if self.__last_select_ is not None and x.get_id() == self.__last_select_:
                x.set_select_status()
                continue
            if self.__now_select_ is not None and x.get_id() == self.__now_select_:
                x.set_select_status(1)
                continue
        return flag

    # 路径分解
    def computer_path_point(self, paths):

        x = paths[::2]
        y = paths[1::2]
        path_p = []
        pose = []
        angle = round(math.atan2(y[1] - y[0], x[1] - x[0]), 2)

        x_pre = x[0]
        y_pre = y[0]
        angle_pre = angle
        pose.append(x_pre)
        pose.append(y_pre)
        pose.append(angle_pre)
        path_p.append(pose.copy())
        pose.clear()
        for i in range(len(x) - 1):
            k = math.ceil(math.sqrt(pow(y[i + 1] - y_pre, 2) + pow(x[i + 1] - x_pre, 2)) / 0.2) + 3
            angle_pre = round(math.atan2(y[i + 1] - y[i], x[i + 1] - x[i]), 2)
            for xx in range(k):
                s = 0.2 * math.sqrt(1 / (pow(((y[i + 1] - y_pre) / (x[i + 1] - x_pre)), 2) + 1))
                if (x_pre + s > x_pre and x_pre + s < x[i + 1]) or (x_pre + s > x[i] and x_pre + s < x_pre):
                    s = +s
                else:
                    s = -s
                y_pre = (y[i + 1] - y_pre) / (x[i + 1] - x_pre) * s + y_pre
                x_pre = x_pre + s
                pose.append(round(x_pre, 2))
                pose.append(round(y_pre, 2))
                pose.append(round(angle_pre, 2))
                path_p.append(pose.copy())
                pose.clear()
        pose.append(round(x[len(x) - 1], 2))
        pose.append(round(y[len(y) - 1], 2))
        pose.append(round(angle_pre, 2))
        path_p.append(pose.copy())
        pose.clear()
        return path_p








