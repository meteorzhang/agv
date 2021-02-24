# -*- coding: utf-8 -*-
import os
import json
import sys
import _thread
import time
from PyQt5.QtWidgets import *
sys.path.append("../")
sys.path.append('../work')
import yaml
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from virtual.work.roadnet_unit import MNode, MLine
from virtual.monitor.monitoring_unit import Agv_Car

car_list = []

class Map_Layer(QLabel):

    def __init__(self, parent=None):
        super(Map_Layer, self).__init__(parent)
        self.draw_highlight = False
        self.light_image = None
        self.left_click = False
        self.paint_map = False
        self.paint_road = False
        self.paint_robot = False
        self.exe_task = False
        self.road_origin_width = 0
        self.road_origin_height = 0
        self.point_ra = 0
        self.yaml = ''
        self.__car_list = []
        self.__movement_p = []
        self.__movement_g = []
        self.__movement_l = []
        self.power_point = []
        self.ku_point = []
        self.light_list = []




    def init_data(self,pixel_translation):
        self.pixel_translation = pixel_translation
        self.__draw_highlight()

    def __draw_highlight(self):
        self.light_image = QImage(50, 50, QImage.Format_ARGB32)
        self.light_image.fill(Qt.transparent)
        painter = QPainter()
        painter.begin(self.light_image)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.setPen(QPen(QColor(25, 0, 90, 200), 4))
        painter.resetTransform()
        painter.translate(self.light_image.width() / 2, self.light_image.height() / 2)
        painter.drawEllipse(-18, -24, 25, 25)
        painter.end()

    def execute_task(self):
        if self.__car_list == [] or self.map_name == None:
            return
        self.paint_robot = True
        i = 0
        for x in self.__car_list:
            if x.get_isbind() == 0:
                x.set_goal(x.get_location())
                i = i+1


    def paintEvent(self, e):
        painter = QPainter()
        painter.begin(self)
        #print("-----------paintEvent----------------------")
        if self.paint_map:
            self.draw_img(painter)
        if self.paint_map and self.paint_road:
            self.draw_road(painter)
        if self.paint_map and self.paint_road and self.paint_robot:
            self.draw_robot(painter)
        if self.draw_highlight and self.paint_map and self.paint_road and self.paint_robot:
            self.draw_light(painter)
        painter.end()

    def draw_img(self, painter):
        painter.drawPixmap(self.point, self.scaled_img)

    def draw_road(self, painter):
        for x in self.__movement_l:
            start_p = x.get_start().get_point()
            end_p = x.get_end().get_point()

            start_p_x = x.get_start().get_point().x()
            start_p_y = x.get_start().get_point().y()
            start_p_x = (start_p_x / self.road_origin_width) * self.scaled_img.width() + self.point.x()
            start_p_y = (start_p_y / self.road_origin_height) * self.scaled_img.height() + self.point.y()
            end_p_x = x.get_end().get_point().x()
            end_p_y = x.get_end().get_point().y()
            end_p_x = (end_p_x / self.road_origin_width) * self.scaled_img.width() + self.point.x()
            end_p_y = (end_p_y / self.road_origin_height) * self.scaled_img.height() + self.point.y()
            trans_start_point = QPoint(start_p_x,start_p_y)
            trans_end_point = QPoint(end_p_x, end_p_y)

            painter.setPen(QPen(QColor(x.get_colour()), x.get_thickness()))
            painter.drawLine(trans_start_point, trans_end_point)

            self.line = QLineF(trans_start_point, trans_end_point)
            v = self.line.unitVector()
            v.setLength(8)
            v.translate(QPointF(self.line.dx()/2, self.line.dy()/2))

            n = v.normalVector()
            n.setLength(n.length() * 0.5)
            n2 = n.normalVector().normalVector()

            p1 = v.p2()
            p2 = n.p2()
            p3 = n2.p2()
            painter.drawPolygon(p1, p2, p3)


        for x in self.__movement_p:
            painter.setPen(QPen(QColor(x.get_colour()), x.get_thickness()))
            qpoint_x = x.get_point().x() - x.get_radius() / 2
            qpoint_y = x.get_point().y() - x.get_radius() / 2

            qpoint_x = (qpoint_x/self.road_origin_width)*self.scaled_img.width() + self.point.x()
            qpoint_y = (qpoint_y /self.road_origin_height)*self.scaled_img.height() + self.point.y()
            painter.drawEllipse(qpoint_x, qpoint_y,x.get_radius(), x.get_radius())  # 圆心位置
        i = 0
        for x in self.__movement_g:
            if i > 77:
                break
            painter.setPen(QPen(QColor(x.get_colour()), x.get_thickness()))
            self.point_ra = x.get_radius()
            qpoint_x = x.get_point().x() - x.get_radius() / 2
            qpoint_y = x.get_point().y() - x.get_radius() / 2

            qpoint_x = (qpoint_x/self.road_origin_width)*self.scaled_img.width() + self.point.x()
            qpoint_y = (qpoint_y /self.road_origin_height)*self.scaled_img.height() + self.point.y()
            painter.drawEllipse(qpoint_x, qpoint_y,x.get_radius(), x.get_radius())  # 圆心位置
            rect = QRect(qpoint_x, qpoint_y - 20, 30, 20)
            # painter.drawRect(rect)
            painter.setPen(QColor(25, 0, 90, 200))
            painter.setFont(QFont("Decorative", 10))
            str_i = '站' + str(i)
            painter.drawText(rect, Qt.AlignCenter, str_i)
            i = i + 1
        i = 0
        for x in self.ku_point:
            painter.setPen(QPen(QColor(x.get_colour()), x.get_thickness()))
            self.point_ra = x.get_radius()
            qpoint_x = x.get_point().x() - x.get_radius() / 2
            qpoint_y = x.get_point().y() - x.get_radius() / 2
            qpoint_x = (qpoint_x/self.road_origin_width)*self.scaled_img.width() + self.point.x()
            qpoint_y = (qpoint_y /self.road_origin_height)*self.scaled_img.height() + self.point.y()
            painter.drawEllipse(qpoint_x, qpoint_y,x.get_radius(), x.get_radius())  # 圆心位置
            rect = QRect(qpoint_x-5, qpoint_y - 20, 30, 20)
            # painter.drawRect(rect)
            painter.setPen(QColor(25, 0, 90, 200))
            painter.setFont(QFont("Decorative", 10))
            if i ==0:
                str_i = '入库点'
            else:
                str_i = '出库点'
            painter.drawText(rect, Qt.AlignCenter, str_i)
            i = i + 1
        i = 0
        for x in self.power_point:
            painter.setPen(QPen(QColor(x.get_colour()), x.get_thickness()))
            self.point_ra = x.get_radius()
            qpoint_x = x.get_point().x() - x.get_radius() / 2
            qpoint_y = x.get_point().y() - x.get_radius() / 2
            qpoint_x = (qpoint_x / self.road_origin_width) * self.scaled_img.width() + self.point.x()
            qpoint_y = (qpoint_y / self.road_origin_height) * self.scaled_img.height() + self.point.y()
            painter.drawEllipse(qpoint_x, qpoint_y, x.get_radius(), x.get_radius())  # 圆心位置
            rect = QRect(qpoint_x-5, qpoint_y - 40, 30, 20)
            # painter.drawRect(rect)
            painter.setPen(QColor(25, 0, 90, 200))
            painter.setFont(QFont("Decorative", 10))
            # str_i = '充电' + str(i)
            # painter.drawText(rect, Qt.AlignCenter, str_i)
            i = i + 1

    def draw_robot(self, painter):
        for x in self.__car_list:
            # if x.get_id() == 19:
            #     print('car id 19 ',x.get_location().x(),x.get_location().y())
            tmp = x.get_location()
            qpoint_x = (tmp.x() / self.road_origin_width) * self.scaled_img.width() + self.point.x()-x.get_image().width()/2
            qpoint_y = (tmp.y() / self.road_origin_height) * self.scaled_img.height() + self.point.y()-x.get_image().height()/2
            if x.get_is_package() == 1:
                painter.drawImage(QPoint(qpoint_x, qpoint_y), x.get_package_image())
            else:
                painter.drawImage(QPoint(qpoint_x, qpoint_y), x.get_image())
            painter.drawImage(QPoint(qpoint_x, qpoint_y - 15), x.get_id_image())

    def draw_light(self, painter):
        for x in self.light_list:
            painter.setPen(QPen(QColor(x.get_colour()), x.get_thickness()))
            qpoint_x = x.get_point().x() - x.get_radius() / 2
            qpoint_y = x.get_point().y() - x.get_radius() / 2
            qpoint_x = (qpoint_x / self.road_origin_width) * self.scaled_img.width() + self.point.x()
            qpoint_y = (qpoint_y / self.road_origin_height) * self.scaled_img.height() + self.point.y()
            painter.drawImage(QPoint(qpoint_x-16, qpoint_y-8), self.light_image)

    def set_light(self,a,b):
        self.light_list.append(a);
        self.light_list.append(b);
        self.draw_highlight = True
        while self.updatesEnabled():
            self.update()
            break
    def clear_light(self):
        self.light_list.clear()
        self.draw_highlight = False
        while self.updatesEnabled():
            self.update()
            break


    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.left_click = True
            self._startPos = e.pos()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.left_click = False
        elif e.button() == Qt.RightButton:
            self.point = self.pixel_translation.getShowImglocation()
            self.scaled_img = self.img.scaled(self.origin_size)
            while self.updatesEnabled():
                self.update()
                break

    def mouseMoveEvent(self, e):  # 重写移动事件
        if self.left_click:
            self._endPos = e.pos() - self._startPos
            self.point = self.point + self._endPos
            self._startPos = e.pos()
            while self.updatesEnabled():
                self.update()
                break

    def wheelEvent(self, e):
        if e.angleDelta().y() < 0:
            # 放大图片
            self.scaled_img = self.img.scaled(self.scaled_img.width()-5, self.scaled_img.height()-5)
            new_w = e.x() - (self.scaled_img.width() * (e.x() - self.point.x())) / (self.scaled_img.width() + 5)
            new_h = e.y() - (self.scaled_img.height() * (e.y() - self.point.y())) / (self.scaled_img.height() + 5)
            self.point = QPoint(new_w, new_h)
            while self.updatesEnabled():
                self.update()
                break
        elif e.angleDelta().y() > 0:
            # 缩小图片
            self.scaled_img = self.img.scaled(self.scaled_img.width()+5, self.scaled_img.height()+5)
            new_w = e.x() - (self.scaled_img.width() * (e.x() - self.point.x())) / (self.scaled_img.width() - 5)
            new_h = e.y() - (self.scaled_img.height() * (e.y() - self.point.y())) / (self.scaled_img.height() - 5)
            self.point = QPoint(new_w, new_h)
            while self.updatesEnabled():
                self.update()
                break

    def resizeEvent(self, e):
        if self.parent is not None:
            self.scaled_img = self.img.scaled(self.origin_size)
            self.point = self.pixel_translation.getShowImglocation()
            self.update()

    def LoadMap(self):
        self.map_name = self.browse_road("打开地图！", "*.png")
        if os.path.exists(self.yaml):  #为真代表之前打开过地图，这次为重新打开地图
            self.yaml = self.map_name[0:-4]+".yaml"#重新打开失败后，应该清空界面显示，待完成
            if os.path.exists(self.yaml):
                f = open(self.yaml,'r',encoding="utf-8")
                data = f.read()
                f.close()
                yaml_data = yaml.load(data, Loader=yaml.FullLoader)
                self.yaml_x = yaml_data['origin'][0]
                self.yaml_y = yaml_data['origin'][1]
                self.yaml_resolution = yaml_data['resolution']
                self.img = QPixmap(self.map_name)
                self.wid = QDesktopWidget().availableGeometry().width() - self.img.width()
                self.origin_size = self.img.size()
                self.origin_point = QPoint(int(self.wid / 2), 0)
                self.scaled_img = self.img.scaled(self.origin_size)
                self.point = self.origin_point
                self.pixel_translation.update_pgm(self.yaml_x,self.yaml_y,self.yaml_resolution,self.scaled_img)
                self.paint_map = True
                print("select roadddd")

                while self.updatesEnabled():
                    self.update()
                    break
                return self.map_name,self.img
            else:
                QMessageBox.information(self, "提示", "yaml文件不存在", QMessageBox.Yes, QMessageBox.Yes)
                self.paint_map = False
                self.paint_road = False
                return None,None
        else:
            self.yaml = self.map_name[0:-4]+".yaml"
            if os.path.exists(self.yaml):
                f = open(self.yaml,'r',encoding="utf-8")
                data = f.read()
                f.close()
                yaml_data = yaml.load(data)
                self.yaml_x = yaml_data['origin'][0]
                self.yaml_y = yaml_data['origin'][1]
                self.yaml_resolution = yaml_data['resolution']
                self.img = QPixmap(self.map_name)
                self.wid = QDesktopWidget().availableGeometry().width() - self.img.width()
                self.origin_size = self.img.size()
                self.origin_point = QPoint(int(self.wid / 2), 0)
                self.scaled_img = self.img.scaled(self.origin_size)
                self.point = self.origin_point
                self.pixel_translation.update_pgm(self.yaml_x, self.yaml_y, self.yaml_resolution, self.scaled_img)
                self.paint_map = True
                self.show()
                self.select_road(self.map_name)
                print("select road",self.map_name)
                self.yaml=""

                #self.clear()
                self.update()
                return self.map_name,self.img
            else:
                QMessageBox.information(self, "提示", "yaml文件不存在", QMessageBox.Yes, QMessageBox.Yes)
                self.paint_map = False
                self.paint_road = False
                return None,None

    def select_road(self, filename):
        if filename and os.path.exists(filename):
            filename = str(filename).replace("png", "json")
            with open(filename, 'r+', encoding='utf-8') as f:
                load_dict = json.load(f)
                # if load_dict.get("mapName").get("mapName") != mapName:
                #     QMessageBox.information(self, "提示", "地图不匹配！！！", QMessageBox.Yes, QMessageBox.Yes)
                #     return
                self.__road_json_path = filename
                self.road_origin_width = load_dict.get("pgm_w").get("pgm_x")
                self.road_origin_height = load_dict.get("pgm_h").get("pgm_x")
                number_p = load_dict.get("number_p").get("number_p")
                number_goal = load_dict.get("number_goal").get("number_goal")
                number_power = load_dict.get("number_power").get("number_power")
                number_ku = load_dict.get("number_ku").get("number_ku")
                for i in range(0, number_goal):
                    index = 'goal' + str(i)

                    self.__movement_g.append(
                        MNode(QPoint(load_dict.get(index).get("point.x"), load_dict.get(index).get("point.y")),
                              load_dict.get(index).get("colour"), load_dict.get(index).get("thickness"),
                              load_dict.get(index).get("radius"), load_dict.get(index).get("id"), None))

                for i in range(0, number_p):
                    index = 'p' + str(i)
                    self.__movement_p.append(MNode(QPoint(load_dict.get(index).get("point.x"), load_dict.get(index).get("point.y")), load_dict.get(index).get("colour"), load_dict.get(index).get("thickness"), load_dict.get(index).get("radius"), load_dict.get(index).get("id"), None))

                for i in range(0, number_power):
                    index = 'power' + str(i)
                    self.power_point.append(MNode(QPoint(load_dict.get(index).get("point.x"), load_dict.get(index).get("point.y")), load_dict.get(index).get("colour"), load_dict.get(index).get("thickness"), load_dict.get(index).get("radius"), load_dict.get(index).get("id"), None))

                for i in range(0, number_ku):
                    index = 'ku' + str(i)
                    self.ku_point.append(MNode(QPoint(load_dict.get(index).get("point.x"), load_dict.get(index).get("point.y")), load_dict.get(index).get("colour"), load_dict.get(index).get("thickness"), load_dict.get(index).get("radius"), load_dict.get(index).get("id"), None))


                number_l = load_dict.get("number_l").get("number_l")
                for i in range(0, number_l):
                    index = 'l' + str(i)
                    self.__movement_l.append(MLine(self.__get_node_by_index(load_dict.get(index).get("start.id")),
                                                   self.__get_node_by_index(load_dict.get(index).get("end.id")),
                                                   load_dict.get(index).get("colour"),
                                                   load_dict.get(index).get("thickness"),
                                                   load_dict.get(index).get("id")))
            self.paint_road = True
        else:
            QMessageBox.information(self, "提示", "输入为空或者文件不存在", QMessageBox.Yes, QMessageBox.Yes)
            self.paint_road = False


    def __get_node_by_index(self, tid):
        for x in self.__movement_p:
            if x.get_id() == tid:
                return x
        for x in self.__movement_g:
            if x.get_id() == tid:
                return x
        return None

    def browse_road(self, path, dataType):
        filename = QFileDialog.getOpenFileName(self, path, "res/map/", dataType)
        return filename[0]

    def get_car_list(self,list):

        self.__car_list =  list
        self.paint_robot = True

        if self.updatesEnabled():
            #print("------while 循环------------------")
            self.update()
            #break

    def output_car_list(self):
        return self.__car_list

    def qpoint_tf(self,q_point):
        qpoint_x = (q_point.x() / self.origin_size.width()) * self.scaled_img.width() + self.point.x()
        qpoint_y = (q_point.y() / self.origin_size.height()) * self.scaled_img.height() + self.point.y()
        return QPoint(qpoint_x,qpoint_y)

    def draw__extra_robot(self, painter,q_point,q_image):
        q_target = self.qpoint_tf(q_point)
        painter.drawImage(q_target, q_image)

    def clear_carlist(self):
        self.__car_list.clear()
        self.paint_robot = False
        while self.updatesEnabled():
            self.update()
            break
    def clear_all(self):
        self.__movement_l.clear()
        self.__movement_g.clear()
        self.__movement_p.clear()