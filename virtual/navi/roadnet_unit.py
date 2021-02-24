from PyQt5.Qt import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class MNode(QPoint):
    def __init__(self, point, colour, thickness, radius, id, goal_state):
        super(MNode, self).__init__()
        self.__point = point
        self.__colour = colour
        self.__thickness = thickness
        self.__radius = radius
        self.__type = 0
        self.__id = id
        self.__erasure = 0
        self.__goal_state = goal_state

    def set_point(self, point):
        self.__point = point

    def set_colour(self, colour):
        self.__colour = colour

    def set_thickness(self, thickness):
        self.__thickness = thickness

    def set_radius(self, radius):
        self.__colour = radius

    def set_erasure(self, erasure):
        self.__erasure = erasure

    def get_point(self):
        return self.__point

    def get_thickness(self):
        return self.__thickness

    def get_colour(self):
        return self.__colour

    def get_radius(self):
        return self.__radius

    def get_type(self):
        return self.__type

    def get_id(self):
        return self.__id

    def to_json(self):
        return {
          "point.x": self.__point.x(),
          "point.y": self.__point.y(),
          "colour": self.__colour,
          "thickness": self.__thickness,
          "radius": self.__radius,
          "type": self.__type,
          "id": self.__id,
        }


    def get_erasure(self):
        return self.__erasure

class MLine(QLine):
    id = 0
    copy_id = 0
    def __init__(self, start=None, end=None, colour=None, thickness=None, mid=None):
        super(MLine, self).__init__()
        self.__start = start
        self.__end = end
        self.__colour = colour
        self.__thickness = thickness
        self.__type = 1
        self.__status = 0
        self.__move = 0
        self.__erasure = 0
        if mid == None:
            self.__id = MLine.id + 1
            self.__id = MLine.id
        else:
            self.__id = mid

    # def update_start_end(self,):


    def set_start(self, start):
        self.__start = start

    def set_end(self, end):
        self.__end = end

    def set_colour(self, colour):
        self.__colour = colour

    def set_thickness(self, thickness):
        self.__thickness = thickness

    def set_status(self, status):
        self.__status = status

    def set_move(self, move):
        self.__move = move

    def set_erasure(self, erasure):
        self.__erasure = erasure

    def set_id(self,id):
        self.__id = id

    def get_start(self):
        return self.__start

    def get_end(self):
        return self.__end

    def get_thickness(self):
        return self.__thickness

    def get_colour(self):
        return self.__colour

    def get_type(self):
        return self.__type

    def get_status(self):
        return self.__status

    def get_move(self):
        return self.__move

    def get_id(self):
        return self.__id

    def get_erasure(self):
        return self.__erasure

    def get_copy(self):
        copy = MLine()
        copy.set_start(self.get_start())
        copy.set_end(self.get_end())
        copy.set_thickness(self.get_thickness())
        copy.set_colour(self.get_colour())
        copy.__type = 1
        copy.__status = 0
        copy.__move = 0
        copy.__id = self.get_id()
        return copy

    def to_json(self):
        return {
          "start.id": self.__start.get_id(),
          "end.id": self.__end.get_id(),
          "colour": self.__colour,
          "thickness": self.__thickness,
          "type": self.__type,
          "id": self.__id
        }

    def to_txt(self):
        return ""+str(self.__start.x())+" "+str(self.__start.y())+" "+str(self.__end.x())+" "+str(self.__end.y())






