from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

class Transition(QWidget):

    def __init__(self, Parent=None):
        super().__init__(Parent)
        self.MainWindowSize = None  # 主窗口
        self.DisplayLabelSize = None  # 客户区
        self._img_ = QPixmap()  # 改变后的地图大小
        self.to_scale = 1
        self.origin_x = None
        self.origin_y = None
        self.resolution = None
        self._img_width = 0
        self._img_height = 0
        self.img_move_x = 0
        self.img_move_y = 0

    def update_pgm(self,x ,y, resolution,scaled_img):   # 服务器AGV
        self.origin_x = x
        self.origin_y = y
        self.resolution = resolution
        self._img_ = scaled_img
        self._img_width = scaled_img.width()
        self._img_height = scaled_img.height()
        self.img_move_x = (self.MainWindowSize.width()-self._img_.width())/2+1
        self.img_move_y = 0

    def setMainWindowSize(self, mMainWindowSize):   #主窗口  设置主窗口的同时，其他几个窗口也会进行改变。
        self.MainWindowSize = mMainWindowSize
        self.DisplayLabelSize = QRect(0, 65, mMainWindowSize.width(), mMainWindowSize.height()-65)

    def getMainWindowSize(self):
        return self.MainWindowSize

    def getDisplayLabelSize(self):
        return self.DisplayLabelSize

    def getShowImglocation(self):
        x = (self.MainWindowSize.width()-self._img_.width())/2+1
        y = 0
        point = QPoint(x, y)
        return point




