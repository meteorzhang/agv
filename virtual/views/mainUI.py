# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QColorDialog


class Ui_MainWindow(object):

    def setupUi(self, MainWindow):

        MainWindow.setObjectName("MainWindow")    #设置顶层控件的名称
        MainWindow.resize(1018, 813)              #设置顶层空间的大小
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("res/image/restrict.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)   #设置按钮图片
        MainWindow.setWindowIcon(icon)     #给应用程序添加图标
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(self.centralwidget)      #设置中心窗口
        self.menubar = QtWidgets.QMenuBar(MainWindow)        #设置窗口的菜单栏
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1018, 23))  #设置菜单栏的几何位置
        self.menubar.setObjectName("menubar")       #设置按钮名称
        #
        self.menu_1 = QtWidgets.QMenu(self.menubar)   #设置下拉菜单栏2
        self.menu_1.setObjectName("menu_1")
        self.menu_2 = QtWidgets.QMenu(self.menubar)  #
        self.menu_2.setObjectName("menu_2")
        self.menu_3 = QtWidgets.QMenu(self.menubar)     #设置下拉菜单栏1
        self.menu_3.setObjectName("menu_3")

        self.menu_4 =  QtWidgets.QMenu(self.menubar)
        self.menu_4.setObjectName("menu_4")

        MainWindow.setMenuBar(self.menubar)    #添加到主窗口中，顶层控件




        self.loadmap = QtWidgets.QAction(self.menu_1)
        self.loadmap.setObjectName("loadmap")
        self.menu_1.addAction(self.loadmap)

        self.menubar.addAction(self.menu_1.menuAction())

        self.actionOpenmap = QtWidgets.QAction(MainWindow)
        self.actionOpenmap.setObjectName("actionOpenmap")
        self.menubar.addSeparator()


        #self.menubar.addAction(self.menu.menuAction())      #将菜单控件添加到菜单栏中



        # 生成任务
        self.actionNewTask = QtWidgets.QAction(MainWindow)
        self.actionNewTask.setObjectName("actionNewTask")
        self.menu_2.addAction(self.actionNewTask)
        #查看任务
        self.actionLookTask = QtWidgets.QAction(MainWindow)
        self.actionLookTask.setObjectName("actionLookTask")
        self.menu_2.addAction(self.actionLookTask)
        # 执行任务
        self.actionExecuteTask = QtWidgets.QAction(MainWindow)
        self.actionExecuteTask.setObjectName("actionExecuteTask")
        self.menu_2.addAction(self.actionExecuteTask)
        #again task
        self.actionAgainTask = QtWidgets.QAction(MainWindow)
        self.actionAgainTask.setObjectName("actionExecuteTask")
        self.menu_2.addAction(self.actionAgainTask)

        #piliang任务
        self.actionPauseTask = QtWidgets.QAction(MainWindow)
        self.actionPauseTask.setObjectName("actionPauseTask")
        self.menu_2.addAction(self.actionPauseTask)
        #取消任务
        self.actionCancelTask = QtWidgets.QAction(MainWindow)
        self.actionCancelTask.setObjectName("actionCancelTask")
        self.menu_2.addAction(self.actionCancelTask)
        #clear task
        # self.actionClearTask = QtWidgets.QAction(MainWindow)
        # self.actionClearTask.setObjectName("actionClearTask")
        # self.menu_2.addAction(self.actionClearTask)

        #计算时长
        #self.actionTime = QtWidgets.QAction(MainWindow)
        #self.actionTime.setObjectName("actionTime")
        #self.menu_2.addAction(self.actionTime)

        self.menubar.addSeparator()
        self.menubar.addAction(self.menu_2.menuAction())
        #
        #select path planning
        self.actionAI = QtWidgets.QAction(MainWindow)
        self.actionAI.setObjectName("actionAI")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("res/image/state_gray.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionAI.setIcon(icon1)
        self.menu_3.addAction(self.actionAI)

        self.actionAstar = QtWidgets.QAction(MainWindow)
        self.actionAstar.setObjectName("actionAstar")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("res/image/state_gray.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionAstar.setIcon(icon2)
        self.menu_3.addAction(self.actionAstar)

        self.actionDijkstra = QtWidgets.QAction(MainWindow)
        self.actionDijkstra.setObjectName("actionDijkstra")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("res/image/state_gray.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionDijkstra.setIcon(icon3)
        self.menu_3.addAction(self.actionDijkstra)

        self.actionBellman = QtWidgets.QAction(MainWindow)
        self.actionBellman.setObjectName("actionBellman")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("res/image/state_gray.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionBellman.setIcon(icon4)
        self.menu_3.addAction(self.actionBellman)

        self.menubar.addSeparator()
        self.menubar.addAction(self.menu_3.menuAction())

        self.actionData =  QtWidgets.QAction(MainWindow)
        self.actionData.setObjectName("actionData")
        self.menu_4.addAction(self.actionData)
        self.menubar.addAction(self.menu_4.menuAction())

        self.retranslateUi(MainWindow)  #
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        self._translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(self._translate("MainWindow", "AGV"))
        self.menu_1.setTitle(self._translate("MainWindow", "打开"))
        self.loadmap.setText(self._translate("MainWindow", "导入地图"))

        self.menu_2.setTitle(self._translate("MainWindow", "调度"))
        self.actionNewTask.setText(self._translate("MainWindow", "新增任务"))
        self.actionLookTask.setText(self._translate("MainWindow", "查看任务"))
        self.actionExecuteTask.setText(self._translate("MainWindow", "执行任务"))
        self.actionAgainTask.setText(self._translate("MainWindow","重置任务"))
        self.actionPauseTask.setText(self._translate("MainWindow","批量加载"))
        self.actionCancelTask.setText(self._translate("MainWindow","停止任务"))
        #self.actionClearTask.setText(self._translate("MainWindow","清空任务"))

        self.menu_3.setTitle(self._translate("MainWindow","算法选择"))
        self.actionAI.setText(self._translate("MainWindow","DRL"))
        self.actionAstar.setText(self._translate("MainWindow","A*"))
        self.actionDijkstra.setText(self._translate("MainWindow","dijkstra"))
        self.actionBellman.setText(self._translate("MainWindow","bellmanFord"))
        #data
        self.menu_4.setTitle(self._translate("MainWindow","数据统计"))
        self.actionData.setText(self._translate("MainWindow","数据统计"))



