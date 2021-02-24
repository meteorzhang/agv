
from virtual.GoNavigation.Navigator_cbs import Navigator

from virtual.monitor.monitoring_unit import Agv_Car

from PyQt5.QtCore import QPoint

navi = Navigator("final/road_grid_2.txt")

car_list = []

for i in range(4):
    car_list.append(Agv_Car())
    car_list[i].set_id(i)

car_list[0].set_location_angle(QPoint(20, 0), 0)

car_list[0].set_goal(QPoint(20, 60))

car_list[1].set_location_angle(QPoint(0, 20), 0)

car_list[1].set_goal(QPoint(60, 20))

car_list[2].set_location_angle(QPoint(40, 60), 0)

car_list[2].set_goal(QPoint(40, 0))

car_list[3].set_location_angle(QPoint(60, 40), 0)

car_list[3].set_goal(QPoint(0, 40))

print(navi.navigation_simulation(car_list[0]))
print(navi.navigation_simulation(car_list[1]))
print(navi.navigation_simulation(car_list[2]))
print(navi.navigation_simulation(car_list[3]))

