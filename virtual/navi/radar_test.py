

import sys
sys.path.append('../')
from navi.Create_virtual_radar import Virtual_radar
from pysim2d import pysim2d






allow_area_url = "/home/brook/NaviPlanningEmulatorGrid/test/custom_map/11_13/tongxing_grid.txt"

emulator = Virtual_radar()

emulator.init_data(allow_area_url)  # 读取通行域的信息

#print(len(emulator.get_lidar(35,35)))
#print(emulator.get_lidar(105*0.05, 35*0.05))

self_emulator = pysim2d.pysim2d()

self_map_url = "/home/brook/NaviPlanningEmulatorGrid/test/custom_map/11_13/init_gnuplot_map.png"
self_one_channel_map_pic_str = "/home/brook/NaviPlanningEmulatorGrid/test/custom_map/11_13/init_gnuplot_map.png"
self_road_url = "/home/brook/NaviPlanningEmulatorGrid/test/custom_map/11_13/road_grid.txt"
self_allow_area_url = "/home/brook/NaviPlanningEmulatorGrid/test/custom_map/11_13/tongxing_grid.txt"

self_emulator.init(self_one_channel_map_pic_str, self_allow_area_url, self_road_url, False, False,False)
self_emulator.set_parameter(0.5, 0.5, 0.1, 0.5, 1, 20)  # 12/0.05 = 240

self_emulator.set_title(0, 0)
#print(len(self_emulator.get_lidar(35, 35)))
print(self_emulator.get_lidar(105*0.05, 35*0.05))