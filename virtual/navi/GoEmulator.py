import sys
# sys.path.append('../')

from virtual.navi.GoConfig import GoConfig
from virtual.navi.GoStatus import AgentState
import numpy as np
from virtual.navi.Create_virtual_radar import Virtual_radar


class GoEmulator:
    def __init__(self, share_env=None, enable_show=False):
        self.emulator = None

        self.share_env = share_env

        if self.share_env is not None:

            self.allow_area_url = "11_13/tongxing_grid.txt"

            self.enable_show = enable_show

            self.emulator = Virtual_radar()

            self.emulator.init_data(self.allow_area_url)  # 读取通行域的信息

    def get_lidar(self, x, y):
        print("_______++++++++++++++++++++++++++++________________+++++++++++++++++++++")
        print("_______++++++++++++++++++++++++++++________________+++++++++++++++++++++")

        print("_______++++++++++++++++++++++++++++________________+++++++++++++++++++++")

        print("_______++++++++++++++++++++++++++++________________+++++++++++++++++++++")

        print("_______++++++++++++++++++++++++++++________________+++++++++++++++++++++")
        return self.emulator.get_lidar(x, y)

