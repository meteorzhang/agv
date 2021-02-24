# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import sys
if sys.version_info >= (3,0):
    from queue import Queue
else:
    from Queue import Queue

import numpy as np
import scipy.misc as misc
import sys

sys.path.append('../')
from navi.GoConfig import GoConfig
from navi.GoEmulator import GoEmulator
from navi.GoStatus import FullStatus

#from environment.environment import Environment


'''
notes:
  1.status: static and dynamic
  2.provide rewards of action/status;
  3.display 
by zoulibing. 
  
'''


class GoEnvironment:
    def __init__(self, id=-1, shared_env=None, actions=None, enable_show=True):
        self.id = id
        # add environment emulator
        self.shared_env = shared_env
        self.nb_frames = GoConfig.STACKED_FRAMES
        self.previous_state = None
        self.current_state = None
        self.total_reward = 0
        self.visualize = enable_show
        self.actions = actions
        self.step_time = GoConfig.STEPPING_TIME
        self.env = GoEmulator( self.shared_env, self.visualize)
        # self.reset()

    # 将数据转化成数组.curent status include static status and dynamic status.

    def get_lidar(self, x, y):
        print("_______++++++++++++++++++++++++++++________________+++++++++++++++++++++")
        print("_______++++++++++++++++++++++++++++________________+++++++++++++++++++++")

        print("_______++++++++++++++++++++++++++++________________+++++++++++++++++++++")

        print("_______++++++++++++++++++++++++++++________________+++++++++++++++++++++")

        print("_______++++++++++++++++++++++++++++________________+++++++++++++++++++++")

        return self.env.get_lidar(x, y)





