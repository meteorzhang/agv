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

from threading import Thread

import numpy as np

from virtual.navi.GoConfig import GoConfig

import time


class ThreadPredictor(Thread):
    def __init__(self, server, id):
        super(ThreadPredictor, self).__init__()
        self.setDaemon(True)

        self.id = id
        self.server = server
        self.exit_flag = False

    def run(self):
        # while not self.exit_flag:
        #     ids = []
        #     maps = []
        #     selfs = []
        #     others = []
        #     observations = []
        #     other_num = []
        #     size = 0
        #     if self.server.prediction_q.empty():
        #         time.sleep(0.001)
        #         continue
        #
        #     while size < GoConfig.PREDICTION_BATCH_SIZE and not self.server.prediction_q.empty():
        #         id, state = self.server.prediction_q.get()
        #         np_state = state.to_numpy_array()
        #         ids.append(id)
        #         maps.append(np_state['map'])
        #         selfs.append(np_state['self'])
        #         observations.append(np_state['observation'])
        #         other_agent_state = np_state['other']
        #         other_static = np.zeros((GoConfig.A_MAP_MAX_AGENT_SIZE-1, GoConfig.LOCAL_OTHER_STATUS_SIZE), dtype=np.float)
        #         o_num = len(other_agent_state)
        #         #print("o_num: ", o_num)
        #
        #         # print("-----------other: ",ids, other_agent_state)
        #         # print("-----------selfs: ", ids, selfs)
        #
        #         #input()
        #         #print("other_agent_state: ", other_agent_state)
        #
        #         for idx in range(o_num):
        #             print("idx: ", idx)
        #             other_static[idx] = np.array(other_agent_state[idx])
        #         others.append(other_static)
        #         other_num.append(o_num)
        #         size += 1
        #     if len(ids) > 0:
        #
        #         s_t = time.time()
        #         p, v = self.server.model.predict_p_and_v(selfs, others, observations, other_num, maps)
        #         print("预测耗时：　", time.time()-s_t)
        #
        #     for i in range(size):
        #             #print("ids[i]:", ids[i])
        #             agent = self.server.get_agent_by_id(ids[i])
        #             if agent is not None:
        #                 agent.wait_q.put((p[i], v[i]))

        # list for training
        while not self.exit_flag:
            if self.server.prediction_q.empty():
                time.sleep(0.001)
                continue

            ids = []
            # maps = []
            selfs = []
            others = []
            observations = []
            other_num = []
            size = 0

            while size < GoConfig.PREDICTION_BATCH_SIZE and not self.server.prediction_q.empty():
                id, state = self.server.prediction_q.get()
                np_state = state.to_numpy_array()
                ids.append(id)
                # maps.append(np_state['map'])
                selfs.append(np_state['self'])
                observations.append(np_state['observation'])
                other_agent_state = np_state['other']
                other_static = np.zeros((GoConfig.A_MAP_MAX_AGENT_SIZE-1, GoConfig.LOCAL_OTHER_STATUS_SIZE), dtype=np.float)
                o_num = len(other_agent_state)
                for idx in range(o_num):
                    other_static[idx] = np.array(other_agent_state[idx])
                others.append(other_static)
                other_num.append(o_num)
                size += 1
            if len(ids) > 0:
                p, v = self.server.model.predict_p_and_v(selfs, others, observations, other_num)

            for i in range(size):
                    agent = self.server.get_agent_by_id(ids[i])
                    if agent is not None:
                        agent.wait_q.put((p[i], v[i]))
