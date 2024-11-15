import os
import sys
sys.path.append(os.path.dirname(__file__))
from app.arrayexp import ArrayExp as Device
import numpy as np


class SDK():
    def __init__(self, chip_num=1):
        # self.ARRAYs = ARRAYs_4K(chip_num)
        pass

    def connect_device(self):
        self.device = Device.ArrayExp()
        self.connect_state = self.device.myapp.client.isConnected()

        return self.connect_state

    def disconnect_device(self):
        self.device.close()

    def read_array(self, arrayid, row_start_index, row_end_index):
        pass

    def config(self, paras):
        pass

    def register_call_back(self, caller):
        return

    def sdk_stop(self):
        return

    def is_device_busy(self):
        return