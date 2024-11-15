import linecache
import os
import sys
sys.path.append(os.path.dirname(__file__))
import time
import copy
import numpy as np
import csv
import apps
import arrayexp
import arrayparams
import reram_register


# ARRAY info
# POSITIVE_READ = 0
# NEGATIVE_READ = 1
class ARRAY4K:
    ID = 1#SDK.ARRAY1
    ROW = 128
    COL= 32
    SIZE = ROW * COL
    SHAPE = ROW, COL
    START_ADDR = [0, 0]
    END_ADDR = [ROW-1, COL-1]
    # Test data
    DATA = np.zeros(SHAPE) #np.reshape(range(SIZE), (ROW, COL))+1

    def __init__(self, unusable_addr,):
        #  unusable_addr = [[1, 2]]  : row 1, col 2 is unusable
        self.unusable_mask = np.zeros((self.ROW, self.COL))
        if None not in unusable_addr:
            self.unusable_mask[unusable_addr] = 1


UNUSED_ADDR_ARRAY1 =  [[None]]
UNUSED_ADDR_ARRAY2 =  [[None]]
UNUSED_ADDR_ARRAY3 =  [[None]]
UNUSED_ADDR = UNUSED_ADDR_ARRAY1+UNUSED_ADDR_ARRAY2+UNUSED_ADDR_ARRAY3

POSITIVE = 1
NEGATIVE = -1
NULLOP = 0

class Platform_4k():
    def __init__(self, Exp=None, used_array=3):
        self.Exp = Exp
        self.is_test = True
        self.connect_state = False

        self.order_number = 0  # assigned cells' count: order number / serial number
        self.addr_pointer = [0, 0, 0]  # [Array, Row, Col]
        self.used_array = used_array
        self.arrays = []
        self.unused_addr_sys = np.zeros(ARRAY4K.SIZE*self.used_array)
        for array_idx in range(self.used_array):
            array = ARRAY4K(UNUSED_ADDR[array_idx])
            self.unused_addr_sys[array_idx*ARRAY4K.SIZE:(array_idx+1)*ARRAY4K.SIZE]= \
                                                    array.unusable_mask.flatten()
            self.arrays.append(array)
        self.op_arrays = np.zeros(ARRAY4K.SIZE*self.used_array)
        # layer
        self.layers = []
        self.layers_ID = 0
        # layer info
        self.layer = {'id': [],
                        'size': [],
                        'total_cells': 0,
                        'order_number_start': 0,
                        'order_number_end':0,
                        'addr_start': [0, 0, 0],  # # [Array, Row, Col]
                        'addr_end': [0, 0, 0],  # # [Array, Row, Col]
                        'unused_addr_mask': np.zeros(1),
                        'array_infos': None,  # addr_start_deployed, addr_end_deployed
                        'sample_count':0,  # batch load mode
                      }

    def __del__(self):
        if not self.is_test:
            print('Array closing...')
            self.array.close()
    def platform_init(self, init_states=None,batch_mode_path=None, is_test=False):
        self.is_test = is_test
        self.saved_read_data_path = batch_mode_path
        if self.saved_read_data_path is None:
            self.batch_mode = False
            if not is_test:
                if not self.connect_state:
                    print('platform initial ...')
                    self.array = arrayexp.ArrayExp()
                    self.params = arrayparams.ArrayParams()

                    if init_states is not None:
                        print('init array conductance randomly ...')
                        for chipNum in range(self.used_array):
                            print('Chip num:'+ str(chipNum+1))
                            addr = apps.AddrInfo(chipNum + 1, 0, 128, 0, 32)

                            mapPara = apps.MapPara()
                            mapPara.errorHigh = 0.3
                            mapPara.errorLow = 0.3
                            mapPara.relaxSleep = 500
                            time.sleep(2)
                            MapCnt = 1
                            win_max, win_min = 4.0, 0.4
                            for m in range(MapCnt):
                                # sleep 1 min
                                print('\tRandom count = ' + str(m))
                                ITargets = np.tile(np.array(init_states),1366)
                                ITargets = ITargets[:4096].reshape((128, 32))

                                targetList = ((ITargets * 1000).astype('int') / 1000).flatten().tolist()
                                time_start = time.time()
                                self.array.myapp.cmdMapArray(mapPara, addr, targetList)
                                time_stamp = time.time()
                                fly_time = time_stamp - time_start
                                print('\tFly_time =\t' + str(fly_time))
                                time.sleep(2)
                                right, rddata = self.array.myapp.readAll(chipNum + 1)
                                IAfterMap = np.array(rddata).reshape(32, 128).transpose().round(decimals=3)

                                error = abs(IAfterMap - ITargets )

                                SucessRate = sum(sum(error < 0.3)) / ARRAY4K.SIZE
                                print('\tRandomized success rate: ' + str(SucessRate))

            elif is_test:
                print('Platform test mode: use virtual hardware platform, return virtual data as read data')
        else:
            self.batch_mode = True
            print('Platform batch load mode: use saved read data')
            print('Read data path: ' + str(batch_mode_path))
            self.no_sample_file = [0]
            self.no_sample = 0
            linecache.clearcache()
            for i, read_path in enumerate(batch_mode_path):
                # read first line = 1
                data_note = linecache.getline(read_path, 1).split(',')
                # data_note = [no_sample, data_len, s, time_stamp, fly_time]
                if i == 0:
                    # add 1 account for one data_note line
                    self.data_block_len = int(data_note[1]) + 1
                    print('Sample data block len: ' + data_note[1])
                    # the time of start sampling
                    self.start_sample_time = float(data_note[3])
                    date = time.asctime(time.localtime(self.start_sample_time))
                    print('Time of start sampling: ' + date)
                self.no_sample += int(data_note[0])
                self.no_sample_file.append(self.no_sample)
                print('Total sample times: ' + data_note[0] + ' in ' + read_path)

    # Calling destructor
    def __del__(self):
        if not self.is_test and not self.batch_mode:
            if self.connect_state == True:
                self.sdk.disconnect_device()
                print('disconnect device')

    def deploy_XBArray(self, layer_id, input_dim, output_dim, N):
        self.layer['id'] = layer_id
        self.layer['size'] = [input_dim, output_dim, N]
        self.layer['total_cells'] = input_dim * output_dim * N
        self.layer['addr_start'] = self.addr_pointer
        self.layer['order_number_start'] = self.order_number # first cell of this layer
        self.layer['order_number_end'] = self.order_number + self.layer['total_cells']

        self.layers.append(copy.deepcopy(self.layer))
        self.addr_pointer = copy.deepcopy(self.layer['addr_end'])
        self.order_number += self.layer['total_cells']
        self.layers_ID += 1
        print('XBArray info:')
        print(self.layer)
        # deploy succ
        return self.layer['id']

    def sample_layer(self, layer_id, no_sample=1):
        if layer_id == None:
            print('Layer id error!')
            return None

        if layer_id==1:
            self.all_sample_data = self.sample_all_arrays(no_sample)

        layer = [layer for layer in self.layers if layer['id'] == layer_id][0]
        layer_sample = self.all_sample_data[:,layer['order_number_start']:layer['order_number_end']]
        layer_sample = layer_sample.reshape([no_sample]+layer['size'])

        return layer_sample
    def sample_all_arrays(self, no_sample):
        currents_data = np.zeros([no_sample, ARRAY4K.SIZE*self.used_array])
        if not self.is_test:
            for s in range(no_sample):
                for array_id in range(self.used_array):
                    right, rddata = self.array.myapp.readAll(array_id + 1)
                    rddata = np.array(rddata).reshape(32,128).transpose().round(decimals=3)

                    if not right:
                        return
                    currents_data[s,ARRAY4K.SIZE*array_id:ARRAY4K.SIZE*(array_id+1)] = rddata.flatten()
        else:
            for s in range(no_sample):
                for array_id in range(self.used_array):
                    currents_data[s, ARRAY4K.SIZE * array_id:ARRAY4K.SIZE * (array_id + 1)] = ARRAY4K.DATA.flatten()

        return currents_data

    def update_layer(self, op_dir, layer_id, op_mask):
        if layer_id == None:
            print('Layer id error!')
            return None

        layer = [layer for layer in self.layers if layer['id'] == layer_id][0]
        # POSITIVE = 1
        # NEGATIVE = -1
        # NULLOP = 0
        if np.sum(op_mask.flatten()) != 0.:
            self.op_arrays[layer['order_number_start']:layer['order_number_end']] += op_mask.flatten()*op_dir
        if layer_id == self.layers_ID and op_dir==NEGATIVE:
            print(np.sum(abs(self.op_arrays) == 1))
            self.program_all_arrays(self.op_arrays)
            self.op_arrays = np.zeros(ARRAY4K.SIZE * self.used_array)

    def program_all_arrays(self, op_arrays):
        op_arrays = op_arrays.reshape([self.used_array,ARRAY4K.SIZE])
        addrInfo = apps.AddrInfo()
        addrInfo.BLCnt = 1
        addrInfo.WLCnt = 1

        if not self.is_test:
            for array_id, op_array in enumerate(op_arrays):
                addrInfo.chipNum = array_id+1
                op_array = op_array.reshape(ARRAY4K.SHAPE)

                right, rddata1 = self.array.myapp.readAll(addrInfo.chipNum)
                pre_test = np.array(rddata1).reshape(32,128).transpose() * 1000

                for cell_addr, op in np.ndenumerate(op_array):
                    if op == POSITIVE:
                        addrInfo.BLStart = cell_addr[0]
                        addrInfo.WLStart = cell_addr[1]
                        self.array.myapp.setOperation(addrInfo,VBL=2.0, VWL=1.2)
                    elif op == NEGATIVE:
                        addrInfo.BLStart = cell_addr[0]
                        addrInfo.SLStart=addrInfo.WLStart = cell_addr[1]
                        # ret = self.array.myapp.resetOperation(addrInfo, VSL=2.0, VWL=2.9, pulseWidth=100, pulseGap=10, pulseCnt=1) # crosstalk
                        self.myResetOp(addrInfo, VSL=1.95, VWL=3.2)

                right, rddata2 = self.array.myapp.readAll(addrInfo.chipNum)
                aft_test = np.array(rddata2).reshape(32,128).transpose() * 1000
                diff = aft_test - pre_test

        else:
            for array_id, op_array in enumerate(op_arrays):
                op_array = op_array.reshape(ARRAY4K.SHAPE)
                for cell_addr, op in np.ndenumerate(op_array):
                    if op== NULLOP:
                        continue
                    if op == POSITIVE:
                        ARRAY4K.DATA[cell_addr] += 1
                    elif op == NEGATIVE:
                        ARRAY4K.DATA[cell_addr] -= 1
        pass


    def myResetOp(self, addrInfo, VSL=2.0, VWL=2.9):
        finderCfg = apps.MapFinderCfg()

        RstVSLStart = VSL
        RstVSLStep = 0.05
        RstVSLEnd = 4.
        RstVWLStart = VWL
        RstVWLStep = 0.05
        RstVWLEnd = 5.0

        finderCfg.rstVWLStart   = RstVWLStart
        finderCfg.rstVWLStep    = RstVWLStep
        finderCfg.rstVWLEnd     = RstVWLEnd
        finderCfg.rstVSLStart   = RstVSLStart
        finderCfg.rstVSLStep    = RstVSLStep
        finderCfg.rstVSLEnd     = RstVSLEnd
        finderCfg.errorLimit = 4.
        finderCfg.nMax = 1
        finderCfg.voltageIncMode = 1
        finderCfg.readDirection = arrayexp.NEGATIVE_READ

        targetList = [0.4]
        self.array.myapp.resetTarget(finderCfg, addrInfo, targetList)
