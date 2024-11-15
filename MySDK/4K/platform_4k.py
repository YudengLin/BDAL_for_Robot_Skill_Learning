import linecache
import os
import sys
sys.path.append(os.path.dirname(__file__))
import time
from sdk import SDK
# from app import arrayexp
import copy
import numpy as np
import csv
# ARRAY info

class ARRAYs_4K():
    def __init__(self, no_array=1):
        self.CHIP_NUM = no_array
        self.ID = [chip for chip in range(no_array)]
        self.ROW = 128
        self.COL= 32
        self.SIZE = self.ROW * self.COL
        self.TOTAL_SIZE = self.SIZE * no_array
        self.ACC_SIZE = [i * self.SIZE for i in range(no_array)] + [self.TOTAL_SIZE]
        self.START_ADDR = [[0, 0] for i in range(no_array)]
        self.END_ADDR = [[self.ROW-1, self.COL-1] for i in range(no_array)]
        # Test data
        self.DATA = np.reshape(range(self.SIZE), (self.ROW, self.COL))+1

    def _order2addr(self, order):
        row, col = divmod(order, self.COL)
        return [row,col]

    def _addr2order(self, addr):
        row, col = addr[0], addr[1]
        order = row * self.COL + col
        return order

    def get_deployed_array(self, order_number_start, order_number_end):
        arrayid_idx_start, arrayid_idx_end = None, None
        array_addr_start, array_addr_end = None, None
        for i, size in enumerate(self.ACC_SIZE):
            if order_number_start < self.ACC_SIZE[i + 1]:
                arrayid_idx_start = i
                array_order_start = order_number_start - size
                array_addr_start = self._order2addr(array_order_start)
                break
        for i, size in enumerate(self.ACC_SIZE):
            if order_number_end < self.ACC_SIZE[i + 1]:
                arrayid_idx_end = i
                array_order_end = order_number_end - size
                array_addr_end = self._order2addr(array_order_end)
                break

        if arrayid_idx_start is None or arrayid_idx_end is None :
            # deploy fail
            return arrayid_idx_start, arrayid_idx_end, False
        else:
            # arrays_addr = [[array_addr_start, self.END_ADDR],
            #                [self.START_ADDR,array_addr_end]]
            # arrays_id = [self.ID[arrayid_idx_start],
            #              self.ID[arrayid_idx_end]]
            # if arrayid_idx_end > arrayid_idx_start:
            #     # this XBArray_ID is deployed on different arrays
            #     arrayid_idx_end = None if arrayid_idx_end == self.CHIP_NUM - 1 else arrayid_idx_end + 1
            #     arrays_id = self.ID[arrayid_idx_start:arrayid_idx_end]
            #     for i in range(len(arrays_id) - 2):
            #         arrays_addr.insert([self.START_ADDR, self.END_ADDR], 1)

            arrays_addr = [array_addr_start, array_addr_end]
            arrayid_idx_end = None if arrayid_idx_end == self.CHIP_NUM - 1 else arrayid_idx_end + 1
            arrayids_idx = [arrayid_idx_start, arrayid_idx_end]

            return arrayids_idx, arrays_addr, True

class Platform_4K():
    def __init__(self, Exp=None):
        self.Exp = Exp
        self.is_test = True
        self.connect_state = False
        self.ARRAYs = ARRAYs_4K()

        self.order_number = 0  # assigned cells' count: order number / serial number
        self.addr_pointer = [0, 0]  # [Row, Col]
        # XBArrays
        self.XBArrays = []
        self.XBArray_ID = 0
        # XBArray_ID info
        self.XBArray = {'id': [], # provide ID to upper function
                        'arrayids_idx':[], # record physical the index of array ID
                        'size': [],
                        'total_cells': 0,
                        'order_number_start': 0,
                        'addr_start': None,  # [Row, Col]
                        'addr_end': None,  # [Row, Col]
                        'SDK_ARRAY': [None, None],  # addr_start_deployed, addr_end_deployed
                        'sample_count':0, # batch load mode
                        }

    def platform_init(self, batch_mode_path=None, is_test=False):
        self.is_test = is_test
        self.saved_read_data_path = batch_mode_path
        if self.saved_read_data_path is None:
            self.batch_mode = False
            if not is_test:
                if not self.connect_state:
                    print('platform initial ...')
                    self.sdk = SDK()
                    self.connect_state = self.sdk.connect_device()
                    if self.connect_state == False:
                        print('connect device fail')
                        exit()
                    self.sdk.register_call_back(self._call_back)
                    print('connect device succ', flush=True)
            elif is_test:
                print('Platform test mode: use virtual hardware platform, return cell index as read data')
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
            # # first line = 1
            # data_note = linecache.getline(batch_mode_path, 1).split(',')
            # # data_note = [no_sample, data_len, s, time_stamp, fly_time]
            # self.total_sample = int(data_note[0])
            # # add 1 for one data_note line
            # self.data_block_len = int(data_note[1]) + 1

    # Calling destructor
    def __del__(self):
        if not self.is_test and not self.batch_mode:
            if self.connect_state == True:
                self.sdk.disconnect_device()
                print('disconnect device')

    def deploy_XBArray(self, input_dim, output_dim, N):
        self.XBArray['id'] = self.XBArray_ID
        self.XBArray['size'] = [input_dim, output_dim, N]
        self.XBArray['total_cells'] = input_dim * output_dim * N
        self.XBArray['addr_start'] = self.addr_pointer
        self.XBArray['order_number_start'] = self.order_number # first cell of this layer

        # check if overflow cells
        order_number_end = self.order_number + self.XBArray['total_cells']

        arrayids_idx, arrays_addr, is_succ = self.ARRAYs.get_deployed_array(self.order_number, order_number_end)
        if not is_succ:
            print('Deploy XBArray_ID error: Overflow cells: ' + str(order_number_end))
            print('XBArrays info:')
            print(self.XBArrays)
            # Reset deploy info
            self.order_number = 0  # assigned cells' count: order number / serial number
            self.addr_pointer = [0, 0]  # [Row, Col]
            self.XBArrays = []
            # deploy false
            return None
        else:
            self.XBArray['arrayids_idx'] = arrayids_idx
            # self.XBArray_ID['SDK_ARRAY'] = [ARRAYs_4K, ARRAYs_4K]
            self.XBArray['addr_end'] = arrays_addr[1]

        self.XBArrays.append(copy.deepcopy(self.XBArray))
        self.addr_pointer = copy.deepcopy(self.XBArray['addr_end'])
        self.order_number = order_number_end
        self.XBArray_ID += 1
        print('XBArray_ID info:')
        print(self.XBArray)
        # deploy succ
        return self.XBArray

    def sample_XBArray(self, XBArray_ID, no_sample=1):
        if XBArray_ID == None:
            print('Layer id error!')
            return None
        XBArray = [XBArray for XBArray in self.XBArrays if XBArray['id'] == XBArray_ID][0]
        currents_sample = np.zeros([no_sample]+XBArray['size'])
        time_start = time.time()
        for s in range(no_sample):
            if XBArray['SDK_ARRAY'][0] == XBArray['SDK_ARRAY'][1]:
                # this XBArray_ID is deployed on SDK.ARRAYs_4K or SDK.ARRAY2
                currents = self._read_block(XBArray['SDK_ARRAY'][0], XBArray['addr_start'], XBArray['addr_end'],XBArray['sample_count'])
            else:
                # this XBArray_ID is deployed across SDK.ARRAYs_4K and SDK.ARRAY2
                currents_array1 = self._read_block(XBArray['SDK_ARRAY'][0], XBArray['addr_start'],
                                                   XBArray['SDK_ARRAY'][0].END_ADDR,XBArray['sample_count'])
                currents_array2 = self._read_block(XBArray['SDK_ARRAY'][1], XBArray['SDK_ARRAY'][1].START_ADDR,
                                                   XBArray['addr_end'],XBArray['sample_count'])
                currents = np.concatenate((currents_array1,currents_array2),axis=0)

            currents = currents.reshape(XBArray['size'])
            currents_sample[s,:,:,:] = currents
            XBArray['sample_count'] += 1
        currents_sample = currents_sample/1000. if not self.is_test else currents_sample
        time_end = time.time()
        # print('totally cost: ', time_end - time_start)
        return currents_sample

    def _read_block(self, ARRAY, addr_start, addr_end, sample_cnt):
        row_start, col_start = addr_start[0], addr_start[1]
        row_end, col_end= addr_end[0], addr_end[1]
        data_start = col_start
        data_end = -(ARRAY.COL - col_end)
        if col_end == 0 :
            row_end = row_end - 1
            data_end = None
        elif addr_end==ARRAY.END_ADDR:
            data_end = None
        currents, _ = self._sdk_read_array(ARRAY, row_start, row_end, sample_cnt)
        currents = currents[0,data_start:data_end]
        return currents

    def _sdk_read_array(self, ARRAY, row_start, row_end, sample_cnt=None):
        if self.batch_mode:
            currents_list = self.get_saved_data(row_start, row_end, sample_cnt)
            currents = np.array(currents_list).reshape((1, -1))
        else:
            if not self.is_test:
                currents_list = self.sdk.read_array(ARRAY.ID, row_start + 1, row_end + 1)
            else:
                currents_list = list(ARRAY.DATA)
            row_end = None if row_end==ARRAY.ROW-1 else row_end+1
            currents_list = currents_list[row_start:row_end]
            currents = np.array(currents_list).reshape((1,-1))

        return currents, currents_list

    def save_read_data(self,save_path, ARRAY, row_start, row_end, no_sample):
        self.saved_read_data_path = save_path
        time_span = 0 # secs
        with open(save_path, 'a+', newline='') as csvfile:
            csv_write = csv.writer(csvfile)
            time_start = time.time()
            for s in range(no_sample):
                _, currents_list = self._sdk_read_array(ARRAY, row_start, row_end)
                time_stamp = time.time()
                fly_time = time_stamp - time_start
                data_len = len(currents_list)
                data_note = [no_sample, data_len, s, time_stamp, fly_time]
                csv_write.writerow(data_note)
                csv_write.writerows(currents_list)
                remain_time = (no_sample-s)*(fly_time/(s+1))
                print('Estimate Stop Time: '+time.asctime(time.localtime( time.time()+remain_time)))
                time.sleep(time_span)
        print('Saved read data path:'+save_path)

    def get_saved_data(self,row_start, row_end, sample_cnt):
        read_path = None
        currents_list = []

        use_section_sample = 360
        total_sample = self.no_sample if not use_section_sample else use_section_sample
        cycle, sample_cnt = divmod(sample_cnt, total_sample)

        offset_sample = self.Exp['offset_sample']
        sample_cnt = (sample_cnt + offset_sample)

        for i,no_sample_file in enumerate(self.no_sample_file):
            if sample_cnt < self.no_sample_file[i+1]:
                read_path = self.saved_read_data_path[i]
                sample_cnt = sample_cnt - no_sample_file
                break

        start_row = sample_cnt * self.data_block_len + 2 + row_start
        total_rows = row_end - row_start + 1

        for index in range(total_rows):
            data = linecache.getline(read_path, start_row + index).split(',')
            current_list = list(map(int, data))
            currents_list.append(current_list)

        return currents_list


    def _print_process(self, i, total):
        percent = int(i * 100 / total)
        charnum = int(percent / 2)
        if charnum > 30:
            charnum = 30
        s = '\r%s[%d/%d]' % (charnum * '#', i, total)
        print(s, end='', flush=True)
        # if i == total:
        #    print('')

    def _call_back(self, mode, value, valuemax):
        # if mode == 'read':
        self._print_process(value, valuemax)

def _write_string(path, data_list, remove=False):
    data = ''
    for l in data_list:
        for v in l:
            data += '%d,' % v
        data += '\n'

    if remove == True and os.path.exists(path) == True:
        os.remove(path)
        # return

    if isinstance(data, str):
        file = open(path, 'a')
    elif isinstance(data, bytes):
        file = open(path, 'ab')
    else:
        return

    file.write(data)
    file.close()


# Function test
if  __name__ == '__main__':
    BNN_Chip = Platform_4K()
    BNN_Chip.platform_init(is_test=False)


    save_path = './read_data_200nA_U37_20kSample.csv'
    ARRAY, row_start, row_end, no_sample = ARRAYs_4K, 0, 163, 20000
    BNN_Chip.save_read_data(save_path, ARRAY, row_start, row_end, no_sample)

    BNN_Chip = Platform_4K()
    BNN_Chip.platform_init(batch_mode_path='./read_data_test.csv')


    input_dim, output_dim, N = 6, 100, 3
    XBArray1_id = BNN_Chip.deploy_XBArray(input_dim, output_dim, N)
    # read_Ndevice: [no_sample, input_dim, output_dim, N]
    read_Ndevice = BNN_Chip.sample_XBArray(XBArray1_id, no_sample=1)
    raw_weight_sample = np.sum(read_Ndevice, axis=3)
    print(read_Ndevice.shape)

    input_dim, output_dim, N = 101, 100 , 3
    XBArray2_id = BNN_Chip.deploy_XBArray(input_dim, output_dim, N)
    # read_Ndevice: [no_sample, input_dim, output_dim, N]
    read_Ndevice = BNN_Chip.sample_XBArray(XBArray2_id, no_sample=1)
    raw_weight_sample = np.sum(read_Ndevice, axis=3)
    print(read_Ndevice.shape)

    input_dim, output_dim, N = 101, 2 , 3
    XBArray3_id = BNN_Chip.deploy_XBArray(input_dim, output_dim, N)
    # read_Ndevice: [no_sample, input_dim, output_dim, N]
    read_Ndevice = BNN_Chip.sample_XBArray(XBArray3_id, no_sample=1)
    raw_weight_sample = np.sum(read_Ndevice, axis=3)
    print(read_Ndevice.shape)