import numpy as np
from enum import Enum

from client_tcp import ClientTCP
import reram_register
import globals.global_func as gf

MAX_PACKET_LEN = 0xFFFF

CMD_HEAD = 0x3A
RESPONSE_HEAD = 0x3B
PACKET_HEAD_LEN = 0x07

CMD_NULL = 0xFF
CMD_COMMON_START = 0x80
CMD_DEBUG = 0x01 + CMD_COMMON_START
CMD_WRITE_REG = 0x02 + CMD_COMMON_START
CMD_READ_REG = 0x03 + CMD_COMMON_START
CMD_SEQ_WRITE_REG = 0x04 + CMD_COMMON_START
CMD_SEQ_READ_REG = 0x05 + CMD_COMMON_START

CMD_CFG_REG = 0x01
CMD_MAP_FLOW = 0x02
CMD_MAP_PROGRESS = 0x03
CMD_MAP_ABORT = 0x04
CMD_MAP_RESULT = 0x05
CMD_SEQ_READ = 0x06
CMD_SET_TARGET = 0x07
CMD_RESET_TARGET = 0x08
CMD_APPLY_OP = 0x09
CMD_OP_RESULT = 0x0A
CMD_MAP_READ = 0x0B
CMD_PULSE_GENERATOR = 0x0C
CMD_MAP_ARRAY = 0x0D
CMD_CELL_PARA = 0x0E
CMD_READ_ALL = 0x0F

ERROR_FLAG = 0x8F
NO_ERROR = 0x00
PAYLOAD_LEN_ERROR = 0x01
CMD_NUM_ERROR = 0x02
PARA1_ERROR = 0x03
PARA2_ERROR = 0x04
HEAD_CHECK_ERROR = 0x05
PAYLOAD_CHECK_ERROR = 0x06

POSITIVE_READ = 0
NEGATIVE_READ = 1

FIRST_INC_BSLV = 0
FIRST_INC_WLV = 1

TIA_SL_Table = (48, 32, 16, 0, 6, 38, 22, 21, 23, 51, 39, 36, 52, 17, 3, 55,
                50, 20, 19, 35, 5, 34, 1, 4, 54, 2, 18, 37, 53, 33, 49, 7)

TIA_BL_Table = (56, 10, 40, 23, 25, 44, 54, 12, 6, 51, 58, 32, 9, 16, 38, 0,
                22, 15, 42, 31, 24, 14, 11, 47, 8, 48, 21, 33, 7, 17, 53, 1,
                39, 30, 55, 63, 57, 61, 60, 46, 26, 50, 20, 62, 5, 18, 52, 34,
                37, 13, 41, 2, 59, 45, 36, 29, 27, 49, 28, 35, 4, 19, 43, 3,
                40, 13, 10, 31, 56, 16, 9, 0, 7, 48, 22, 1, 24, 47, 8, 63, 57,
                15, 42, 62, 39, 17, 41, 14, 6, 32, 21, 46, 43, 49, 23, 29, 25,
                50, 37, 34, 38, 18, 11, 2, 27, 30, 36, 51, 20, 45, 52, 61, 26,
                33, 4, 19, 58, 28, 55, 35, 59, 60, 53, 12, 5, 44, 54, 3)

# TIA_SL_Table = (48, 32, 16, 0, 15, 31, 47, 63, 49, 33, 17, 1, 14, 30, 46, 62,
#                 50, 34, 18, 2, 13, 29, 45, 61, 51, 35, 19, 3, 55, 39, 23, 7)

# TIA_BL_Table = (57, 12, 41, 28, 25, 44, 9, 60, 6, 48, 22, 32, 38, 16, 54, 0,
#                 58, 15, 42, 31, 26, 47, 10, 63, 5, 49, 21, 33, 37, 17, 53, 1,
#                 59, 14, 43, 30, 27, 46, 11, 62, 4, 50, 20, 34, 36, 18, 52, 2,
#                 56, 13, 40, 29, 24, 45, 8, 61, 7, 51, 23, 35, 39, 19, 55, 3,
#                 57, 48, 41, 32, 25, 16, 9, 0, 6, 15, 22, 31, 38, 47, 54, 63,
#                 58, 49, 42, 33, 26, 17, 10, 1, 5, 14, 21, 30, 37, 46, 53, 62,
#                 59, 50, 43, 34, 27, 18, 11, 2, 4, 13, 20, 29, 36, 45, 52, 61,
#                 56, 51, 40, 35, 24, 19, 8, 3, 7, 60, 23, 44, 39, 28, 55, 12)

TIA_FEEDBACK = [2400000, 2500, 24900, 249000]  # FIXME 24.9 or 2400000

BASE_ADDR = 0x43C00000
REG160_ADDR = (BASE_ADDR + 160 * 4)

READ_DATA_SIZE_INT = 32
READ_DATA_SIZE_BYTE = (READ_DATA_SIZE_INT * 4)

PROGRESS_FINISH_FLAG = 0xFFFF


def makeInputOrV(active, inactive, start, count, len):
    voltage = []
    for i in range(len):
        if start <= i < start + count:
            voltage.append(active)
        else:
            voltage.append(inactive)
    return voltage


class AddrInfo():
    def __init__(self,
                 chipNum=1,
                 BLStart=0,
                 BLCnt=1,
                 SLStart=0,
                 SLCnt=1,
                 WLStart=0,
                 WLCnt=1):
        self.chipNum = chipNum
        self.BLStart = BLStart
        self.BLCnt = BLCnt
        self.SLStart = SLStart
        self.SLCnt = SLCnt
        self.WLStart = WLStart
        self.WLCnt = WLCnt


class MapFinderCfg():
    def __init__(self):
        self.formVWLStart = 0  # unit(V)
        self.formVWLStep = 0
        self.formVWLEnd = 0
        self.formVBLStart = 0
        self.formVBLStep = 0
        self.formVBLEnd = 0
        self.setVWLStart = 1.1
        self.setVWLStep = 0.1
        self.setVWLEnd = 3.0
        self.setVBLStart = 3.0
        self.setVBLStep = 0.1
        self.setVBLEnd = 5.0
        self.rstVWLStart = 2.6
        self.rstVWLStep = 0.1
        self.rstVWLEnd = 5.
        self.rstVSLStart = 3.5
        self.rstVSLStep = 0.3
        self.rstVSLEnd = 4.4
        self.errorLimit = 0  # unit(uA)
        self.nMax = 0
        self.readDirection = 0
        self.voltageIncMode = 0


class SetFinderCfg():
    def __init__(self):
        self.setVWLStart = 1.1
        self.setVWLStep = 0.1
        self.setVWLEnd = 3.0
        self.setVBLStart = 3.0
        self.setVBLStep = 0.1
        self.setVBLEnd = 5.0
        self.nMax = 30
        self.readDirection = NEGATIVE_READ
        self.voltageIncMode = FIRST_INC_BSLV  # FIRST_INC_BSLV, FIRST_INC_WLV


class ResetFinderCfg():
    def __init__(self):
        self.rstVWLStart = 2.6
        self.rstVWLStep = 0.1
        self.rstVWLEnd = 5.
        self.rstVSLStart = 3.5
        self.rstVSLStep = 0.3
        self.rstVSLEnd = 4.4
        self.nMax = 30
        self.readDirection = NEGATIVE_READ
        self.voltageIncMode = FIRST_INC_BSLV  # FIRST_INC_BSLV, FIRST_INC_WLV


class ReadCfg():
    def __init__(self):
        self.chipNum = 1
        self.BLActive = 0
        self.SLActive = 0
        self.readDirection = NEGATIVE_READ
        self.readV = 0.2
        self.accessV = 5
        self.TIAFeedback = reram_register.R_250K


class Channel(Enum):
    BL = 0
    WL = 1
    SL = 2


class MapPara():
    def __init__(self):
        # self.setVWLStart = 1.1
        # self.setVWLStep = 0.05
        # self.setVWLEnd = 3.0
        # self.setVBLStart = 1.8
        # self.setVBLStep = 0.05
        # self.setVBLEnd = 3
        # self.rstVWLStart = 3.2
        # self.rstVWLStep = 0.05
        # self.rstVWLEnd = 5.0
        # self.rstVSLStart = 1.9
        # self.rstVSLStep = 0.05
        # self.rstVSLEnd = 4.0
        # self.errorHigh = 0.3  # unit(uA)
        # self.errorLow = 0.3
        # self.maxProgramNum = 100
        # self.maxCheckNum = 7
        # self.checkThreshold = 5
        # self.relaxSleep = 1
        self.setVBLStart = 1.5
        self.setVBLStep = 0.1
        self.setVBLEnd = 3.
        self.setVWLStart = 1.
        self.setVWLStep = 0.1
        self.setVWLEnd = 3.

        self.rstVSLStart = 1.8  # 1.9
        self.rstVSLStep = 0.1
        self.rstVSLEnd = 4.0
        self.rstVWLStart = 3.2
        self.rstVWLStep = 0.1
        self.rstVWLEnd = 5.0

        self.errorHigh = 0.3  # unit(uA)
        self.errorLow = 0.3
        self.maxProgramNum = 300
        self.maxCheckNum = 20
        self.checkThreshold = 10
        self.relaxSleep = 10000  #unit(us)
        self.setPulseWidth = 20  # 单位ns，20ns的整数倍
        self.rstPulseWidth = 20
class App():
    def __init__(self, **kwargs):
        self.client = ClientTCP()
        self.bBusy = False
        self._regCfg = reram_register.ReramReg()
        self.configParas(**kwargs)
        self.bCheckPacket = True
        # self.tick = 0

    def checkDevice(self):
        """Check that the device is idle.

        Args:
            None.

        Returns:
            result: bool
                Return True, if the device is idle.
        """
        if not self.client.isConnected():
            print('Please connect device')
            return False

        if self.bBusy:
            print('Device is busy, and please try again later')
            return False
        return True

    def checkHead(self, head):
        ret = 0
        if self.bCheckPacket:
            ret = gf.calcXor(head)
        return ret

    def checkPayload(self, payload):
        ret = bytes([0, 0])
        if self.bCheckPacket:
            ret = gf.crc16(payload)
            ret = gf.uint16ToBytes(ret, 'little')
        return ret

    def sendPacket(self, cmdNum, para1, para2, payload):
        payload = bytes(payload)

        # self.tick += len(payload)
        # print(self.tick)
        # if self.tick > 50000:
        #     self.client.netRefresh()
        #     self.tick = 0

        if len(payload) != 0:
            payload = payload + self.checkPayload(payload)
        # print('send:')
        payloadLen = len(payload)
        packetHead = np.zeros(PACKET_HEAD_LEN, np.uint8)
        packetHead[0] = CMD_HEAD
        packetHead[1] = cmdNum
        packetHead[2] = para1
        packetHead[3] = para2
        # print(payloadLen)
        # time.sleep(0.01)
        if payloadLen != 0:
            packetHead[4] = payloadLen & 0xFF
            packetHead[5] = payloadLen >> 8
        packetHead[6] = self.checkHead(packetHead[:6])
        packet = bytes(packetHead) + payload
        self.client.send(packet, bLog=False)

    def cmdDebug(self, debugOn):
        cmdNum = CMD_DEBUG
        if debugOn:
            para1 = 0
        else:
            para1 = 1

        para2 = 0
        payload = []
        self.sendPacket(cmdNum, para1, para2, payload)
        return self.disposeReport()

    def cmdWriteReg(self, addr, data):
        cmdNum = CMD_WRITE_REG
        para1 = 0
        para2 = 0
        payloadArr = np.zeros(2 * len(addr), dtype='<u4')
        count = 0
        for i, j in zip(addr, data):
            payloadArr[count] = i
            payloadArr[count + 1] = j
            count = count + 2
        payload = bytes(payloadArr)
        self.sendPacket(cmdNum, para1, para2, payload)
        return self.disposeReport()

    def cmdReadReg(self, addr):
        cmdNum = CMD_READ_REG
        para1 = 0
        para2 = 0
        payloadArr = np.array(addr, dtype='<u4')
        payload = bytes(payloadArr)
        self.sendPacket(cmdNum, para1, para2, payload)
        return self.disposeReport()

    def cmdSeqWriteReg(self, startAddr, data):
        cmdNum = CMD_SEQ_WRITE_REG
        para1 = 0
        para2 = 0
        payloadArr = np.zeros(len(data) + 1, dtype='<u4')
        payloadArr[0] = startAddr

        for i, j in enumerate(data):
            payloadArr[i + 1] = j
        payload = bytes(payloadArr)
        self.sendPacket(cmdNum, para1, para2, payload)
        return self.disposeReport()

    def cmdSeqReadReg(self, startAddr, count):
        cmdNum = CMD_SEQ_READ_REG
        para1 = count & 0xFF
        para2 = (count >> 8) & 0xFF
        payload = gf.uint32ToBytes(startAddr, 'little')
        self.sendPacket(cmdNum, para1, para2, payload)
        return self.disposeReport()

    def disposeReport(self):
        data = self.client.recv(1, bLog=False)
        if data[0] != RESPONSE_HEAD:
            print('Response Head Error')
            return (False, None)

        data = self.client.recv(PACKET_HEAD_LEN - 1, bLog=False)
        if data[0] == CMD_NULL:
            print('Command Number Error')
            return (False, None)

        if data[1] != 0 or data[2] != 0:
            print('Return error code: 0x%02X%02X' % (data[1], data[2]), end='')
            if data[2] == PAYLOAD_LEN_ERROR:
                print('(PAYLOAD LENGTH ERROR)')
            elif data[2] == CMD_NUM_ERROR:
                print('(COMMAND NUMBER ERROR)')
            elif data[2] == PARA1_ERROR:
                print('(PARA1 ERROR)')
            elif data[2] == PARA2_ERROR:
                print('(PARA2 ERROR)')
            elif data[2] == HEAD_CHECK_ERROR:
                print('(HEAD CHECK ERROR)')
            elif data[2] == PAYLOAD_CHECK_ERROR:
                print('(PAYLOAD CHECK ERROR)')
            else:
                print('(UNDEFINED ERROR)')
            return (False, None)

        if data[5] != self.checkHead(bytes([RESPONSE_HEAD]) + data[:5]):
            print('Check Head Error')
            return (False, None)

        if data[3] != 0 or data[4] != 0:
            payloadLen = (data[4] << 8) | data[3]
            # print('recv:')
            # print(payloadLen)
            payload = self.client.recv(payloadLen, bLog=False)
            if payload[-2:] == self.checkPayload(payload[:-2]):
                return (True, payload[:-2])
            else:
                print('Check Payload Error')
                return (False, None)
        else:
            # if data[0]==CMD_OP_RESULT:
            #     print("OP Result NONE ERROR")
            return (True, None)

    def readReg(self, addr):
        ret = self.cmdReadReg(addr)
        if ret[0] is True:
            npData = np.frombuffer(ret[1], dtype='<u4')
            data = list(map(int, npData))
            return True, data
        else:
            return False, None

    def seqReadReg(self, startAddr, count):
        ret = self.cmdSeqReadReg(startAddr, count)
        if ret[0] is True:
            npData = np.frombuffer(ret[1], dtype='<u4')
            data = list(map(int, npData))
            return True, data
        else:
            return False, None

    def cmdCfgReg(self):
        cmdNum = CMD_CFG_REG
        para1 = 0
        para2 = 0
        payload = bytes(self._regCfg.getRegTable())
        self.sendPacket(cmdNum, para1, para2, payload)
        return self.disposeReport()

    def configParas(self, **kwargs):
        # 芯片选择，取值范围[1,8]
        self.chipNum = kwargs.get('chipNum')
        # 操作模式，取值范围[0x0A,0x0D]
        self.mode = kwargs.get('mode')
        # 反馈电阻,取值范围[0,3]
        self.TIAFeedback = kwargs.get('TIAFeedback')
        # BL起始,取值范围[0,127]
        self.BLStart = kwargs.get('BLStart')
        # BL数量,取值范围[1,128]
        self.BLCnt = kwargs.get('BLCnt')
        # SL起始,取值范围[0,31]
        self.SLStart = kwargs.get('SLStart')
        # SL数量,取值范围[1,32]
        self.SLCnt = kwargs.get('SLCnt')
        # WL起始,取值范围[0,31]
        self.WLStart = kwargs.get('WLStart')
        # WL数量,取值范围[1,32]
        self.WLCnt = kwargs.get('WLCnt')
        # 选中的cell BL的输入选择GND_SWITCH, DAC_SWITCH, FLOAT_SWITCH, TIA_SWITCH
        self.BLInputActive = kwargs.get('BLInputActive')
        # 选中的cell SL的输入选择GND_SWITCH, DAC_SWITCH, FLOAT_SWITCH, TIA_SWITCH
        self.SLInputActive = kwargs.get('SLInputActive')
        # 选中的cell WL的输入选择GND_SWITCH, DAC_SWITCH, FLOAT_SWITCH, TIA_SWITCH
        self.WLInputActive = kwargs.get('WLInputActive')
        # 选中cell BL电压
        self.BLVActive = kwargs.get('BLVActive')
        # 选中cell SL电压
        self.SLVActive = kwargs.get('SLVActive')
        # 选中cell WL电压
        self.WLVActive = kwargs.get('WLVActive')
        # 未选中的cell BL的输入选择GND_SWITCH, DAC_SWITCH, FLOAT_SWITCH, TIA_SWITCH
        self.BLInputInactive = kwargs.get('BLInputInactive')
        # 未选中的cell SL的输入选择GND_SWITCH, DAC_SWITCH, FLOAT_SWITCH, TIA_SWITCH
        self.SLInputInactive = kwargs.get('SLInputInactive')
        # 未选中的cell WL的输入选择GND_SWITCH, DAC_SWITCH, FLOAT_SWITCH, TIA_SWITCH
        self.WLInputInactive = kwargs.get('WLInputInactive')
        # 未选中cell BL电压
        self.BLVInactive = kwargs.get('BLVInactive')
        # 未选中cell SL电压
        self.SLVInactive = kwargs.get('SLVInactive')
        # 未选中cell WL电压
        self.WLVInactive = kwargs.get('WLVInactive')
        # TIA板的电压
        self.TIAVoltage = kwargs.get('TIAVoltage')
        # 脉冲宽度，单位ns，必须是5ns整数倍
        self.pulseWidth = kwargs.get('pulseWidth')
        # 间隔时间，单位ns，必须是5ns整数倍
        self.pulseGap = kwargs.get('pulseGap')
        # 脉冲数量
        self.pulseCnt = kwargs.get('pulseCnt')
        # 读方向flag，0：正向读， 1：反向读
        self.readDirection = kwargs.get('readDirection')

    def _regTableCfg(self):
        self._regCfg.regTableClear()
        self._regCfg.setChip(self.chipNum)
        self._regCfg.setMode(self.mode)
        self._regCfg.setFeedback(self.TIAFeedback)
        voltage = makeInputOrV(self.BLVActive, self.BLVInactive, self.BLStart,
                               self.BLCnt, reram_register.BL_COUNT)
        self._regCfg.setDacBLV(voltage)
        voltage = makeInputOrV(self.SLVActive, self.SLVInactive, self.SLStart,
                               self.SLCnt, reram_register.SL_COUNT)
        self._regCfg.setDacSLV(voltage)
        voltage = makeInputOrV(self.WLVActive, self.WLVInactive, self.WLStart,
                               self.WLCnt, reram_register.WL_COUNT)
        self._regCfg.setDacWLV(voltage)
        self._regCfg.setTIAV(self.TIAVoltage)

        inputSwitch = makeInputOrV(self.BLInputActive, self.BLInputInactive,
                                   self.BLStart, self.BLCnt,
                                   reram_register.BL_COUNT)
        self._regCfg.setBLInputSwitch(inputSwitch)
        inputSwitch = makeInputOrV(self.SLInputActive, self.SLInputInactive,
                                   self.SLStart, self.SLCnt,
                                   reram_register.SL_COUNT)
        self._regCfg.setSLInputSwitch(inputSwitch)
        inputSwitch = makeInputOrV(self.WLInputActive, self.WLInputInactive,
                                   self.WLStart, self.WLCnt,
                                   reram_register.WL_COUNT)
        self._regCfg.setWLInputSwitch(inputSwitch)
        self._regCfg.setPulse(self.pulseWidth, self.pulseGap, self.pulseCnt)

    def formOperation(self, addrInfo, VBL, VWL, pulseWidth, pulseGap,
                      pulseCnt):
        formDict = {
            'chipNum': addrInfo.chipNum,
            'mode': reram_register.FORM_MODE,
            'TIAFeedback': reram_register.R_250K,
            'BLStart': addrInfo.BLStart,
            'BLCnt': addrInfo.BLCnt,
            'SLStart': addrInfo.SLStart,
            'SLCnt': addrInfo.SLCnt,
            'WLStart': addrInfo.WLStart,
            'WLCnt': addrInfo.WLCnt,
            'BLInputActive': reram_register.DAC_SWITCH,
            'SLInputActive': reram_register.GND_SWITCH,
            'WLInputActive': reram_register.DAC_SWITCH,
            'BLVActive': VBL,
            'SLVActive': 0.0,
            'WLVActive': VWL,
            'BLInputInactive': reram_register.GND_SWITCH,
            'SLInputInactive': reram_register.GND_SWITCH,
            'WLInputInactive': reram_register.GND_SWITCH,
            'BLVInactive': 0.0,
            'SLVInactive': 0.0,
            'WLVInactive': 0.0,
            'TIAVoltage': 0.0,
            'pulseWidth': pulseWidth,
            'pulseGap': pulseGap,
            'pulseCnt': pulseCnt,
            'readDirection': None
        }
        self.configParas(**formDict)
        self._regTableCfg()
        ret = self.cmdCfgReg()
        return ret[0]

    def setOperation(self, addrInfo, VBL, VWL, pulseWidth=100, pulseGap=0, pulseCnt=1):
        setDict = {
            'chipNum': addrInfo.chipNum,
            'mode': reram_register.SET_MODE,
            'TIAFeedback': reram_register.R_250K,
            'BLStart': addrInfo.BLStart,
            'BLCnt': addrInfo.BLCnt,
            'SLStart': addrInfo.SLStart,
            'SLCnt': addrInfo.SLCnt,
            'WLStart': addrInfo.WLStart,
            'WLCnt': addrInfo.WLCnt,
            'BLInputActive': reram_register.DAC_SWITCH,
            'SLInputActive': reram_register.GND_SWITCH,
            'WLInputActive': reram_register.DAC_SWITCH,
            'BLVActive': VBL,
            'SLVActive': 0.0,
            'WLVActive': VWL,
            'BLInputInactive': reram_register.GND_SWITCH,
            'SLInputInactive': reram_register.GND_SWITCH,
            'WLInputInactive': reram_register.GND_SWITCH,
            'BLVInactive': 0.0,
            'SLVInactive': 0.0,
            'WLVInactive': 0.0,
            'TIAVoltage': 0.0,
            'pulseWidth': pulseWidth,
            'pulseGap': pulseGap,
            'pulseCnt': pulseCnt,
            'readDirection': None
        }
        self.configParas(**setDict)
        self._regTableCfg()
        ret = self.cmdCfgReg()
        return ret[0]

    def resetOperation(self, addrInfo, VSL, VWL, pulseWidth, pulseGap,
                       pulseCnt):
        resetDict = {
            'chipNum': addrInfo.chipNum,
            'mode': reram_register.RESET_MODE,
            'TIAFeedback': reram_register.R_250K,
            'BLStart': addrInfo.BLStart,
            'BLCnt': addrInfo.BLCnt,
            'SLStart': addrInfo.SLStart,
            'SLCnt': addrInfo.SLCnt,
            'WLStart': addrInfo.WLStart,
            'WLCnt': addrInfo.WLCnt,
            'BLInputActive': reram_register.GND_SWITCH,
            'SLInputActive': reram_register.DAC_SWITCH,
            'WLInputActive': reram_register.DAC_SWITCH,
            'BLVActive': 0.0,
            'SLVActive': VSL,
            'WLVActive': VWL,
            'BLInputInactive': reram_register.FLOAT_SWITCH, #DAC_SWITCH
            'SLInputInactive': reram_register.GND_SWITCH, #GND_SWITCH
            'WLInputInactive': reram_register.GND_SWITCH,
            'BLVInactive': 0.0,#VSL,
            'SLVInactive': 0.0,
            'WLVInactive': 0.0,
            'TIAVoltage': 0.0,
            'pulseWidth': pulseWidth,
            'pulseGap': pulseGap,
            'pulseCnt': pulseCnt,
            'readDirection': None
        }
        self.configParas(**resetDict)
        self._regTableCfg()
        ret = self.cmdCfgReg()
        return ret[0]

    def readOperation(self,
                      addrInfo,
                      VRead=0.2,
                      VWL=5,
                      readDirection=NEGATIVE_READ):
        if readDirection == POSITIVE_READ:
            readDict = {
                'chipNum': addrInfo.chipNum,
                'mode': reram_register.READ_MODE,
                'TIAFeedback': reram_register.R_250K,
                'BLStart': addrInfo.BLStart,
                'BLCnt': addrInfo.BLCnt,
                'SLStart': addrInfo.SLStart,
                'SLCnt': addrInfo.SLCnt,
                'WLStart': addrInfo.WLStart,
                'WLCnt': addrInfo.WLCnt,
                'BLInputActive': reram_register.DAC_SWITCH,
                'SLInputActive': reram_register.TIA_SWITCH,
                'WLInputActive': reram_register.DAC_SWITCH,
                'BLVActive': VRead,
                'SLVActive': 0.0,
                'WLVActive': VWL,
                'BLInputInactive': reram_register.GND_SWITCH,
                'SLInputInactive': reram_register.GND_SWITCH,
                'WLInputInactive': reram_register.GND_SWITCH,
                'BLVInactive': 0.0,
                'SLVInactive': 0.0,
                'WLVInactive': 0.0,
                'TIAVoltage': 0.0,
                'pulseWidth': 200000,
                'pulseGap': 0,
                'pulseCnt': 1,
                'readDirection': readDirection
            }
        elif readDirection == NEGATIVE_READ:
            readDict = {
                'chipNum': addrInfo.chipNum,
                'mode': reram_register.READ_MODE,
                'TIAFeedback': reram_register.R_250K,
                'BLStart': addrInfo.BLStart,
                'BLCnt': addrInfo.BLCnt,
                'SLStart': addrInfo.SLStart,
                'SLCnt': addrInfo.SLCnt,
                'WLStart': addrInfo.WLStart,
                'WLCnt': addrInfo.WLCnt,
                'BLInputActive': reram_register.TIA_SWITCH,
                'SLInputActive': reram_register.DAC_SWITCH,
                'WLInputActive': reram_register.DAC_SWITCH,
                'BLVActive': 0.0,
                'SLVActive': VRead,
                'WLVActive': VWL,
                'BLInputInactive': reram_register.GND_SWITCH,
                'SLInputInactive': reram_register.GND_SWITCH,
                'WLInputInactive': reram_register.GND_SWITCH,
                'BLVInactive': 0.0,
                'SLVInactive': 0.0,
                'WLVInactive': 0.0,
                'TIAVoltage': 0.0,
                'pulseWidth': 200000,
                'pulseGap': 0,
                'pulseCnt': 1,
                'readDirection': readDirection
            }
        else:
            raise ValueError('readDirection is POSITIVE_READ or NEGATIVE_READ')
        self.configParas(**readDict)
        self._regTableCfg()
        ret = self.cmdCfgReg()
        if ret[0]:
            regData = gf.bytesToUint32Array(ret[1], 'little')
            current = []
            for i in regData:
                current.append(self.regToCurrent(i & 0xFFFF))
                current.append(self.regToCurrent(i >> 16))

            if self.readDirection == POSITIVE_READ:
                current = gf.SpecifySort(current, TIA_SL_Table)
            else:
                current = gf.SpecifySort(current, TIA_BL_Table)
            return True, current
        else:
            return False, None

    def regToCurrent(self, value):
        adcVolt = value * 5.0 / 4095
        current = adcVolt / TIA_FEEDBACK[self.TIAFeedback] * 1000000
        return current  # uA

    def currentToReg(self, current):
        ret = (current * TIA_FEEDBACK[self.TIAFeedback] / 1e6) * 4095 / 5
        ret = int(round(ret))
        # assert (0 <= ret < 0x10000)
        return ret

    def readCurrentReg(self):
        ret = self.seqReadReg(REG160_ADDR, READ_DATA_SIZE_INT)
        if ret[0]:
            current = []
            for i in ret[1]:
                current.append(self.regToCurrent(i & 0xFFFF))
                current.append(self.regToCurrent(i >> 16))

            if self.readDirection == POSITIVE_READ:
                current = gf.SpecifySort(current, TIA_SL_Table)
            else:
                current = gf.SpecifySort(current, TIA_BL_Table)
            return True, current
        else:
            return False, None

    def pulseGeneratorUnuse(self,
                            voltageArr,
                            channel=Channel.WL,
                            pulseWidth=1e6):
        self._regCfg.regTableClear()
        self._regCfg.setChip(1)
        self._regCfg.setMode(reram_register.FORM_MODE)
        self._regCfg.setFeedback(reram_register.R_250K)
        self._regCfg.setPulse(pulseWidth, 0, 1)
        if channel == Channel.BL:
            self._regCfg.setDacBLV(voltageArr)
            inputSwitch = reram_register.BL_COUNT * [reram_register.DAC_SWITCH]
            self._regCfg.setBLInputSwitch(inputSwitch)
        elif channel == Channel.SL:
            self._regCfg.setDacSLV(voltageArr)
            inputSwitch = reram_register.SL_COUNT * [reram_register.DAC_SWITCH]
            self._regCfg.setSLInputSwitch(inputSwitch)
        elif channel == Channel.WL:
            self._regCfg.setDacWLV(voltageArr)
            inputSwitch = reram_register.WL_COUNT * [reram_register.DAC_SWITCH]
            self._regCfg.setWLInputSwitch(inputSwitch)
        else:
            return False

        ret = self.cmdCfgReg()
        return ret[0]

    def applyOp(self, finderCfg, addrInfo, targetList):
        CMD_APPLY_OP = 1
        applyOpType = 0
        # TIAFeedback = 3
        cmdNum = CMD_APPLY_OP
        para1 = 0
        para2 = 0
        payload = []
        payload.append(applyOpType)
        payload.append(self.TIAFeedback)
        payload.append(gf.VToReg(finderCfg.formVWLStart))
        payload.append(gf.VToReg(finderCfg.formVWLStep))
        payload.append(gf.VToReg(finderCfg.formVWLEnd))
        payload.append(gf.VToReg(finderCfg.formVBLStart))
        payload.append(gf.VToReg(finderCfg.formVBLStep))
        payload.append(gf.VToReg(finderCfg.formVBLEnd))
        payload.append(gf.VToReg(finderCfg.setVWLStart))
        payload.append(gf.VToReg(finderCfg.setVWLStep))
        payload.append(gf.VToReg(finderCfg.setVWLEnd))
        payload.append(gf.VToReg(finderCfg.setVBLStart))
        payload.append(gf.VToReg(finderCfg.setVBLStep))
        payload.append(gf.VToReg(finderCfg.setVBLEnd))
        payload.append(gf.VToReg(finderCfg.rstVWLStart))
        payload.append(gf.VToReg(finderCfg.rstVWLStep))
        payload.append(gf.VToReg(finderCfg.rstVWLEnd))
        payload.append(gf.VToReg(finderCfg.rstVSLStart))
        payload.append(gf.VToReg(finderCfg.rstVSLStep))
        payload.append(gf.VToReg(finderCfg.rstVSLEnd))
        payload.append(self.currentToReg(finderCfg.errorLimit))
        payload.append(finderCfg.nMax)
        payload.append(finderCfg.readDirection)
        payload.append(addrInfo.BLStart)
        payload.append(addrInfo.BLCnt)
        payload.append(addrInfo.SLStart)
        payload.append(addrInfo.SLCnt)
        payload = gf.uint32ArrayToBytes(payload, 'little')
        self.sendPacket(cmdNum, para1, para2, payload)
        return self.disposeReport()

    def cmdMapProgress(self):
        cmdNum = CMD_MAP_PROGRESS
        para1 = 0
        para2 = 0
        payload = []
        self.sendPacket(cmdNum, para1, para2, payload)
        return self.disposeReport()

    def cmdMapAbort(self):
        cmdNum = CMD_MAP_ABORT
        para1 = 0
        para2 = 0
        payload = []
        self.sendPacket(cmdNum, para1, para2, payload)
        return self.disposeReport()

    def mapProgress(self):
        ret = self.cmdMapProgress()
        progress = None
        if ret[0]:
            progress = gf.bytesToUint32(ret[1], 'little')
        return ret[0], progress

    def cmdSeqRead(self, BLActive, SLActive, arrReadV):
        cmdNum = CMD_SEQ_READ
        para1 = BLActive
        para2 = SLActive
        # arrRegV = list(map(gf.VToReg(), arrReadV))
        arrRegV = [gf.VToReg(i) for i in arrReadV]
        payload = gf.uint32ArrayToBytes(arrRegV, 'little')
        self.sendPacket(cmdNum, para1, para2, payload)
        return self.disposeReport()

    def seqRead(self, BLActive, SLActive, arrReadV):
        # 计算电流需要用这个反馈电阻，需要和下位机设定的相同
        self.TIAFeedback = reram_register.R_250K
        ret = self.cmdSeqRead(BLActive, SLActive, arrReadV)
        retData = None
        if ret[0]:
            retData = [
                self.regToCurrent(i)
                for i in gf.bytesToUint32Array(ret[1], 'little')
            ]
        return ret[0], retData

    def cmdApplyOp(self, paraList):
        cmdNum = CMD_APPLY_OP
        para1 = 0
        para2 = 0
        payload = gf.uint32ArrayToBytes(paraList, 'little')
        self.sendPacket(cmdNum, para1, para2, payload)
        return self.disposeReport()

    def cmdOpResult(self, offset=0, count=16383):
        cmdNum = CMD_OP_RESULT
        para1 = 0
        para2 = 0
        payload = [offset, count]
        payload = gf.uint32ArrayToBytes(payload, 'little')
        self.sendPacket(cmdNum, para1, para2, payload)
        return self.disposeReport()

    def opResult(self, offset=0, count=16383):
        ret = self.cmdOpResult(offset, count)
        retData = None
        if ret[0]:
            retData = gf.bytesToUint32Array(ret[1], 'little')
        return ret[0], retData

    def cmdMapResult(self):
        cmdNum = CMD_MAP_RESULT
        para1 = 0
        para2 = 0
        payload = []
        self.sendPacket(cmdNum, para1, para2, payload)
        return self.disposeReport()

    def mapResult(self):
        ret = self.cmdMapResult()
        retData = None
        if ret[0]:
            retData = [
                self.regToCurrent(i)
                for i in gf.bytesToUint32Array(ret[1], 'little')
            ]
        return ret[0], retData

    # targetList: uint(uA)
    def cmdMapFlow(self, finderCfg, addrInfo, targetList):
        # 计算电流需要用这个反馈电阻，需要和下位机设定的相同
        self.TIAFeedback = reram_register.R_250K
        cmdNum = CMD_MAP_FLOW
        para1 = 0
        para2 = 0
        payload = []
        payload.append(gf.VToReg(finderCfg.formVWLStart))
        payload.append(gf.VToReg(finderCfg.formVWLStep))
        payload.append(gf.VToReg(finderCfg.formVWLEnd))
        payload.append(gf.VToReg(finderCfg.formVBLStart))
        payload.append(gf.VToReg(finderCfg.formVBLStep))
        payload.append(gf.VToReg(finderCfg.formVBLEnd))
        payload.append(gf.VToReg(finderCfg.setVWLStart))
        payload.append(gf.VToReg(finderCfg.setVWLStep))
        payload.append(gf.VToReg(finderCfg.setVWLEnd))
        payload.append(gf.VToReg(finderCfg.setVBLStart))
        payload.append(gf.VToReg(finderCfg.setVBLStep))
        payload.append(gf.VToReg(finderCfg.setVBLEnd))
        payload.append(gf.VToReg(finderCfg.rstVWLStart))
        payload.append(gf.VToReg(finderCfg.rstVWLStep))
        payload.append(gf.VToReg(finderCfg.rstVWLEnd))
        payload.append(gf.VToReg(finderCfg.rstVSLStart))
        payload.append(gf.VToReg(finderCfg.rstVSLStep))
        payload.append(gf.VToReg(finderCfg.rstVSLEnd))
        payload.append(self.currentToReg(finderCfg.errorLimit))
        payload.append(finderCfg.nMax)
        payload.append(finderCfg.readDirection)
        payload.append(addrInfo.chipNum)
        payload.append(addrInfo.BLStart)
        payload.append(addrInfo.BLCnt)
        payload.append(addrInfo.SLStart)
        payload.append(addrInfo.SLCnt)
        targetList = list(map(self.currentToReg, targetList))
        payload.extend(targetList)
        payload = gf.uint32ArrayToBytes(payload, 'little')
        self.sendPacket(cmdNum, para1, para2, payload)
        return self.disposeReport()

    def mapFlow(self, finderCfg, addrInfo, targetList):
        ret = self.cmdMapFlow(finderCfg, addrInfo, targetList)
        retData = None
        if ret[0]:
            retData = [
                self.regToCurrent(i)
                for i in gf.bytesToUint32Array(ret[1], 'little')
            ]
        return ret[0], retData

    def cmdSetTarget(self, finderCfg, addrInfo, targetList):
        # 计算电流需要用这个反馈电阻，需要和下位机设定的相同
        self.TIAFeedback = reram_register.R_250K
        cmdNum = CMD_SET_TARGET
        para1 = 0
        para2 = 0
        payload = []
        payload.append(gf.VToReg(finderCfg.setVWLStart))
        payload.append(gf.VToReg(finderCfg.setVWLStep))
        payload.append(gf.VToReg(finderCfg.setVWLEnd))
        payload.append(gf.VToReg(finderCfg.setVBLStart))
        payload.append(gf.VToReg(finderCfg.setVBLStep))
        payload.append(gf.VToReg(finderCfg.setVBLEnd))
        payload.append(finderCfg.nMax)
        payload.append(finderCfg.readDirection)
        payload.append(finderCfg.voltageIncMode)
        payload.append(addrInfo.chipNum)
        payload.append(addrInfo.BLStart)
        payload.append(addrInfo.BLCnt)
        payload.append(addrInfo.SLStart)
        payload.append(addrInfo.SLCnt)
        targetList = list(map(self.currentToReg, targetList))
        payload.extend(targetList)
        payload = gf.uint32ArrayToBytes(payload, 'little')
        self.sendPacket(cmdNum, para1, para2, payload)
        return self.disposeReport()

    def setTarget(self, finderCfg, addrInfo, targetList):
        ret = self.cmdSetTarget(finderCfg, addrInfo, targetList)
        retData = None
        if ret[0]:
            retData = [
                self.regToCurrent(i)
                for i in gf.bytesToUint32Array(ret[1], 'little')
            ]
        return ret[0], retData

    def cmdResetTarget(self, finderCfg, addrInfo, targetList):
        # 计算电流需要用这个反馈电阻，需要和下位机设定的相同
        self.TIAFeedback = reram_register.R_250K
        cmdNum = CMD_RESET_TARGET
        para1 = 0
        para2 = 0
        payload = []
        payload.append(gf.VToReg(finderCfg.rstVWLStart))
        payload.append(gf.VToReg(finderCfg.rstVWLStep))
        payload.append(gf.VToReg(finderCfg.rstVWLEnd))
        payload.append(gf.VToReg(finderCfg.rstVSLStart))
        payload.append(gf.VToReg(finderCfg.rstVSLStep))
        payload.append(gf.VToReg(finderCfg.rstVSLEnd))
        payload.append(finderCfg.nMax)
        payload.append(finderCfg.readDirection)
        payload.append(finderCfg.voltageIncMode)
        payload.append(addrInfo.chipNum)
        payload.append(addrInfo.BLStart)
        payload.append(addrInfo.BLCnt)
        payload.append(addrInfo.SLStart)
        payload.append(addrInfo.SLCnt)
        targetList = list(map(self.currentToReg, targetList))
        payload.extend(targetList)
        payload = gf.uint32ArrayToBytes(payload, 'little')
        self.sendPacket(cmdNum, para1, para2, payload)
        return self.disposeReport()

    def resetTarget(self, finderCfg, addrInfo, targetList):
        ret = self.cmdResetTarget(finderCfg, addrInfo, targetList)
        retData = None
        if ret[0]:
            retData = [
                self.regToCurrent(i)
                for i in gf.bytesToUint32Array(ret[1], 'little')
            ]
        return ret[0], retData

    def cmdMapRead(self, readConfig):
        # 计算电流需要用这个反馈电阻
        self.TIAFeedback = readConfig.TIAFeedback
        cmdNum = CMD_MAP_READ
        para1 = 0
        para2 = 0
        payload = []
        payload.append(readConfig.chipNum)
        payload.append(readConfig.BLActive)
        payload.append(readConfig.SLActive)
        payload.append(readConfig.readDirection)
        payload.append(gf.VToReg(readConfig.readV))
        payload.append(gf.VToReg(readConfig.accessV))
        payload.append(readConfig.TIAFeedback)
        payload = gf.uint32ArrayToBytes(payload, 'little')
        self.sendPacket(cmdNum, para1, para2, payload)
        return self.disposeReport()

    def mapRead(self, readConfig):
        ret = self.cmdMapRead(readConfig)
        retData = 0
        if ret[0]:
            retData = gf.bytesToUint32(ret[1], 'little')
            retData = self.regToCurrent(retData)
        return ret[0], retData

    # pulseWidth(20us), pulseGap(20us), delay 1.08ms
    def cmdPulseGenerator(self, pulseWidth, pulseGap, voltageData):
        regArr = np.zeros(len(voltageData) * 16, dtype='<u4')
        for index, VList in enumerate(voltageData):
            for RegNum in range(16):
                RegTemp1 = gf.VToReg(VList[RegNum * 2])
                RegTemp2 = gf.VToReg(VList[RegNum * 2 + 1])
                regArr[index * 16 + RegNum] = (RegTemp2 << 16) | RegTemp1

        cmdNum = CMD_PULSE_GENERATOR
        para1 = pulseWidth
        para2 = pulseGap
        payload = bytes(regArr)
        self.sendPacket(cmdNum, para1, para2, payload)
        return self.disposeReport()

    # 32路SL电压值，只有选中的SL那些路的值有效
    def parallelRead(self,
                     readV,
                     SLStart=0,
                     SLCnt=32,
                     BLStart=0,
                     BLCnt=18,
                     TIAFeedback=reram_register.R_250K):
        self.TIAFeedback = TIAFeedback
        self._regCfg.regTableClear()
        self._regCfg.setChip(1)
        self._regCfg.setMode(reram_register.READ_MODE)
        self._regCfg.setFeedback(TIAFeedback)
        voltage = reram_register.BL_COUNT * [0]
        self._regCfg.setDacBLV(voltage)
        self._regCfg.setDacSLV(readV)
        voltage = reram_register.WL_COUNT * [5]
        self._regCfg.setDacWLV(voltage)

        inputSwitch = makeInputOrV(reram_register.TIA_SWITCH,
                                   reram_register.GND_SWITCH, BLStart, BLCnt,
                                   reram_register.BL_COUNT)
        self._regCfg.setBLInputSwitch(inputSwitch)
        inputSwitch = makeInputOrV(reram_register.DAC_SWITCH,
                                   reram_register.GND_SWITCH, SLStart, SLCnt,
                                   reram_register.SL_COUNT)
        self._regCfg.setSLInputSwitch(inputSwitch)
        inputSwitch = makeInputOrV(reram_register.DAC_SWITCH,
                                   reram_register.GND_SWITCH, SLStart, SLCnt,
                                   reram_register.WL_COUNT)
        self._regCfg.setWLInputSwitch(inputSwitch)
        self._regCfg.setPulse(10000, 0, 1)

        ret = self.cmdCfgReg()
        if ret[0]:
            regData = gf.bytesToUint32Array(ret[1], 'little')
            current = []
            for i in regData:
                current.append(self.regToCurrent(i & 0xFFFF))
                current.append(self.regToCurrent(i >> 16))

            current = gf.SpecifySort(current, TIA_BL_Table)
            return True, current
        else:
            return False, None

    def cmdMapArray(self, mapPara, addrInfo, targetList):
        # print(mapPara.relaxSleep)
        self.TIAFeedback = reram_register.R_250K
        cmdNum = CMD_MAP_ARRAY
        para1 = 0
        para2 = 0
        payload = []
        payload.append(gf.VToReg(mapPara.setVWLStart))
        payload.append(gf.VToReg(mapPara.setVWLStep))
        payload.append(gf.VToReg(mapPara.setVWLEnd))
        payload.append(gf.VToReg(mapPara.setVBLStart))
        payload.append(gf.VToReg(mapPara.setVBLStep))
        payload.append(gf.VToReg(mapPara.setVBLEnd))
        payload.append(gf.VToReg(mapPara.rstVWLStart))
        payload.append(gf.VToReg(mapPara.rstVWLStep))
        payload.append(gf.VToReg(mapPara.rstVWLEnd))
        payload.append(gf.VToReg(mapPara.rstVSLStart))
        payload.append(gf.VToReg(mapPara.rstVSLStep))
        payload.append(gf.VToReg(mapPara.rstVSLEnd))
        payload.append(self.currentToReg(mapPara.errorHigh))
        payload.append(self.currentToReg(mapPara.errorLow))
        payload.append(mapPara.maxProgramNum)
        payload.append(mapPara.maxCheckNum)
        payload.append(mapPara.checkThreshold)
        payload.append(mapPara.relaxSleep)
        payload.append(mapPara.setPulseWidth // reram_register.PULSE_UNIT_TIME)
        payload.append(mapPara.rstPulseWidth // reram_register.PULSE_UNIT_TIME)
        payload.append(addrInfo.chipNum)
        payload.append(addrInfo.BLStart)
        payload.append(addrInfo.BLCnt)
        payload.append(addrInfo.SLStart)
        payload.append(addrInfo.SLCnt)
        targetList = list(map(self.currentToReg, targetList))
        payload.extend(targetList)
        payload = gf.uint32ArrayToBytes(payload, 'little')
        self.sendPacket(cmdNum, para1, para2, payload)
        return self.disposeReport()

    def cmdCellPara(self, offset=0, count=1489):
        cmdNum = CMD_CELL_PARA
        para1 = 0
        para2 = 0
        payload = [offset, count]
        payload = gf.uint32ArrayToBytes(payload, 'little')
        self.sendPacket(cmdNum, para1, para2, payload)
        return self.disposeReport()

    def cellPara(self, offset=0, count=1489):
        ret = self.cmdCellPara(offset, count)
        retData = None
        if ret[0]:
            retData = gf.bytesToUint32Array(ret[1], 'little')
        return ret[0], retData

    def cmdReadAll(self, chipNum):
        cmdNum = CMD_READ_ALL
        para1 = chipNum
        para2 = 0
        payload = []
        self.sendPacket(cmdNum, para1, para2, payload)
        return self.disposeReport()

    def readAll(self, chipNum):
        # 计算电流需要用这个反馈电阻，需要和下位机设定的相同
        self.TIAFeedback = reram_register.R_250K
        ret = self.cmdReadAll(chipNum)
        retData = None
        if ret[0]:
            retData = [
                self.regToCurrent(i)
                for i in gf.bytesToUint16Array(ret[1], 'little')
            ]
        return ret[0], retData
