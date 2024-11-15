import time
import reram_register
import globals.global_func as gf
import apps
import numpy as np
import arrayparams
import math

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

TIA_FEEDBACK = [24.9, 2500, 24900, 249000]

# 
#
# 0-200uA
# 0-20uA
class ArrayExp():
    """
    docstring
    """
    def __init__(self, dut_ip='101.6.93.189'):
        self.myapp = apps.App()
        # Connect the device
        self.myapp.client.connect(dut_ip, 7, 2)
        # No connection is successful until 3 seconds, an exception is thrown
        for i in range(3):
            if i == 3:
                raise Exception('Connect failed')
            if not self.myapp.client.isConnected():
                time.sleep(0.1)
                print('Connecting to device')
            else:
                break
        print(1)
        self.myapp.client.startListener()  # Start monitoring the port
        print(2)
        self.params = arrayparams.ArrayParams()
        print(3)
        self.myapp.TIAFeedback = 3

        self.r2i = [5.0/4095/i*1000000 for i in TIA_FEEDBACK]

    def abort(self):
        myapp = apps.App()
        # Connect the device
        myapp.client.connect('192.168.1.10', 5000, 2)

        # No connection is successful until 3 seconds, an exception is thrown
        for i in range(4):
            if i == 3:
                raise Exception('Connect failed')
            if not myapp.client.isConnected():
                time.sleep(0.01)
                print('Connecting to device')
            else:
                break

        myapp.client.startListener()  # Start monitoring the port
        ret = myapp.cmdMapAbort()
        if ret[0]:
            print('Success abort')
        else:
            print('Fail')

    def ReadOp(self,
               BLStart,
               BLCnt,
               WLStart,
               WLCnt,
               Vread=0.2,
               RdDirect=NEGATIVE_READ):
        pass

    def SetReadOp(self,
                  BLStart,
                  BLCnt,
                  WLStart,
                  WLCnt,
                  VBLStart,
                  VBLStep,
                  VBLEnd,
                  VWLStart,
                  VWLStep,
                  VWLEnd,
                  Target,
                  PulseMax=300,
                  RdDirect=NEGATIVE_READ,
                  voltageIncMode=FIRST_INC_WLV,
                  OnFPGA=True):
        self.myapp.TIAFeedback = reram_register.R_250K
        cmdNum = CMD_SET_TARGET
        para1 = 0
        para2 = 0
        payload = []
        payload.append(gf.VToReg(VWLStart))
        payload.append(gf.VToReg(VWLStep))
        payload.append(gf.VToReg(VWLEnd))
        payload.append(gf.VToReg(VBLStart))
        payload.append(gf.VToReg(VBLStep))
        payload.append(gf.VToReg(VBLEnd))
        payload.append(PulseMax)
        payload.append(RdDirect)
        payload.append(voltageIncMode)
        payload.append(BLStart)
        payload.append(BLCnt)
        payload.append(WLStart)
        payload.append(WLCnt)
        targetList = list(map(self.myapp.currentToReg, [Target]*(BLCnt*WLCnt)))
        payload.extend(targetList)
        payload = gf.uint32ArrayToBytes(payload, 'little')
        self.myapp.sendPacket(cmdNum, para1, para2, payload)
        return self.myapp.disposeReport()

    def ResetReadOp(self,
                    BLStart,
                    BLCnt,
                    WLStart,
                    WLCnt,
                    VSLStart,
                    VSLStep,
                    VSLEnd,
                    VWLStart,
                    VWLStep,
                    VWLEnd,
                    Target,
                    PulseMax=300,
                    RdDirect=NEGATIVE_READ,
                    voltageIncMode=FIRST_INC_WLV,
                    OnFPGA=True):
        self.myapp.TIAFeedback = reram_register.R_250K
        cmdNum = CMD_RESET_TARGET
        para1 = 0
        para2 = 0
        payload = []
        payload.append(gf.VToReg(VWLStart))
        payload.append(gf.VToReg(VWLStep))
        payload.append(gf.VToReg(VWLEnd))
        payload.append(gf.VToReg(VSLStart))
        payload.append(gf.VToReg(VSLStep))
        payload.append(gf.VToReg(VSLEnd))
        payload.append(PulseMax)
        payload.append(RdDirect)
        payload.append(voltageIncMode)
        payload.append(BLStart)
        payload.append(BLCnt)
        payload.append(WLStart)
        payload.append(WLCnt)
        targetList = list(map(self.myapp.currentToReg, [Target]*(BLCnt*WLCnt)))
        payload.extend(targetList)
        payload = gf.uint32ArrayToBytes(payload, 'little')
        self.myapp.sendPacket(cmdNum, para1, para2, payload)
        Tag = self.myapp.disposeReport()[0]
        if Tag == PAYLOAD_CHECK_ERROR:
            print('[ERROR]: Found payload check error') 
            while (Tag == PAYLOAD_CHECK_ERROR):
                self.myapp.sendPacket(cmdNum, para1, para2, payload)
                Tag = self.myapp.disposeReport()[0]
        return Tag, None

    def close(self):
        self.myapp.client.close()

    def ApplyOp(self, paraList):
        self.myapp.cmdApplyOp(paraList)

    def GetOpResult(self, offset=0, count=16383, op=0):
        cmdNum = CMD_OP_RESULT
        para1 = 0
        para2 = 0
        ret = None
        data = None
        SplitCount = 10000
        if count > SplitCount:
            data = np.zeros(count)
            num = math.floor(count / SplitCount)
            for i in range(num):
                payload = bytes(np.array([offset+i*SplitCount, SplitCount], dtype='<u4'))
                for re in range(10):
                    self.myapp.sendPacket(cmdNum, para1, para2, payload)
                    ret = self.myapp.disposeReport()
                    if ret[1] is None:
                        continue
                    if ret[0]:
                        if re > 0:
                            print('[PASS]: Receive data again pass')
                        data[i * SplitCount:(i + 1) * SplitCount] = np.frombuffer(ret[1], dtype='<u4')
                        break
            if (count%SplitCount > 0):
                payload = bytes(np.array([offset+num*SplitCount, count%SplitCount], dtype='<u4'))
                for re in range(10):
                    self.myapp.sendPacket(cmdNum, para1, para2, payload)
                    ret = self.myapp.disposeReport()
                    if ret[1] is None:
                        continue 
                    if ret[0]:
                        if re > 0:
                            print('[PASS]: Receive data again pass')
                        data[num*SplitCount:] = np.frombuffer(ret[1], dtype='<u4')
                        break
        else:
            payload = bytes(np.array([offset, count], dtype='<u4'))
            for re in range(10):
                self.myapp.sendPacket(cmdNum, para1, para2, payload)
                ret = self.myapp.disposeReport()
                #if ret[1] is None:
                #    continue
                if ret[0]:
                    if re > 0:
                        print('[PASS]: Receive data again pass')
                    data = np.frombuffer(ret[1], dtype='<u4')
                    break
        return data



    def FormOp(self,
               BLStart,
               BLCnt,
               WLStart,
               WLCnt,
               VBLStart,
               VBLStep,
               VBLEnd,
               VWLStart,
               VWLStep,
               VWLEnd,
               ITarget,
               ReadV=0.2,
               AccessV=2.5,
               PulseWidth=100000,
               PulseMax=300,
               LogLevel=0,
               RdDirect=NEGATIVE_READ,
               TIAFeedback=3,
               VoltMode=1,
               OnFPGA=True):
        self.TIAFeedback = TIAFeedback
        self.myapp.TIAFeedback = TIAFeedback
        cmdNum = CMD_APPLY_OP
        para1 = 0
        para2 = 0
        paraList = [
            1, BLStart, BLCnt, WLStart, WLCnt, PulseWidth // 100, PulseMax,
            gf.VToReg(ReadV),
            gf.VToReg(AccessV),
            gf.VToReg(VWLStart),
            gf.VToReg(VWLStep),
            gf.VToReg(VWLEnd),
            gf.VToReg(VBLStart),
            gf.VToReg(VBLStep),
            gf.VToReg(VBLEnd),
            self.myapp.currentToReg(ITarget),
            LogLevel,
            TIAFeedback,
            RdDirect,
            VoltMode
        ]
        payload = gf.uint32ArrayToBytes(paraList, 'little')
        ret = None
        for _ in range(10):
            self.myapp.sendPacket(cmdNum, para1, para2, payload)
            ret = self.myapp.disposeReport()
            if ret[0]:
                print('[PASS]: Resend Check Pass') 
                break
        return ret

    def SetOp_MC(self, addrInfo):
        self.myapp._regCfg.setChip(addrInfo.chipNum)
        VBL = 2.0
        VWL = 1.2
        ret = self.SetOp(addrInfo.BLStart, addrInfo.BLCnt, addrInfo.WLStart, addrInfo.WLCnt, VBLStart=VBL, VBLStep=0.1,
                        VBLEnd=VBL+0.5, VWLStart=VWL, VWLStep=0.1, VWLEnd=VWL+0.5, ITarget=4., # PulseMax = 1 to ignore this setting
                        ReadV=0.2, AccessV=1.8, PulseWidth=100, PulseMax=3, LogLevel=2, RdDirect=NEGATIVE_READ,
                        TIAFeedback=reram_register.R_250K, VoltMode=1)

        return ret

    # def SetOp(self, BLStart, BLCnt, WLStart, WLCnt, VBLStart, VBLStep, VBLEnd, VWLStart, VWLStep, VWLEnd, ITarget, ReadV=0.2, AccessV=1.8, PulseWidth=100, PulseMax=1, LogLevel=0, RdDirect=NEGATIVE_READ, TIAFeedback=3, VoltMode=1, OnFPGA=True):
    def SetOp(self, BLStart, BLCnt, WLStart, WLCnt, VBLStart, VBLStep, VBLEnd, VWLStart, VWLStep, VWLEnd, ITarget,
              ReadV=0.2, AccessV=1.8, PulseWidth=100, PulseMax=1, LogLevel=0, RdDirect=NEGATIVE_READ, TIAFeedback=3,
              VoltMode=1, OnFPGA=True):

        self.TIAFeedback = TIAFeedback
        self.myapp.TIAFeedback = TIAFeedback
        cmdNum = CMD_APPLY_OP
        para1 = 0
        para2 = 0
        paraList = [
            2, BLStart, BLCnt, WLStart, WLCnt, PulseWidth // 100, PulseMax,
            gf.VToReg(ReadV),
            gf.VToReg(AccessV),
            gf.VToReg(VWLStart),
            gf.VToReg(VWLStep),
            gf.VToReg(VWLEnd),
            gf.VToReg(VBLStart),
            gf.VToReg(VBLStep),
            gf.VToReg(VBLEnd),
            self.myapp.currentToReg(ITarget),
            LogLevel,
            TIAFeedback,
            RdDirect,
            VoltMode
        ]
        payload = gf.uint32ArrayToBytes(paraList, 'little')
        ret = None
        for re in range(10):
            self.myapp.sendPacket(cmdNum, para1, para2, payload)
            ret = self.myapp.disposeReport()
            if ret[0]:
                if re > 0:
                    print('[PASS]: Resend operation pass')
                break
        return ret


    def ResetOp_MC(self, addrInfo):
        self.myapp._regCfg.setChip(addrInfo.chipNum)
        VSL = 2.5
        VWL = 4
        ret = self.ResetOp(addrInfo.BLStart, addrInfo.BLCnt, addrInfo.WLStart, addrInfo.WLCnt, VSLStart=VSL, VSLStep=0.1,
                        VSLEnd=VSL+0.5, VWLStart=VWL, VWLStep=0.1, VWLEnd=VWL+0.5, ITarget=0.4, # PulseMax = 1 to ignore this setting
                        ReadV=0.2, AccessV=1.8, PulseWidth=100, PulseMax=3, LogLevel=2, RdDirect=NEGATIVE_READ,
                        TIAFeedback=reram_register.R_250K, VoltMode=1)

        return ret
    def ResetOp(self,
                BLStart,
                BLCnt,
                WLStart,
                WLCnt,
                VSLStart,
                VSLStep,
                VSLEnd,
                VWLStart,
                VWLStep,
                VWLEnd,
                ITarget,
                ReadV=0.2,
                AccessV=1.8,
                PulseWidth=100,
                PulseMax=1,
                LogLevel=0,
                RdDirect=NEGATIVE_READ,
                TIAFeedback=3,
                VoltMode=1,
                OnFPGA=True):
        self.TIAFeedback = TIAFeedback
        self.myapp.TIAFeedback = TIAFeedback
        cmdNum = CMD_APPLY_OP
        para1 = 0
        para2 = 0
        paraList = [
            3, BLStart, BLCnt, WLStart, WLCnt, PulseWidth // 100, PulseMax,
            gf.VToReg(ReadV),
            gf.VToReg(AccessV),
            gf.VToReg(VWLStart),
            gf.VToReg(VWLStep),
            gf.VToReg(VWLEnd),
            gf.VToReg(VSLStart),
            gf.VToReg(VSLStep),
            gf.VToReg(VSLEnd),
            self.myapp.currentToReg(ITarget),
            LogLevel,
            TIAFeedback,
            RdDirect,
            VoltMode
        ]
        payload = gf.uint32ArrayToBytes(paraList, 'little')
        ret = None
        for re in range(10):
            self.myapp.sendPacket(cmdNum, para1, para2, payload)
            ret = self.myapp.disposeReport()
            if ret[0]:
                if re > 0:
                    print('[PASS]: Resend operation pass')
                break
        return ret

    def MapOp(self, ArrayW, params):
        # Numpy Array 32x128
        cmdNum = CMD_APPLY_OP
        para1 = 0
        para2 = 0
        paraList = [
            4, params.BLStart, params.BLCnt, params.WLStart, params.WLCnt,
            params.PulseWidth // 100, params.PulseMax, gf.VToReg(params.ReadV),
            gf.VToReg(params.AccessV),
            gf.VToReg(params.FormVWLStart),
            gf.VToReg(params.FormVWLStep),
            gf.VToReg(params.FormVWLEnd),
            gf.VToReg(params.FormVBLStart),
            gf.VToReg(params.FormVBLStep),
            gf.VToReg(params.FormVBLEnd),
            gf.VToReg(params.SetVWLStart),
            gf.VToReg(params.SetVWLStep),
            gf.VToReg(params.SetVWLEnd),
            gf.VToReg(params.SetVBLStart),
            gf.VToReg(params.SetVBLStep),
            gf.VToReg(params.SetVBLEnd),
            gf.VToReg(params.RstVWLStart),
            gf.VToReg(params.RstVWLStep),
            gf.VToReg(params.RstVWLEnd),
            gf.VToReg(params.RstVSLStart),
            gf.VToReg(params.RstVSLStep),
            gf.VToReg(params.RstVSLEnd),
            self.myapp.currentToReg(params.ErrorHigh),
            self.myapp.currentToReg(params.ErrorLow),
            params.LogLevel,
            params.TIAFeedback,
            params.ReadDirection,
            params.VoltMode,
            params.RdAfterW
        ]
        targetList = list(map(self.myapp.currentToReg, ArrayW.reshape(-1).tolist()))
        payload = gf.uint32ArrayToBytes(paraList+targetList, 'little')
        ret = None
        for re in range(10):
            self.myapp.sendPacket(cmdNum, para1, para2, payload)
            ret = self.myapp.disposeReport()
            if ret[0]:
                if re > 0:
                    print('[PASS]: Resend operation pass')
                break
        return ret
    
    def ParallelMapOp(self, ArrayW, params):
        # Numpy Array 32x128
        cmdNum = CMD_APPLY_OP
        para1 = 0
        para2 = 0
        paraList = [
            5, params.BLStart, params.BLCnt, params.WLStart, params.WLCnt,
            params.PulseWidth // 100, params.PulseMax, gf.VToReg(params.ReadV),
            gf.VToReg(params.FormVWLStart),
            gf.VToReg(params.FormVWLStep),
            gf.VToReg(params.FormVWLEnd),
            gf.VToReg(params.FormVBLStart),
            gf.VToReg(params.FormVBLStep),
            gf.VToReg(params.FormVBLEnd),
            gf.VToReg(params.SetVWLStart),
            gf.VToReg(params.SetVWLStep),
            gf.VToReg(params.SetVWLEnd),
            gf.VToReg(params.SetVBLStart),
            gf.VToReg(params.SetVBLStep),
            gf.VToReg(params.SetVBLEnd),
            gf.VToReg(params.RstVWLStart),
            gf.VToReg(params.RstVWLStep),
            gf.VToReg(params.RstVWLEnd),
            gf.VToReg(params.RstVSLStart),
            gf.VToReg(params.RstVSLStep),
            gf.VToReg(params.RstVSLEnd),
            self.myapp.currentToReg(params.Errorbar),
            params.BLPar,
            params.WLPar
        ]
        targetList = list(map(self.myapp.currentToReg, ArrayW.reshape(-1).tolist()))
        print(targetList)
        payload = gf.uint32ArrayToBytes(paraList+targetList, 'little')
        self.myapp.sendPacket(cmdNum, para1, para2, payload)
        return self.myapp.disposeReport()


if __name__ == '__main__':
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    arrayexp = ArrayExp()
    ArrayW = np.asarray([[2.3, 2.5], [3.5, 1.2]])
    # print(ArrayW)
    ErrorMargin = 0.1
    BLStart = 5
    WLStart = 5
    BLCnt = ArrayW.shape[1]
    WLCnt = ArrayW.shape[0]
    print(arrayexp.MapOp(ArrayW, ErrorMargin, BLStart, BLCnt, WLStart, WLCnt))
    arrayexp.myapp.client.close()
