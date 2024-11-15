import os
import threading
import traceback

from PyQt5.QtWidgets import QWidget
from PyQt5.uic import loadUi

import reram_register
import apps
import globals.global_func as gf

FILE_PATH, FILE_FULL_NAME = os.path.split(os.path.realpath(__file__))
FILE_NAME, FILE_EXT = os.path.splitext(FILE_FULL_NAME)


def taskThread(obj, taskFunc):
    try:
        print('Start operation')
        obj.app.bBusy = True

        taskFunc()

    except Exception:
        print(traceback.format_exc())
    finally:
        obj.app.bBusy = False
        print('Stop operation')


class HardwareDebug(QWidget):
    def __init__(self, app):
        super(HardwareDebug, self).__init__()
        loadUi(FILE_PATH + '/' + FILE_NAME + '.ui', self)
        self.app = app
        self.PB_operation.clicked.connect(self.PB_operation_click)

    def runTask(self, taskFunc):
        try:
            if not self.app.checkDevice():
                return

            args = (self, taskFunc)
            self.tRun = threading.Thread(target=taskThread, args=args)
            self.tRun.start()

        except Exception:
            print(traceback.format_exc())

    def PB_operation_click(self):
        self.runTask(self.operationTask)

    def operationTask(self):
        hardwareDict = {
            'chipNum': int(self.lineEdit_chipNum.text()),
            'mode':
            self.comboBox_mode.currentIndex() + reram_register.FORM_MODE,
            'TIAFeedback': self.comboBox_TIAFeedback.currentIndex(),
            'BLStart': int(self.lineEdit_BLStart.text()),
            'BLCnt': int(self.lineEdit_BLCnt.text()),
            'SLStart': int(self.lineEdit_SLStart.text()),
            'SLCnt': int(self.lineEdit_SLCnt.text()),
            'WLStart': int(self.lineEdit_WLStart.text()),
            'WLCnt': int(self.lineEdit_WLCnt.text()),
            'BLInputActive': self.comboBox_BLInputActive.currentIndex(),
            'SLInputActive': self.comboBox_SLInputActive.currentIndex(),
            'WLInputActive': self.comboBox_WLInputActive.currentIndex(),
            'BLVActive': float(self.lineEdit_BLVActive.text()),
            'SLVActive': float(self.lineEdit_SLVActive.text()),
            'WLVActive': float(self.lineEdit_WLVActive.text()),
            'BLInputInactive': self.comboBox_BLInputInactive.currentIndex(),
            'SLInputInactive': self.comboBox_SLInputInactive.currentIndex(),
            'WLInputInactive': self.comboBox_WLInputInactive.currentIndex(),
            'BLVInactive': float(self.lineEdit_BLVInactive.text()),
            'SLVInactive': float(self.lineEdit_SLVInactive.text()),
            'WLVInactive': float(self.lineEdit_WLVInactive.text()),
            'TIAVoltage': 0,
            'pulseWidth': int(self.lineEdit_pulseWidth.text()),
            'pulseGap': int(self.lineEdit_pulseGap.text()),
            'pulseCnt': int(self.lineEdit_pulseCnt.text()),
            'readDirection': self.comboBox_readDirection.currentIndex()
        }

        self.app.configParas(**hardwareDict)
        self.app._regTableCfg()
        ret = self.app.cmdCfgReg()
        if not ret[0]:
            print('Operation Fail')
            return

        if self.app.mode == reram_register.READ_MODE:
            regData = gf.bytesToUint32Array(ret[1], 'little')
            current = []
            for i in regData:
                current.append(self.app.regToCurrent(i & 0xFFFF))
                current.append(self.app.regToCurrent(i >> 16))

            if self.app.readDirection == apps.POSITIVE_READ:
                current = gf.SpecifySort(current, apps.TIA_SL_Table)
            else:
                current = gf.SpecifySort(current, apps.TIA_BL_Table)
            print(current)
