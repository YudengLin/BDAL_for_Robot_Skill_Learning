import os
import threading
import traceback
import importlib
import sys

from PyQt5.QtWidgets import QWidget, QFileDialog
from PyQt5.uic import loadUi

import globals.global_func as gf

FILE_PATH, FILE_FULL_NAME = os.path.split(os.path.realpath(__file__))
FILE_NAME, FILE_EXT = os.path.splitext(FILE_FULL_NAME)


def writeThread(obj):
    try:
        print('Start operation')
        obj.app.bBusy = True

        addr = [int(obj.LE_writeAddr.text(), 16)]
        data = [int(obj.LE_writeData.text(), 16)]
        ret = obj.app.cmdWriteReg(addr, data)
        if ret[0]:
            print('Write Reg Success')
        else:
            print('Write Reg Fail')

    except Exception:
        print(traceback.format_exc())
    finally:
        obj.app.bBusy = False
        print('Stop operation')


def readThread(obj):
    try:
        print('Start operation')
        obj.app.bBusy = True

        addr = int(obj.LE_readAddr.text(), 16)
        count = int(obj.LE_readCount.text(), 16)
        ret = obj.app.cmdSeqReadReg(addr, count)
        if ret[0]:
            gf.printBytesHex(ret[1], dataType='uint32', byteorder='little')
        else:
            print('Read Reg Fail')

    except Exception:
        print(traceback.format_exc())
    finally:
        obj.app.bBusy = False
        print('Stop operation')


def taskRunThread(obj):
    try:
        print('Start operation')
        obj.app.bBusy = True

        taskpath, taskFullName = os.path.split(obj.LE_taskFile.text())
        taskName, _ = os.path.splitext(taskFullName)
        if taskpath not in sys.path:
            sys.path.append(taskpath)
        taskLab = importlib.import_module(taskName)
        taskLab = importlib.reload(taskLab)
        taskLab.userTask(obj.app)

    except Exception:
        print(traceback.format_exc())
    finally:
        obj.app.bBusy = False
        print('Stop operation')


class CommonCmd(QWidget):
    """Base operation"""
    def __init__(self, app):
        super(CommonCmd, self).__init__()
        self.app = app
        loadUi(FILE_PATH + '/' + FILE_NAME + '.ui', self)
        self.PB_write.clicked.connect(self.PB_write_click)
        self.PB_read.clicked.connect(self.PB_read_click)
        self.PB_taskFile.clicked.connect(self.taskFile_click)
        taskPath, _ = os.path.split(os.path.realpath(FILE_PATH))
        self.LE_taskFile.setText(taskPath + r"\task.py")
        self.PB_taskRun.clicked.connect(self.taskRun_click)

    def PB_write_click(self):
        try:
            if not self.app.checkDevice():
                return

            args = (self, )
            self.tWrite = threading.Thread(target=writeThread, args=args)
            self.tWrite.start()

        except Exception:
            print(traceback.format_exc())

    def PB_read_click(self):
        try:
            if not self.app.checkDevice():
                return

            args = (self, )
            self.tRead = threading.Thread(target=readThread, args=args)
            self.tRead.start()

        except Exception:
            print(traceback.format_exc())

    def taskFile_click(self):
        fileName = QFileDialog.getOpenFileName(self, 'Select File', '',
                                               '*.py')[0]

        if fileName:
            self.LE_taskFile.setText(fileName)

    def taskRun_click(self):
        try:
            if not self.app.checkDevice():
                return

            args = (self, )
            self.tTaskRun = threading.Thread(target=taskRunThread, args=args)
            self.tTaskRun.start()

        except Exception:
            print(traceback.format_exc())
