import os
import threading
import traceback
import time

import numpy as np
from PyQt5.QtWidgets import QWidget, QFileDialog
from PyQt5.uic import loadUi

import crossbar_panel
import apps
# import globals.global_func as gf

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


def mapAbortTask():
    try:
        myapp = apps.App()
        myapp.client.connect('101.6.93.189', 5000, 2)

        for i in range(4):
            if i == 3:
                return
            if not myapp.client.isConnected():
                time.sleep(1)
                print('Connecting to device')
            else:
                break

        myapp.client.startListener()
        ret = myapp.cmdMapAbort()
        if ret[0]:
            print('Success')
        else:
            print('Fail')
    finally:
        myapp.client.close()


class PulseOperation(QWidget):
    """Base operation"""
    def __init__(self, app):
        super(PulseOperation, self).__init__()
        loadUi(FILE_PATH + '/' + FILE_NAME + '.ui', self)
        self.app = app
        self.PB_read.clicked.connect(self.PB_read_click)
        self.crossbarPanel = crossbar_panel.CrossbarPanel()
        # self.crossbarPanel = None
        self.PB_targetFile.clicked.connect(self.PB_targetFile_click)
        self.LE_targetFile.setText(FILE_PATH + r"\target_current.csv")
        self.PB_mapFlow.clicked.connect(self.PB_mapFlow_click)
        self.PB_mapAbort.clicked.connect(self.PB_mapAbort_click)
        # self.tInit = threading.Thread(target=self.initThread)
        # self.tInit.start()

    # def initThread(self):
    #     self.crossbarPanel = crossbar_panel.CrossbarPanel()
    #     self.update()

    def runTask(self, taskFunc):
        try:
            if not self.app.checkDevice():
                return

            args = (self, taskFunc)
            self.tRun = threading.Thread(target=taskThread, args=args)
            self.tRun.start()

        except Exception:
            print(traceback.format_exc())

    def PB_read_click(self):
        tBLCnt = int(self.lineEdit_BLCnt.text())
        tWLCnt = int(self.lineEdit_WLCnt.text())
        if tBLCnt == 1 and tWLCnt == 1:
            self.runTask(self.readTask)
        else:
            self.crossbarPanel.cells.clearAllColor()
            self.crossbarPanel.show()
            self.crossbarPanel.center()
            self.runTask(self.readTask)

    def readTask(self):
        tBLStart = int(self.lineEdit_BLStart.text())
        tBLCnt = int(self.lineEdit_BLCnt.text())
        tWLStart = int(self.lineEdit_WLStart.text())
        tWLCnt = int(self.lineEdit_WLCnt.text())
        addrInfo = apps.AddrInfo()
        addrInfo.chipNum = int(self.lineEdit_chipNum.text())
        addrInfo.BLCnt = 1
        addrInfo.SLCnt = 1
        addrInfo.WLCnt = 1
        readV = float(self.lineEdit_VRead.text())
        readWLV = float(self.lineEdit_readVWL.text())
        readDirection = self.CB_readDirection.currentIndex()

        for bl in range(tBLStart, tBLStart + tBLCnt):
            for wl in range(tWLStart, tWLStart + tWLCnt):
                addrInfo.BLStart = bl
                addrInfo.SLStart = wl
                addrInfo.WLStart = wl
                ret = self.app.readOperation(addrInfo, readV, readWLV,
                                             readDirection)
                if not ret[0]:
                    return
                currentList = ret[1]
                if readDirection == apps.POSITIVE_READ:
                    if tBLCnt == 1 and tWLCnt == 1:
                        print('BL = %d, WL = %d, current = %.3fuA' %
                              (bl, wl, currentList[wl]))
                    else:
                        self.crossbarPanel.cells.setCellColorValue(
                            currentList[wl], wl, bl)
                else:
                    if tBLCnt == 1 and tWLCnt == 1:
                        print('BL = %d, WL = %d, current = %.3fuA' %
                              (bl, wl, currentList[bl]))
                    else:
                        self.crossbarPanel.cells.setCellColorValue(
                            currentList[bl], wl, bl)

    def PB_targetFile_click(self):
        fileName = QFileDialog.getOpenFileName(self, 'Select File', '',
                                               '*.csv')[0]

        if fileName:
            self.LE_targetFile.setText(fileName)

    def PB_mapFlow_click(self):
        self.runTask(self.mapFlowTask)

    def mapFlowTask(self):
        finderCfg = apps.MapFinderCfg()
        finderCfg.formVWLStart = float(self.formVWLStart.text())
        finderCfg.formVWLStep = float(self.formVWLStep.text())
        finderCfg.formVWLEnd = float(self.formVWLEnd.text())
        finderCfg.formVBLStart = float(self.formVBLStart.text())
        finderCfg.formVBLStep = float(self.formVBLStep.text())
        finderCfg.formVBLEnd = float(self.formVBLEnd.text())
        finderCfg.setVWLStart = float(self.setVWLStart.text())
        finderCfg.setVWLStep = float(self.setVWLStep.text())
        finderCfg.setVWLEnd = float(self.setVWLEnd.text())
        finderCfg.setVBLStart = float(self.setVBLStart.text())
        finderCfg.setVBLStep = float(self.setVBLStep.text())
        finderCfg.setVBLEnd = float(self.setVBLEnd.text())
        finderCfg.rstVWLStart = float(self.resetVWLStart.text())
        finderCfg.rstVWLStep = float(self.resetVWLStep.text())
        finderCfg.rstVWLEnd = float(self.resetVWLEnd.text())
        finderCfg.rstVSLStart = float(self.resetVSLStart.text())
        finderCfg.rstVSLStep = float(self.resetVSLStep.text())
        finderCfg.rstVSLEnd = float(self.resetVSLEnd.text())
        finderCfg.errorLimit = float(self.LE_error.text())
        finderCfg.nMax = 200
        finderCfg.readDirection = self.CB_readDirection.currentIndex()
        addrInfo = apps.AddrInfo()
        addrInfo.chipNum = int(self.lineEdit_chipNum.text())
        addrInfo.BLCnt = int(self.lineEdit_BLCnt.text())
        addrInfo.SLCnt = int(self.lineEdit_WLCnt.text())
        addrInfo.WLCnt = int(self.lineEdit_WLCnt.text())
        addrInfo.BLStart = int(self.lineEdit_BLStart.text())
        addrInfo.SLStart = int(self.lineEdit_WLStart.text())
        addrInfo.WLStart = int(self.lineEdit_WLStart.text())
        targetList = list(
            np.loadtxt(self.LE_targetFile.text(),
                       dtype=float,
                       delimiter=',',
                       skiprows=0,
                       usecols=0))
        if len(targetList) >= addrInfo.BLCnt * addrInfo.SLCnt:
            targetList = targetList[0:addrInfo.BLCnt * addrInfo.SLCnt]
            ret = self.app.mapFlow(finderCfg, addrInfo, targetList)
            if ret[0]:
                print('Map Success')
            else:
                print('Map Fail')
        else:
            print('The target file length is not sufficient')

    def PB_mapAbort_click(self):
        tMapAbort = threading.Thread(target=mapAbortTask)
        tMapAbort.start()
