import os
import sys

from PyQt5 import QtCore, QtGui, QtWidgets

import cell

FILE_PATH = os.path.split(os.path.realpath(__file__))[0]


class CrossbarCells(QtWidgets.QWidget):
    def __init__(self, colorbarRuler, WLCount=32, BLCount=128):
        super(CrossbarCells, self).__init__()
        self.__BLCount = BLCount
        self.__WLCount = WLCount
        self.__colorbarRuler = colorbarRuler
        self.initUI()

    @property
    def BLCount(self):
        return self.__BLCount

    @property
    def WLCount(self):
        return self.__WLCount

    def initUI(self):
        layout = QtWidgets.QGridLayout(self)
        self.setLayout(layout)
        layout.setSpacing(0)

        self.cells = [[[] for x in range(0, self.BLCount)]
                      for y in range(0, self.WLCount)]

        for r in range(0, self.WLCount):
            for c in range(0, self.BLCount):
                self.cells[r][c] = cell.Cell(r, c, self.__colorbarRuler)
                self.cells[r][c].setMinimumWidth(10)
                self.cells[r][c].setMinimumHeight(10)
                layout.addWidget(self.cells[r][c], r, c)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showContextMenu)
        self.contextMenu = QtWidgets.QMenu(self)
        self.saveAction = QtWidgets.QAction(
            QtGui.QIcon(FILE_PATH + '/graphics/save.png'), 'Save', self)
        # self.saveAction.setShortcut('Ctrl+S')
        self.saveAction.setStatusTip('Save')
        self.saveAction.triggered.connect(self.saveAct)
        self.contextMenu.addAction(self.saveAction)

    def setCellColorValue(self, value, w, b):
        self.cells[w][b].colorValue = value

    def readCellColorValue(self, w, b):
        return self.cells[w][b].colorValue

    def clearCellColor(self, w, b):
        self.cells[w][b].clearColor()

    def setAllColorValue(self, value):
        for w in range(self.WLCount):
            for b in range(self.BLCount):
                self.setCellColorValue(value, w, b)

    def clearAllColor(self):
        for w in range(self.WLCount):
            for b in range(self.BLCount):
                self.clearCellColor(w, b)

    def showContextMenu(self, pos):
        '''
        右键点击时调用的函数
        '''
        self.contextMenu.exec_(QtGui.QCursor.pos())  # 在鼠标位置显示

    def saveAct(self):
        fileName, fileType = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save File', '', '*.csv')
        if fileName:
            if fileType == '*.csv':
                with open(fileName, "w") as f:
                    data = ''
                    for w in range(self.WLCount):
                        if w != 0:
                            data = data + '\n'
                        for b in range(self.BLCount):
                            if self.cells[w][b].colorValue is None:
                                colorValueStr = 'None'
                            else:
                                colorValueStr = "{:.2f}".format(
                                    self.cells[w][b].colorValue)
                            data = data + colorValueStr + ','
                    f.write(data)
            else:
                pass


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    import colorbar_ruler
    ruler = colorbar_ruler.ColorbarRuler()
    mainForm = CrossbarCells(ruler)
    mainForm.setAllColorValue(0.2)
    mainForm.setCellColorValue(0.1, 0, 0)
    mainForm.setCellColorValue(0.3, 1, 1)
    # mainForm.setCellColorValue(20, 2, 4)
    # mainForm.clearCellColor(1,1)
    # mainForm.clearAllColor()
    # print(mainForm.readCellColorValue(2, 5))
    mainForm.show()
    sys.exit(apps.exec_())
