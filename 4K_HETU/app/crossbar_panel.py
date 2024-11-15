import sys

from PyQt5 import QtWidgets
from PyQt5 import QtCore

import colorbar_ruler
import crossbar_cells


class CrossbarPanel(QtWidgets.QWidget):
    def __init__(self):
        super(CrossbarPanel, self).__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Crossbar")
        mainLayout = QtWidgets.QHBoxLayout()
        mainLayout.setSpacing(0)
        mainLayout.setContentsMargins(0, 0, 0, 0)

        self.ruler = colorbar_ruler.ColorbarRuler()
        self.cells = crossbar_cells.CrossbarCells(self.ruler)
        self.cells.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                 QtWidgets.QSizePolicy.Expanding)

        mainLayout.addWidget(self.cells)
        mainLayout.addWidget(self.ruler)
        self.setLayout(mainLayout)
        # self.setContentsMargins(100, 0, 0, 100)

    def changeEvent(self, event):
        # 顶层窗口激活状态改变
        if event.type() == QtCore.QEvent.ActivationChange:
            self.repaint()

    def center(self):
        frameGm = self.frameGeometry()
        centerPoint = QtWidgets.QDesktopWidget().availableGeometry().center()
        frameGm.moveCenter(centerPoint)
        self.move(frameGm.topLeft())


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainForm = CrossbarPanel()
    mainForm.cells.setCellColorValue(0.5, 2, 2)
    mainForm.show()
    sys.exit(app.exec_())
