import matplotlib.cm as cm
from PyQt5 import QtGui, QtWidgets

import globals.global_font as gFont
from colorbar_slice import ColorbarSlice


class ColorbarRuler(QtWidgets.QWidget):
    def __init__(self, minScale=0, maxScale=6):
        if minScale > maxScale:
            raise ValueError('The minScale is greater than the maxScale')
        super(ColorbarRuler, self).__init__()
        self.__minScale = minScale
        self.__maxScale = maxScale
        self.colorList = []
        self.initUI()

    @property
    def minScale(self):
        return self.__minScale

    @minScale.setter
    def minScale(self, value):
        self.__minScale = value
        self.resTicksLabels[2].setText(str(self.minScale))
        self.resTicksLabels[1].setText(str(
            (self.maxScale + self.minScale) / 2))

    @property
    def maxScale(self):
        return self.__maxScale

    @maxScale.setter
    def maxScale(self, value):
        self.__maxScale = value
        self.resTicksLabels[0].setText(str(self.maxScale))
        self.resTicksLabels[1].setText(str(
            (self.maxScale + self.minScale) / 2))

    def initUI(self):
        # colorTuple = cm.rainbow
        colorTuple = cm.plasma
        for i in range(colorTuple.N):
            aux_color = QtGui.QColor()
            aux_color.setRgbF(
                colorTuple(i)[0],
                colorTuple(i)[1],
                colorTuple(i)[2],
                colorTuple(i)[3])
            self.colorList.append(aux_color)

        self.colorList = self.colorList[::-1]  # revert the list

        # Colorbar setup
        colorBarLay = QtWidgets.QHBoxLayout()
        colorBarLeft = QtWidgets.QVBoxLayout()
        for i in range(len(self.colorList)):
            aux = ColorbarSlice()
            aux.recolor(self.colorList[255 - i])
            aux.setMinimumWidth(20)
            aux.setMaximumWidth(20)
            colorBarLeft.addWidget(aux)

        # Create Tick labels
        resTicks = [
            str(self.maxScale),
            str((self.maxScale + self.minScale) / 2),
            str(self.minScale)
        ]
        self.resTicksLabels = []
        for i in range(len(resTicks)):
            aux = QtWidgets.QLabel(self)
            aux.setText(resTicks[i])
            aux.setFont(gFont.font1)
            self.resTicksLabels.append(aux)

        # Add ticks
        colorBarRight = QtWidgets.QVBoxLayout()
        colorBarRight.addWidget(self.resTicksLabels[0])
        colorBarRight.addStretch()
        colorBarRight.addWidget(self.resTicksLabels[1])
        colorBarRight.addStretch()
        colorBarRight.addWidget(self.resTicksLabels[2])

        colorBarLeft.setSpacing(0)
        colorBarLay.addLayout(colorBarLeft)
        colorBarLay.addLayout(colorBarRight)
        colorBarLay.setContentsMargins(0, 12, 10, 21)

        self.setLayout(colorBarLay)
        self.setContentsMargins(0, 0, 0, 0)

    def getColor(self, value):
        try:
            idx = int(
                round((value - self.minScale) * 255 /
                      (self.maxScale - self.minScale)))
            color = self.colorList[idx]
        except (OverflowError, ValueError, IndexError):
            color = 255
        finally:
            return color

    def changeScope(self, minScale, maxScale):
        self.minScale = minScale
        self.maxScale = maxScale


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainForm = ColorbarRuler()
    mainForm.changeScope(50, 100)
    mainForm.show()
    sys.exit(apps.exec_())
