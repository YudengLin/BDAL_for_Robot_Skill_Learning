import sys

from PyQt5 import QtCore, QtGui, QtWidgets

import globals.global_font as gFont
import globals.global_style as gs


class Cell(QtWidgets.QWidget):
    def __init__(self, r, c, colorbarRuler):
        super(Cell, self).__init__()
        self.r = r
        self.c = c
        self.__colorValue = None
        self.__colorbarRuler = colorbarRuler
        self.invalidColor = QtGui.QColor(125, 125, 125)
        self.initUI()

    @property
    def colorValue(self):
        return self.__colorValue

    @colorValue.setter
    def colorValue(self, value):
        self.__colorValue = value
        self._recolor()

    def initUI(self):
        self.setStyleSheet("padding-right: 1px; padding-bottom: 1px")
        self.pen = QtGui.QPen(QtGui.QColor(200, 200, 200))
        self.brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        # self.setMaximumWidth(200)
        # self.setMaximumHeight(200)

        # SETUP HOVER PANEL
        #######################
        hoverLayout = QtWidgets.QVBoxLayout()
        self.posLabel = QtWidgets.QLabel()
        self.posLabel.setStyleSheet(gs.labelStyle)
        self.mLabel = QtWidgets.QLabel()
        self.mLabel.setFont(gFont.font1)
        self.mLabel.setStyleSheet(gs.labelStyle)
        hoverLayout.addWidget(self.posLabel)
        hoverLayout.addWidget(self.mLabel)
        hoverLayout.setContentsMargins(2, 2, 2, 2)
        hoverLayout.setSpacing(0)
        self.hoverPanel = QtWidgets.QWidget()
        self.hoverPanel.setWindowFlags(QtCore.Qt.FramelessWindowHint
                                       | QtCore.Qt.Tool)
        self.hoverPanel.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.hoverPanel.setFixedSize(83, 40)
        self.hoverPanel.setStyleSheet("background-color: rgb(0,32,87)")
        self.hoverPanel.setLayout(hoverLayout)
        self.hoverPanel.hide()

    def paintEvent(self, e):  # this is called whenever the Widget is resized
        qp = QtGui.QPainter()  # initialise the Painter Object
        qp.begin(self)  # Begin the painting process
        self.drawRectangle(qp)  # Call the function
        qp.end()  # End the painting process

    def drawRectangle(self, qp):
        # get the size of this Widget (which by default fills the parent
        size = self.size()
        qp.setPen(self.pen)  # set the pen
        qp.setBrush(self.brush)  # set the brush
        qp.drawRect(0, 0, size.width(), size.height())

    # def highlight(self):
    #     self.pen.setColor(QtGui.QColor(0, 0, 0))
    #     self.pen.setWidth(4)
    #     self.update()

    # def dehighlight(self):
    #     self.pen.setColor(QtGui.QColor(200, 200, 200))
    #     self.pen.setWidth(1)
    #     self.update()

    def _recolor(self):
        color = self.__colorbarRuler.getColor(self.colorValue)
        if color is None:
            color = self.invalidColor
            print('R=%d | C=%d, The colorValue(%d) is invalid' %
                  (self.r, self.c, self.colorValue))

        self.brush.setColor(color)
        self.update()

    def enterEvent(self, event):
        self.posLabel.setText("R=" + str(self.r) + " | C=" + str(self.c))

        if self.colorValue is None:
            self.mLabel.setText("None")
        else:
            self.mLabel.setText(str("{:.2f}".format(self.colorValue)))

        newX = event.globalX() + self.width()
        newY = event.globalY() + self.height()

        self.hoverPanel.move(newX, newY)
        self.hoverPanel.show()

    def leaveEvent(self, event):
        self.hoverPanel.hide()

    def clearColor(self):
        self.__colorValue = None
        self.pen = QtGui.QPen(QtGui.QColor(200, 200, 200))
        self.brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        self.update()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    import colorbar_ruler
    ruler = colorbar_ruler.ColorbarRuler()
    mainForm = Cell(3, 4, ruler)
    # mainForm.colorValue = 256
    mainForm.show()
    sys.exit(apps.exec_())
