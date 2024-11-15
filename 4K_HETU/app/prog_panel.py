import importlib
import os
import sys

from PyQt5 import QtWidgets

import globals.global_style as gs

FILE_PATH, FILE_FULL_NAME = os.path.split(os.path.realpath(__file__))
FILE_NAME, FILE_EXT = os.path.splitext(FILE_FULL_NAME)
sys.path.append(FILE_PATH + r'\panels')


class ProgPanel(QtWidgets.QWidget):
    def __init__(self, app):
        super(ProgPanel, self).__init__()
        self.app = app
        self.initUI()

    def initUI(self):
        # self.resize(800, 600)
        mainLayout = QtWidgets.QVBoxLayout()

        hbox_1 = QtWidgets.QHBoxLayout()

        label_panels = QtWidgets.QLabel('Panels:')
        label_panels.setMaximumWidth(40)

        self.comboBox_panels = QtWidgets.QComboBox()
        self.comboBox_panels.setStyleSheet(gs.comboStyle)

        files = [
            f for f in os.listdir(FILE_PATH + '/panels/') if f.endswith(".py")
        ]  # populate prog panel dropbox
        for f in files:
            f = self.toPanelName(f[:-3])
            self.comboBox_panels.addItem(f)

        self.pushBtn_add = QtWidgets.QPushButton('Add')
        self.pushBtn_add.setStyleSheet(gs.btnStyle)
        self.pushBtn_add.clicked.connect(self.pushBtn_add_click)

        self.pushBtn_remove = QtWidgets.QPushButton('Remove')
        self.pushBtn_remove.setStyleSheet(gs.btnStyle)
        self.pushBtn_remove.clicked.connect(self.pushBtn_remove_click)

        self.tabFrame = QtWidgets.QTabWidget()
        self.tabFrame.setMinimumSize(3000, 3000)

        panelName = 'Pulse Operation'
        moduleName = self.toModuleName(panelName)
        thisPanel = importlib.import_module(moduleName)
        panel_class = getattr(thisPanel, panelName.replace(' ', ''))
        widg = panel_class(self.app)
        self.tabFrame.addTab(widg, panelName)
        self.tabFrame.setCurrentWidget(widg)

        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidget(self.tabFrame)

        hbox_1.addWidget(label_panels)
        hbox_1.addWidget(self.comboBox_panels)
        hbox_1.addWidget(self.pushBtn_add)
        hbox_1.addWidget(self.pushBtn_remove)

        mainLayout.addLayout(hbox_1)
        mainLayout.addWidget(self.scroll)
        mainLayout.setContentsMargins(0, 0, 0, 0)
        self.setContentsMargins(0, 0, 0, 0)
        self.setLayout(mainLayout)

    def toPanelName(self, fileName):
        nameList = fileName.split('_')
        PanelName = ''
        for i in nameList:
            PanelName += i.capitalize()
            PanelName = PanelName + ' '
        return PanelName[:-1]

    def toModuleName(self, fileName):
        fileName = fileName.lower()
        fileName = fileName.replace(' ', '_')
        return fileName

    def pushBtn_add_click(self):
        panelName = self.comboBox_panels.currentText()
        moduleName = self.toModuleName(
            panelName)  # format module name from drop down
        thisPanel = importlib.import_module(moduleName)  # import the module
        panel_class = getattr(thisPanel,
                              panelName.replace(' ',
                                                ''))  # get it's main class
        widg = panel_class(self.app)
        self.tabFrame.addTab(widg,
                             panelName)  # instantiate it and add to tabWidget
        self.tabFrame.setCurrentWidget(widg)

    def pushBtn_remove_click(self):
        self.tabFrame.removeTab(self.tabFrame.currentIndex())
