import os
import sys

from PyQt5 import QtCore, QtGui, QtWidgets

import about
import apps
import globals.global_style as gStyle
import prog_panel

FILE_PATH = os.path.split(os.path.realpath(__file__))[0]


class EmittingStream(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))


class MainForm(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainForm, self).__init__()
        self.app = apps.App()
        self.initUI()

    def initUI(self):
        ##########################
        # SPLASH SCREEN #
        pixmap = QtGui.QPixmap(FILE_PATH + '/graphics/logo.png')
        splashScreen = QtWidgets.QSplashScreen(pixmap)
        splashScreen.show()
        ##########################

        splashScreen.showMessage("Starting up...",
                                 alignment=QtCore.Qt.AlignBottom,
                                 color=QtCore.Qt.black)

        self.resize(1000, 800)
        self.setWindowTitle("APP")
        self.setWindowIcon(QtGui.QIcon(FILE_PATH + '/graphics/icon.jpg'))

        ##########################
        # Setup menubar
        menuBar = self.menuBar()

        fileMenu = menuBar.addMenu('File')  # File menu
        # settingsMenu = menuBar.addMenu('Settings')	# Setting menu
        helpMenu = menuBar.addMenu('Help')  # help menu

        # Help menu
        documentationAction = QtWidgets.QAction('Documentation', self)
        documentationAction.setStatusTip('Show documentation')
        documentationAction.triggered.connect(self.showDocumentation)

        aboutAction = QtWidgets.QAction('About', self)
        aboutAction.setStatusTip('Information about APP')
        aboutAction.triggered.connect(self.showAbout)

        # Populate help menu
        helpMenu.addAction(documentationAction)
        helpMenu.addSeparator()
        helpMenu.addAction(aboutAction)

        # Setup toolbar
        self.toolbar = self.addToolBar('Toolbar')

        self.pushBtn_connect = QtWidgets.QAction(
            QtGui.QIcon(FILE_PATH + '/graphics/connect.png'), 'Connect', self)
        # self.pushBtn_connect.setShortcut('Ctrl+N')
        self.pushBtn_connect.setStatusTip('Connect to device')
        self.pushBtn_connect.triggered.connect(self.pushBtn_connect_click)

        self.pushBtn_disconnect = QtWidgets.QAction(
            QtGui.QIcon(FILE_PATH + '/graphics/disconnect.png'), 'Disconnect',
            self)
        # self.pushBtn_connect.setShortcut('Ctrl+N')
        self.pushBtn_disconnect.setStatusTip('Disconnect device')
        self.pushBtn_disconnect.triggered.connect(
            self.pushBtn_disconnect_click)

        self.clearAction = QtWidgets.QAction(
            QtGui.QIcon(FILE_PATH + '/graphics/clear.png'), 'Clear', self)
        # self.clearAction.setShortcut('Ctrl+D')
        self.clearAction.setStatusTip('Clear log')
        self.clearAction.triggered.connect(self.clearAct)

        self.saveAction = QtWidgets.QAction(
            QtGui.QIcon(FILE_PATH + '/graphics/save.png'), 'Save log', self)
        # self.saveAction.setShortcut('Ctrl+D')
        self.saveAction.setStatusTip('Save log')
        self.saveAction.triggered.connect(self.saveAct)

        self.toolbar.addAction(self.pushBtn_connect)
        self.toolbar.addAction(self.pushBtn_disconnect)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.clearAction)
        self.toolbar.addAction(self.saveAction)

        splitterMain = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitterMain.setHandleWidth(5)
        splitterMain.setContentsMargins(0, 5, 0, 0)
        leftSplitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        leftSplitter.setHandleWidth(5)
        leftSplitter.setContentsMargins(0, 0, 0, 0)

        rightSplitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        rightSplitter.setHandleWidth(5)
        rightSplitter.setContentsMargins(0, 0, 0, 0)

        self.programPanel = prog_panel.ProgPanel(self.app)
        self.logShow = QtWidgets.QTextEdit()

        leftSplitter.addWidget(self.programPanel)
        rightSplitter.addWidget(self.logShow)

        splitterMain.addWidget(leftSplitter)
        splitterMain.addWidget(rightSplitter)
        splitterMain.setStretchFactor(0, 1)
        splitterMain.setStretchFactor(1, 1)
        self.setCentralWidget(splitterMain)
        self.setContentsMargins(5, 0, 5, 0)
        statusShow = self.statusBar()
        self.label_info = QtWidgets.QLabel('Disconnected')
        self.label_info.setStyleSheet(gStyle.labelDisconnected)
        statusShow.addPermanentWidget(self.label_info)

        fileMenu.addAction(self.pushBtn_connect)
        fileMenu.addAction(self.pushBtn_disconnect)
        fileMenu.addSeparator()
        fileMenu.addAction(self.clearAction)
        fileMenu.addAction(self.saveAction)
        # 重定向输出
        sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)
        sys.stderr = EmittingStream(textWritten=self.normalOutputWritten)
        splashScreen.finish(self)

    def normalOutputWritten(self, text):
        cursor = self.logShow.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.logShow.setTextCursor(cursor)
        self.logShow.ensureCursorVisible()

    def clearAct(self):
        self.logShow.clear()

    def pushBtn_connect_click(self):
        self.app.client.connect('101.6.93.189', 7, 2, self.onConnect)
        self.pushBtn_connect.setEnabled(False)

    def onConnect(self, bSuccess):
        if bSuccess:
            print('TCP PORT-7: Connect success')
            self.app.client.startListener(onDisconnectFunc=self.onDisconnect)
            self.pushBtn_connect.setEnabled(False)
            self.label_info.setText('Connected')
            self.label_info.setStyleSheet(gStyle.labelConnected)
        else:
            print('TCP PORT-7: Connect fail')
            self.pushBtn_connect.setEnabled(True)

    def onDisconnect(self):
        self.app.client.close()
        self.pushBtn_connect.setEnabled(True)
        self.label_info.setText('Disconnected')
        self.label_info.setStyleSheet(gStyle.labelDisconnected)
        print('TCP PORT-7: Client disconnected')

    def pushBtn_disconnect_click(self):
        self.app.client.close()

    def closeEvent(self, event):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        self.app.client.close()

    def showDocumentation(self):
        doc = FILE_PATH + '/doc/' + 'app.txt'
        os.system(doc)

    def showAbout(self):
        self.abt = about.About()
        self.abt.show()

    def saveAct(self):
        fileName, fileType = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save File', '', '*.txt')
        if fileName:
            if fileType == '*.txt':
                with open(fileName, "w") as f:
                    data = self.logShow.toPlainText()
                    f.write(data)
            else:
                pass


if __name__ == '__main__':
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    myApp = QtWidgets.QApplication(sys.argv)
    mainForm = MainForm()
    mainForm.show()
    sys.exit(myApp.exec_())
