import os

from PyQt5.QtWidgets import QWidget
from PyQt5.uic import loadUi

FILE_PATH, FILE_FULL_NAME = os.path.split(os.path.realpath(__file__))
FILE_NAME, FILE_EXT = os.path.splitext(FILE_FULL_NAME)


class About(QWidget):
    """Base operation"""
    def __init__(self):
        super(About, self).__init__()
        loadUi(FILE_PATH + '/' + FILE_NAME + '.ui', self)
        self.pushButton_close.clicked.connect(self.pushBtn_close_click)

    def pushBtn_close_click(self):
        self.close()
