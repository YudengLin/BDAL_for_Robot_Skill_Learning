toolBtn = """
QPushButton {
    background-color: rgb(255,255,255);
    border:1px solid rgb(0,32,87);
    border-radius: 4px;
    color: rgb(0,32,87);
    height: 23px;
    margin-left: 2px;
    margin-right: 2px;
    padding: 0 5px 0 5px;
    font: bold;
    }
QPushButton:hover{
    background-color: rgb(0,32,87);
    color: rgb(255, 255, 255);
    }
QPushButton:pressed{
    margin: 3px;
    }"""

btnStyle = """
QPushButton {
    background-color: rgb(255,255,255);
    border:1px solid rgb(0, 32, 87);
    border-radius: 4px;
    color: rgb(0, 32, 87);
    height: 23px;
    margin: 0px;
    font: bold;
    }
QPushButton:hover{
    background-color: rgb(0, 32, 87);
    color: rgb(255, 255, 255);
    }
QPushButton:pressed{
    margin: 3px;
    }"""

comboStyle = """
QComboBox{
    border: 1px solid rgb(0,32,87);
    border-radius: 0px;
    padding: 0 0 0 2px;
    height: 22px;
    }
QComboBox:down-button {
    background-color: rgb(255, 255, 255);
    }"""

labelDisconnected = """
QLabel{
    background-color: red;
    color: rgb(255,255,255);
    font-size:16px;
    }"""

labelConnected = """
QLabel{
    background-color: green;
    color: rgb(255,255,255);
    font-size:16px;
    }"""

labelStyle = """
QLabel{
    background-color: rgb(255,255,255);
    color: rgb(0, 32,87);
    padding-left: 2px;
    }"""
