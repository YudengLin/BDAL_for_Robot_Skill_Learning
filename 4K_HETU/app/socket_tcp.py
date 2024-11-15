import socket

import globals.global_func as gf


def to_bytes(seq):
    """convert a sequence to a bytes type"""
    if isinstance(seq, bytes):
        return seq
    elif isinstance(seq, bytearray):
        return bytes(seq)
    else:
        return bytes(bytearray(seq))


class SocketException(OSError):
    """Base class for socket related exceptions."""


portNotConnectError = SocketException(
    'Attempting to use a socket that is not connect')


class SocketTCP():
    def __init__(self):
        self.sock = None
        self.isConnect = False

    # def __del__(self):
    #     self.close()

    def close(self):
        if self.sock is None:
            return

        self.sock.close()
        self.sock = None
        self.isConnect = False

    def _create(self):
        self.close()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.isConnect = False
        return self.sock

    def connect(self, ip, port, timeoutS):
        if self.sock is None:
            self._create()

        self.sock.setblocking(False)
        self.sock.settimeout(timeoutS)

        try:
            self.sock.connect((ip, port))
        except Exception:
            self.close()
            return False

        self.sock.setblocking(True)
        self.isConnect = True
        return True

    def send(self, data, bLog=False):
        if not self.isConnect:
            raise portNotConnectError

        data = to_bytes(data)
        if data:
            if bLog:
                print('Start-Print the sent data:')
                gf.printBytesHex(data, dataType='uint8')
                print('End-Print the sent data:')
            return self.sock.send(data)
        else:
            return 0

    def recv(self, bufsize, bLog=False, flags=socket.MSG_WAITALL):
        if not self.isConnect:
            raise portNotConnectError

        data = self.sock.recv(bufsize, flags)
        if bLog:
            print('Start-Print the received data:')
            gf.printBytesHex(data, dataType='uint8')
            print('End-Print the received data:')
        return data

    def isConnected(self):
        return self.isConnect
