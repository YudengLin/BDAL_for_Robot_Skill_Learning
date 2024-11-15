import socket
import threading

import socket_tcp


def listenerThread(client, onRecvFunc, onDisconnectFunc):
    if not client.isConnect:
        raise socket_tcp.portNotConnectError
    try:
        while client.tbListenerRun:
            # 接受一个字节不取出
            data = client.recv(bufsize=1, bLog=False,
                               flags=socket.MSG_PEEK)
            if data:
                onRecvFunc()
            else:
                break
    except OSError:
        pass

    finally:
        onDisconnectFunc()


def connectThread(client, onConnectFunc, timeoutS):
    if socket_tcp.SocketTCP.connect(client, client.serverIP, client.serverPort,
                                    timeoutS):
        onConnectFunc(True)
    else:
        onConnectFunc(False)


class ClientTCP(socket_tcp.SocketTCP):
    def __init__(self):
        super(ClientTCP, self).__init__()
        self.__serverIP = ''
        self.__serverPort = 0
        self.__tbListenerRun = False

    @property
    def serverIP(self):
        return self.__serverIP

    @property
    def serverPort(self):
        return self.__serverPort

    @property
    def tbListenerRun(self):
        return self.__tbListenerRun

    def _setServer(self, serverIP, serverPort):
        self.__serverIP = serverIP
        self.__serverPort = serverPort

    def startListener(self, onRecvFunc=None, onDisconnectFunc=None):
        """Start monitoring the port.

        Args:
            onRecvFunc: function, optional
                The event triggered after the port receives the data,
                default: None, call the default trigger function.
            onDisconnectFunc: function, optional
                The event triggered after the port is disconnected,
                default: None, call the default trigger function.

        Returns:
            None.
        """
        if onRecvFunc is None:
            onRecvFunc = self.onRecv
        if onDisconnectFunc is None:
            onDisconnectFunc = self.onDisconnect
        args = (self, onRecvFunc, onDisconnectFunc)
        self.tListener = threading.Thread(target=listenerThread, args=args)
        self.__tbListenerRun = True
        self.tListener.start()

    def stopListener(self):
        self.__tbListenerRun = False

    def connect(self, ip, port, timeoutS=2, onConnectFunc=None):
        """Connect to server

        Args:
            ip: string
                Server IP address.
            port: int
                Network port number.
            timeoutS: float, optional
                The longest waiting time, unit(s), default: 2.
            onConnectFunc: function, optional
                The event triggered by a successful or failed connection,
                default: None, call the default trigger function.

        Returns:
            None.
        """
        self._setServer(ip, port)
        if onConnectFunc is None:
            onConnectFunc = self.onConnect
        args = (self, onConnectFunc, timeoutS)
        self.tConnect = threading.Thread(target=connectThread, args=args)
        self.tConnect.start()

    def close(self):
        self.stopListener()
        super(ClientTCP, self).close()

    def onConnect(self, bSuccess):
        if bSuccess:
            print('Connect success')
        else:
            print('Connect fail')

    def onRecv(self):
        pass

    def onDisconnect(self):
        self.close()
        print('The listening thread has stopped')
