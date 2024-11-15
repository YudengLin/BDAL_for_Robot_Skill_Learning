import os
import time

import apps

FILE_PATH, FILE_FULL_NAME = os.path.split(os.path.realpath(__file__))
FILE_NAME, FILE_EXT = os.path.splitext(FILE_FULL_NAME)

if __name__ == '__main__':
    try:
        myapp = apps.App()
        # Connect the device
        myapp.client.connect('101.6.93.189', 5000, 2)

        # No connection is successful until 3 seconds, an exception is thrown
        for i in range(4):
            if i == 3:
                raise Exception('Connect failed')
            if not myapp.client.isConnected():
                time.sleep(1)
                print('Connecting to device')
            else:
                break

        myapp.client.startListener()  # Start monitoring the port
        ret = myapp.cmdMapAbort()
        if ret[0]:
            print('Success')
        else:
            print('Fail')
    finally:
        myapp.client.close()
