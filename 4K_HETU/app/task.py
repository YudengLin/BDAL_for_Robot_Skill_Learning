import apps
# import reram_register


def userTask(myapp):
    userTask_6(myapp)


def userTask_1(myapp):
    pulseWidth = 100  # 20 * 100 = 2000us
    pulseGap = 46  # 20 * 46 = 920us
    voltage_0 = 32 * [3]
    voltage_1 = 32 * [3]
    voltage_2 = 32 * [3]
    voltage_0[1] = voltage_1[1] = voltage_2[1] = 5

    voltageData = [voltage_0, voltage_1, voltage_2]
    ret = myapp.cmdPulseGenerator(pulseWidth, pulseGap, voltageData)
    print(ret[0])


def userTask_2(myapp):
    addrInfo = apps.AddrInfo()
    addrInfo.chipNum = 3
    addrInfo.BLStart = 0
    addrInfo.BLCnt = 1
    addrInfo.WLStart = addrInfo.SLStart = 0
    addrInfo.WLCnt = addrInfo.SLCnt = 2
    ret = myapp.readOperation(addrInfo)
    print(ret[1])


def userTask_3(myapp):
    SLStart = 0
    SLCnt = 32
    BLStart = 0
    BLCnt = 18
    readV = 32 * [0.2]
    ret = myapp.parallelRead(readV, SLStart, SLCnt, BLStart, BLCnt)
    if ret[0]:
        for index, value in enumerate(ret[1]):
            print(index, value)


def userTask_4(myapp):
    ret = myapp.readAll(3)
    print(ret[1])


def userTask_5(myapp):
    ret = myapp.cellPara()
    print(len(ret[1]))


def userTask_6(myapp):
    mapPara = apps.MapPara()
    addr = apps.AddrInfo(3, 0, 1, 0, 1)
    targetList = 4096 * [3]
    mapPara.maxProgramNum = 50
    mapPara.errorHigh = 0.3
    mapPara.errorLow = 0.3
    myapp.cmdMapArray(mapPara, addr, targetList)

    ret = myapp.cellPara(0, 20)
    if ret[0]:
        print(len(ret[1]))
        for i in range(20):
            print(ret[1][11 * i : 11 * i + 11])
