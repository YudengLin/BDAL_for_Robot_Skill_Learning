import numpy as np

import globals.global_func as gf

REG_PULSE_GAP = 3
REG_MODE = 4
REG_PULSE_COUNT = 5
REG_PULSE_WIDTH = 6
REG_CHIP_NUM = 11
REG_DAC_WL_BASE = 12
REG_DAC_SL_BASE = 28
REG_DAC_BL_BASE = 44
REG_DAC_TIA_BASE = 108
REG_TIAR_SWITCH_BASE = 140
REG_SWITCH_SL_BASE = 144
REG_SWITCH_WL_BASE = 146
REG_SWITCH_BL_BASE = 148

CHIP_COUNT = 8
BL_COUNT = 128
SL_COUNT = 32
WL_COUNT = 32
TIAV_COUNT = 64
REG_TABLE_SIZE_INT = 156
REG_TABLE_SIZE_BYTE = (REG_TABLE_SIZE_INT * 4)

FORM_MODE = 0x0A
READ_MODE = 0x0B
SET_MODE = 0x0C
RESET_MODE = 0x0D

GND_SWITCH = 0
DAC_SWITCH = 1
FLOAT_SWITCH = 2
TIA_SWITCH = 3

R_25 = 0
R_2500 = 1
R_25K = 2
R_250K = 3

PULSE_UNIT_TIME = 20


class ReramReg():
    def __init__(self):
        self.__regTable = np.zeros(REG_TABLE_SIZE_INT, np.uint32)

    def setChip(self, num):
        if 1 <= num <= CHIP_COUNT:
            self.__regTable[REG_CHIP_NUM] = num
        else:
            raise ValueError(str(1) + ' <= chip number <= ' + str(CHIP_COUNT))

    def setMode(self, mode):
        if FORM_MODE <= mode <= RESET_MODE:
            reg = self.__regTable[REG_MODE]
            reg &= 0xFFFFFFF0
            reg |= mode
            self.__regTable[REG_MODE] = reg
        else:
            raise ValueError(str(FORM_MODE) + ' <= mode <= ' + str(RESET_MODE))

    def setFeedback(self, feedback):
        if not 0 <= feedback <= 3:
            raise ValueError('0 <= feedback <= 3')

        if feedback == R_25:
            RegTemp = 0x0
        elif feedback == R_2500:
            RegTemp = 0x55555555
        elif feedback == R_25K:
            RegTemp = 0xAAAAAAAA
        elif feedback == R_250K:
            RegTemp = 0xFFFFFFFF
        else:
            pass
        self.__regTable[REG_TIAR_SWITCH_BASE] = RegTemp
        self.__regTable[REG_TIAR_SWITCH_BASE + 1] = RegTemp
        self.__regTable[REG_TIAR_SWITCH_BASE + 2] = RegTemp
        self.__regTable[REG_TIAR_SWITCH_BASE + 3] = RegTemp

    def setDacBLV(self, voltage):
        if not len(voltage) == BL_COUNT:
            raise ValueError('The length of BL voltage list is ' +
                             str(BL_COUNT))
        BLNum = 0
        for RegNum in range(BL_COUNT // 2):
            RegTemp1 = gf.VToReg(voltage[BLNum])
            BLNum += 1
            RegTemp2 = gf.VToReg(voltage[BLNum])
            BLNum += 1
            self.__regTable[REG_DAC_BL_BASE +
                            RegNum] = (RegTemp2 << 16) | RegTemp1

    def setDacSLV(self, voltage):
        if not len(voltage) == SL_COUNT:
            raise ValueError('The length of SL voltage list is ' +
                             str(SL_COUNT))
        SLNum = 0
        for RegNum in range(SL_COUNT // 2):
            RegTemp1 = gf.VToReg(voltage[SLNum])
            SLNum += 1
            RegTemp2 = gf.VToReg(voltage[SLNum])
            SLNum += 1
            self.__regTable[REG_DAC_SL_BASE +
                            RegNum] = (RegTemp2 << 16) | RegTemp1

    def setDacWLV(self, voltage):
        if not len(voltage) == WL_COUNT:
            raise ValueError('The length of WL voltage list is ' +
                             str(WL_COUNT))
        WLNum = 0
        for RegNum in range(WL_COUNT // 2):
            RegTemp1 = gf.VToReg(voltage[WLNum])
            WLNum += 1
            RegTemp2 = gf.VToReg(voltage[WLNum])
            WLNum += 1
            self.__regTable[REG_DAC_WL_BASE +
                            RegNum] = (RegTemp2 << 16) | RegTemp1

    def setTIAV(self, voltage):
        for RegNum in range(TIAV_COUNT // 2):
            RegTemp = gf.VToReg(voltage)
            self.__regTable[REG_DAC_TIA_BASE +
                            RegNum] = (RegTemp << 16) | RegTemp

    def _setInputA0(self, number, InputMode, AddrBase):
        if not 0 <= InputMode <= 3:
            raise ValueError('0 <= InputMode <= 3')
        if InputMode & 1:
            self.__regTable[AddrBase] |= (1 << number)
        else:
            self.__regTable[AddrBase] &= ~(1 << number)

    def _setInputA1(self, number, InputMode, AddrBase):
        if not 0 <= InputMode <= 3:
            raise ValueError('0 <= InputMode <= 3')
        if (InputMode >> 1) & 1:
            self.__regTable[AddrBase] |= (1 << number)
        else:
            self.__regTable[AddrBase] &= ~(1 << number)

    def setBLInputSwitch(self, SwitchValue):
        if not len(SwitchValue) == BL_COUNT:
            raise ValueError('The length of BL voltage list is ' +
                             str(BL_COUNT))
        for count in range(BL_COUNT):
            ValueTemp = (count >> 5)
            remainder = (count & 0x1F)
            self._setInputA1(remainder, SwitchValue[count],
                             REG_SWITCH_BL_BASE + ValueTemp)
            self._setInputA0(remainder, SwitchValue[count],
                             REG_SWITCH_BL_BASE + 4 + ValueTemp)

    def setSLInputSwitch(self, SwitchValue):
        if not len(SwitchValue) == SL_COUNT:
            raise ValueError('The length of SL voltage list is ' +
                             str(SL_COUNT))
        for count in range(SL_COUNT):
            self._setInputA1(count, SwitchValue[count], REG_SWITCH_SL_BASE)
            self._setInputA0(count, SwitchValue[count], REG_SWITCH_SL_BASE + 1)

    def setWLInputSwitch(self, SwitchValue):
        if not len(SwitchValue) == WL_COUNT:
            raise ValueError('The length of WL voltage list is ' +
                             str(WL_COUNT))
        for count in range(WL_COUNT):
            self._setInputA1(count, SwitchValue[count], REG_SWITCH_WL_BASE)
            self._setInputA0(count, SwitchValue[count], REG_SWITCH_WL_BASE + 1)

    def setPulse(self, width, gap, count):
        self.__regTable[REG_PULSE_WIDTH] = width // PULSE_UNIT_TIME
        self.__regTable[REG_PULSE_GAP] = gap // PULSE_UNIT_TIME
        self.__regTable[REG_PULSE_COUNT] = count

    def regTableClear(self):
        self.__regTable = np.zeros(REG_TABLE_SIZE_INT, np.uint32)

    def getRegTable(self):
        if self.__regTable[1] or self.__regTable[2]:
            raise Exception('regTable[1] and __regTable[2] must be zreo')
        return self.__regTable

    def setSeqReadV(self, readV, BLOrSL):
        regTemp = gf.VToReg(readV)
        self.__regTable[7] = (BLOrSL << 16) | regTemp


if __name__ == '__main__':
    RegCfg = ReramReg()
    RegCfg.setChip(6)
    RegCfg.setMode(READ_MODE)
    RegCfg.regTablePrint(REG_MODE, 1)
    RegCfg.setFeedback(2)
    RegCfg.regTablePrint(REG_TIAR_SWITCH_BASE, 4)

    voltage = []
    for index in range(BL_COUNT):
        voltage.append(2.3)
    RegCfg.setDacBLV(voltage)
    RegCfg.regTablePrint(REG_DAC_BL_BASE, BL_COUNT/2)

    voltage = []
    for index in range(SL_COUNT):
        voltage.append(2.9)
    RegCfg.setDacSLV(voltage)
    RegCfg.regTablePrint(REG_DAC_SL_BASE, SL_COUNT/2)

    voltage = []
    for index in range(WL_COUNT):
        voltage.append(3)
    RegCfg.setDacWLV(voltage)
    RegCfg.regTablePrint(REG_DAC_WL_BASE, WL_COUNT/2)

    RegCfg.setTIAV(3)
    RegCfg.regTablePrint(REG_DAC_TIA_BASE, TIAV_COUNT/2)

    InputSwitch = []
    for index in range(BL_COUNT):
        InputSwitch.append(DAC_SWITCH)
    RegCfg.setBLInputSwitch(InputSwitch)
    RegCfg.regTablePrint(REG_SWITCH_BL_BASE, 8, 4)

    InputSwitch = []
    for index in range(SL_COUNT):
        InputSwitch.append(TIA_SWITCH)
    RegCfg.setSLInputSwitch(InputSwitch)
    RegCfg.regTablePrint(REG_SWITCH_SL_BASE, 2, 1)

    InputSwitch = []
    for index in range(WL_COUNT):
        InputSwitch.append(FLOAT_SWITCH)
    RegCfg.setWLInputSwitch(InputSwitch)
    RegCfg.regTablePrint(REG_SWITCH_WL_BASE, 2, 1)

    RegCfg.setPulse(10, 12, 4)
    RegCfg.regTablePrint()
