import sys
sys.path.append('../../4K_HETU/app')

import arrayexp
import numpy as np
# import arrayparams
import time
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style('ticks')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'

if __name__ == '__main__':
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    array = arrayexp.ArrayExp()
    #params = arrayparams.ArrayParams()
    # BLStart = 96
    # BLCnt = 8
    # WLStart = 0
    # WLCnt = 32
    # PulseWidth = 100000
    # PulseMax = 300
    # ReadV = 0.1
    # AccessV = 1.8
    # VBLStart = 2.
    # VBLStep = 0.05
    # VBLEnd = 3.3
    # VWLStart = 0.4
    # VWLStep = 0.05
    # VWLEnd = 0.8
    # VSLStart = 0.
    # VSLStep = 0.
    # VSLEnd = 0.
    # ITarget = 4.5
    # LogLevel = 2
    # RdDirect = 1
    # VoltMode = 0
    # TIAFeedback = 3
    BLStart     = 0
    BLCnt       = 1
    WLStart     = 0
    WLCnt       = 32
    PulseWidth  = 100000
    PulseMax    = 400
    ReadV       = 0.2
    AccessV     = 5.
    VBLStart    = 3.
    VBLStep     = 0.1
    VBLEnd      = 5.
    VWLStart    = 1.1
    VWLStep     = 0.1
    VWLEnd      = 2.7
    VSLStart    = 0.
    VSLStep     = 0.
    VSLEnd      = 0.
    ITarget     = 4.
    LogLevel    = 2
    RdDirect    = 1
    VoltMode    = 0
    TIAFeedback = 3

    array.FormOp(BLStart, BLCnt, WLStart, WLCnt, VBLStart, VBLStep, VBLEnd,
                 VWLStart, VWLStep, VWLEnd, ITarget, ReadV, AccessV, PulseWidth,
                 PulseMax, RdDirect=RdDirect, LogLevel=LogLevel, VoltMode=VoltMode)

    PrintData = True
    SaveData = False
    SaveFig = False
    SplitFig = False
    if LogLevel == 2:
        opdata = array.GetOpResult(0, 1) # Get the over all pulse
        pulsenum = int(opdata[0])
        print('All pulse number: ', pulsenum)
        LEN_LOGDATA = 9
        opdata = array.GetOpResult(1, pulsenum*LEN_LOGDATA)
        if PrintData:
            print("Op\tPulse\tBL\tSL\tVBL\tVSL\tVWL\tIRd\tState\n")
            for i in range(pulsenum):
                print('%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d'%\
                    (opdata[0+i*9],
                    opdata[1+i*9],
                    opdata[2+i*9],
                    opdata[3+i*9],
                    opdata[4+i*9],
                    opdata[5+i*9],
                    opdata[6+i*9],
                    opdata[7+i*9] *array.r2i[TIAFeedback]*1000,
                    opdata[8+i*9]))
        opdata = np.asarray(opdata).reshape([pulsenum, LEN_LOGDATA])

        logt = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        filepath = '../../Data/20210112/'
        if SaveData:
            filename = 'opdata_'+logt+'.npz'
            np.savez(filepath+filename, opdata=opdata)
        if SaveFig:
            rdI = []
            for i in range(pulsenum):
                if int(opdata[i*9+1]) == 0 and not(i==0) or i==pulsenum-1:
                    plt.plot(rdI[:-1])
                    if SplitFig:
                        plt.savefig('figs/'+logt+'_opdata'+str(i)+'.pdf', 
                            dpi=300, bbox_inches='tight', transparent=True)
                    rdI = []
                rdI.append(opdata[i*9+7]*array.r2i[TIAFeedback])
            FontSize = 13
            plt.xlabel('Pulse /#', fontsize=FontSize)
            plt.ylabel('Current /uA', fontsize=FontSize)
            plt.xticks(fontsize=FontSize)
            plt.yticks(fontsize=FontSize)
            plt.savefig('figs/opdata.pdf', 
                dpi=300, bbox_inches='tight', transparent=True)
            plt.savefig('figs/'+logt+'_opdata.pdf', 
                dpi=300, bbox_inches='tight', transparent=True)
            plt.close()

    array.close()
