import sys

from numpy.core.records import array
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

#if __name__ == '__main__':
class TestSet(object):
    def __init__(self,array,blstart,blcnt,wlstart,wlcnt,vwlstart,vwlstep,vwlend,vblstart,vblstep,vblend,set_target):
        self.array = array
        self.BLStart = blstart
        self.BLCnt = blcnt
        self.WLStart = wlstart
        self.WLCnt = wlcnt
        self.VWLStart = vwlstart
        self.VWLStep = vwlstep
        self.VWLEnd = vwlend
        self.VBLStart = vblstart
        self.VBLStep = vblstep
        self.VBLEnd = vblend
        self.SetTarget = set_target
        self.main()

    def main(self):
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        #array = arrayexp.ArrayExp(dut_ip='101.6.93.189')
        # dut_ip='101.6.93.189' dut_ip='192.168.1.10'
        #params = arrayparams.ArrayParams()
        array = self.array

        BLStart = self.BLStart
        BLCnt = self.BLCnt
        WLStart = self.WLStart
        WLCnt = self.WLCnt

        PulseWidth = 100
        PulseMax = 500
        ReadV = 0.2
        AccessV = 1.8
        CONDITION = 1
        #if CONDITION==0:
        #    VBLStart = 0.8
        #    VBLStep = 0.00
        #    VBLEnd = VBLStart
        #    VWLStart = 0.6
        #    VWLStep = 0.01
        #    VWLEnd = 0.9
        #elif CONDITION==1:
        '''
        VBLStart = 0.8
        VBLStep = 0.05
        VBLEnd = 1.5
        VWLStart = 0.4
        VWLStep = 0.05
        VWLEnd = 0.8
        '''
        VBLStart = self.VBLStart
        VBLStep = self.VBLStep
        VBLEnd = self.VBLEnd
        VWLStart = self.VWLStart
        VWLStep = self.VWLStep
        VWLEnd = self.VWLEnd
        #elif CONDITION==2:
        #    VBLStart = 0.8
        #    VBLStep = 0.05
        #    VBLEnd = 1.5
        #    VWLStart = 0.4
        #    VWLStep = 0.05
        #    VWLEnd = 0.8
        VSLStart = 0.
        VSLStep = 0.
        VSLEnd = 0.
        # ITarget = 4.
        ITarget = self.SetTarget
        LogLevel = 2
        RdDirect = 1
        VoltMode = 1
        TIAFeedback = 3

        assert AccessV == 1.8
        array.SetOp(BLStart, BLCnt, WLStart, WLCnt, VBLStart, VBLStep, VBLEnd,
                    VWLStart, VWLStep, VWLEnd, ITarget, ReadV, AccessV, PulseWidth,
                    PulseMax, RdDirect=RdDirect, LogLevel=LogLevel, VoltMode=VoltMode)
        
        PrintData = False
        SaveData = False
        PrintFig = False
        SaveFig = SaveData
        SplitFig = False
        if LogLevel == 2:
            opdata = array.GetOpResult(0, 1) # Get the over all pulse
            pulsenum = int(opdata[0])
            print('All pulse number: %d'%pulsenum)
            opdata = array.GetOpResult(1, pulsenum*9)
            if PrintData:
                for i in range(pulsenum):
                    print('%d\t%d\t%d\t%d\t%.3f\t%.3f\t%.3f\t%.3f\t%d'%\
                        (opdata[0+i*9],
                        opdata[1+i*9],
                        opdata[2+i*9],
                        opdata[3+i*9],
                        opdata[4+i*9]* 5.0 / 65535.0,
                        opdata[5+i*9]* 5.0 / 65535.0,
                        opdata[6+i*9]* 5.0 / 65535.0,
                        int(opdata[7+i*9])*array.r2i[TIAFeedback],
                        opdata[8+i*9]))
            opdata = np.asarray(opdata)
            
            logt = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
            filepath = 'data/set/'
            if SaveData:
                filename =  'opdata_'+logt+ '_set'+f'_{VWLStep}&{VBLStep}'+ '.npz'
                np.savez(filepath+filename, opdata=opdata)
            
            # get set data
            rdI = []
            for i in range(pulsenum):
                # if int(opdata[i*9+1]) == 0 and not(i==0) or i==pulsenum-1:
                #     plt.plot(rdI[:-1])
                #     rdI = []
                rdI.append(opdata[i*9+7]*array.r2i[TIAFeedback])
            self.set_data = rdI
            
            
            if SaveFig or PrintFig:
                rdI = []
                for i in range(pulsenum):
                    if int(opdata[i*9+1]) == 0 and not(i==0) or i==pulsenum-1:
                        plt.plot(rdI[:-1])
                        if SplitFig:
                            plt.savefig('figs/set/'+ logt+'_set' +'_opdata'+str(i)+'.pdf', 
                                dpi=300, bbox_inches='tight', transparent=True)
                        rdI = []
                    rdI.append(opdata[i*9+7]*array.r2i[TIAFeedback])

                FontSize = 13
                plt.ylim([0,5.5])
                plt.xlabel('Pulse /#', fontsize=FontSize)
                plt.ylabel('Current /uA', fontsize=FontSize)
                plt.xticks(fontsize=FontSize)
                plt.yticks(fontsize=FontSize)
                if PrintFig:
                    plt.savefig('figs/set/opdata.pdf', 
                        dpi=300, bbox_inches='tight', transparent=True)
                if SaveFig:
                    plt.savefig('figs/set/'+logt+'_set'+f'_{VWLStep}&{VBLStep}'+'_opdata.pdf', 
                        dpi=300, bbox_inches='tight', transparent=True)
                plt.close()
        
        #array.close()
