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

#if __name__ == '__main__':
class TestReset(object):
    def __init__(self,array,blstart,blcnt,wlstart,wlcnt,vwlstart,vwlstep,vwlend,vslstart,vslstep,vslend,reset_target=0.4):
        self.array = array
        self.BLStart = blstart
        self.BLCnt = blcnt
        self.WLStart = wlstart
        self.WLCnt = wlcnt
        self.VWLStart = vwlstart
        self.VWLStep = vwlstep
        self.VWLEnd = vwlend
        self.VSLStart = vslstart
        self.VSLStep = vslstep
        self.VSLEnd = vslend
        self.reset_target = reset_target
        self.main()

    def main(self):
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        #array = arrayexp.ArrayExp(dut_ip='101.6.93.189')
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
        VBLStart = 0
        VBLStep = 0
        VBLEnd = 0

        '''
        if False:
            VWLStart = 1.4 #1.0 
            VWLStep = 0.01
            VWLEnd = 2.5
            VSLStart = 0.9 #1.1
            VSLStep = 0.00
            VSLEnd = VSLStart
        else:
            VWLStart = 3.2 #1.0 
            VWLStep = 0.05
            VWLEnd = 5.0
            VSLStart = 1.8 #1.1
            VSLStep = 0.05
            VSLEnd = 4.0
        '''
        '''
        VWLStart = 1.2 #1.0 
        VWLStep = 0.05
        VWLEnd = 2.5
        VSLStart = 0.8 #1.1
        VSLStep = 0.05
        VSLEnd = 2.4
        '''
        VWLStart = self.VWLStart
        VWLStep = self.VWLStep
        VWLEnd = self.VWLEnd
        VSLStart = self.VSLStart
        VSLStep = self.VSLStep
        VSLEnd = self.VSLEnd

        #ITarget = 0.4
        ITarget = self.reset_target
        LogLevel = 2
        RdDirect = 1
        VoltMode = 1
        TIAFeedback = 3

        assert AccessV == 1.8
        array.ResetOp(BLStart, BLCnt, WLStart, WLCnt, VSLStart, VSLStep, VSLEnd,
                    VWLStart, VWLStep, VWLEnd, ITarget, ReadV, AccessV, PulseWidth,
                    PulseMax, RdDirect=RdDirect, LogLevel=LogLevel, VoltMode=VoltMode)

        PrintData = False
        SaveData = False
        PrintFig = False
        SaveFig = SaveData
        #SplitFig = False
        if LogLevel == 2:
            opdata = array.GetOpResult(0, 1) # Get the over all pulse
            pulsenum = int(opdata[0])
            print('All pulse number: %d'%pulsenum)
            opdata = array.GetOpResult(1, pulsenum*9)
            print(opdata.shape)
            if PrintData:
                for i in range(pulsenum):
                    print('%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.3f\t%d'%\
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
            #rddata = array.GetOpResult(0, BLCnt*WLCnt)*array.r2i[3]
            #self.rddata = rddata.reshape(BLCnt,WLCnt,-1)[:,:,0]

            logt = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
            filepath = 'data/reset/'
            if SaveData:
                filename = 'opdata_'+logt+'_reset'+f'_{VSLStep}&{VWLStep}'+'.npz'
                np.savez(filepath+filename, opdata=opdata)

            #get reset data
            rdI = []
            for i in range(pulsenum):
                # if int(opdata[i*9+1]) == 0 and not(i==0) or i==pulsenum-1:
                #     plt.plot(rdI[:-1])
                #     rdI = []
                rdI.append(opdata[i*9+7]*array.r2i[TIAFeedback])
            self.reset_data = rdI
            
            if SaveFig or PrintFig:
                rdI = []
                for i in range(pulsenum):
                    if int(opdata[i*9+1]) == 0 and not(i==0) or i==pulsenum-1:
                        plt.plot(rdI[:-1])
                        rdI = []
                    rdI.append(opdata[i*9+7]*array.r2i[TIAFeedback])
                
                FontSize = 13
                plt.ylim([0,5.5])
                plt.xlabel('Pulse /#', fontsize=FontSize)
                plt.ylabel('Current /uA', fontsize=FontSize)
                plt.xticks(fontsize=FontSize)
                plt.yticks(fontsize=FontSize)
                if PrintFig:
                    plt.savefig('figs/reset/opdata.pdf', 
                        dpi=300, bbox_inches='tight', transparent=True)
                if SaveFig:
                    plt.savefig('figs/reset/'+logt+'_reset'+f'_{VSLStep}&{VWLStep}'+'_opdata.pdf', 
                        dpi=300, bbox_inches='tight', transparent=True)
                plt.close()
            
        #array.close()
