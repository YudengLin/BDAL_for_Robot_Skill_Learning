import os, sys
sys.path.append('../../4K_HETU/app')
import arrayexp
import globals.global_func as gf
import numpy as np
import arrayparams

if __name__ == '__main__':
    on_chip=True
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    if on_chip:
        array = arrayexp.ArrayExp()
    else:
        array = None
    params = arrayparams.ArrayParams()
    
    params.PulseWidth = 100
    params.PulseMax = 1000
    params.ReadV = 0.2
    params.AccessV = 5.0
    params.FormVBLStart = 3.
    params.FormVBLStep = 0.3
    params.FormVBLEnd = 5.
    params.FormVWLStart = 1.1
    params.FormVWLStep = 0.3
    params.FormVWLEnd = 2.7
    params.SetVBLStart = 1.8
    params.SetVBLStep = 0.05
    params.SetVBLEnd = 3.
    params.SetVWLStart = 1.1
    params.SetVWLStep = 0.05
    params.SetVWLEnd = 3.0
    params.RstVSLStart = 1.9
    params.RstVSLStep = 0.05
    params.RstVSLEnd = 4.
    params.RstVWLStart = 3.2
    params.RstVWLStep = 0.05
    params.RstVWLEnd = 5.0
    params.ErrorHigh = 0.2
    params.ErrorLow = 0.2
    ITarget = 4.5
    params.BLStart = 0
    params.BLCnt = 1
    params.WLStart = 0
    params.WLCnt = 32
    params.LogLevel = 3
    params.TIAFeedback = 3

    ITargets = np.ones((params.BLCnt, params.WLCnt))+0.5

    print(ITargets)
    params.RdAfterW = 0
    READNUM = params.RdAfterW +1

    array.MapOp(ITargets, params)
    
    rddata = array.GetOpResult(0, READNUM*params.BLCnt*params.WLCnt)*array.r2i[3]
    rddata = rddata.reshape(params.BLCnt, params.WLCnt, READNUM)
    print(rddata)
    
    if on_chip:
        array.close()
