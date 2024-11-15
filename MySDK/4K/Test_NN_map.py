import os, sys
sys.path.append('../../4K_HETU/app')
import arrayexp
import globals.global_func as gf
import numpy as np
import arrayparams
import torch
import math
import time
import logging

if __name__ == '__main__':
    logname = './logs/log'
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logt = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    logf = logging.FileHandler('%s_%s.txt'%(logname,logt), mode='w')
    logf.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")#- %(asctime)s - %(levelname)s")
    logf.setFormatter(formatter)
    logger.addHandler(logf)

    on_chip=True
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    if on_chip:
        array = arrayexp.ArrayExp( dut_ip='101.6.93.189')#dut_ip='192.168.1.10')
    else:
        array = None
    params = arrayparams.ArrayParams()
    # arrayexp.FormReadOp(7, 1, 7, 1, 3.0, 0.5, 4.5, 1.0, 0.5, 2.5, 4.5,
    #                    NEGATIVE_READ)
    # arrayexp.FormReadOp(7, 1, 7, 1, 3.0, 0.1, 5.0, 1.0, 0.1, 3.0, 4.5,
    #                    NEGATIVE_READ)
    ReCs=[30,
            8 ,
            11,
            4 ,
            2 ,
            3 ,
            2 ,
            5 ,
            2 ,
            14,
            3 ,
            16,
            7 ,
            32,
            3 ,
            3 ,
            6 ,
            1 ,
            5 ,
            1 ,
            5 ,
            1 ,
            4 ,
            1 ,
            10,
            10,
            4 ,
            5 ,
            3 ,
            6 ,
            3 ,
            5 ,
            2 ,
            5 ,
            1 ,
            4 ]
    #ReCs = [1]*36
    params.BLStart = 0
    params.BLCnt = 16
    params.WLStart = 0
    params.WLCnt = 16
    params.PulseWidth = 100
    params.PulseMax = 1
    params.ReadV = 0.2
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
    params.Errorbar = 0.1
    params.LogLevel = 0
    params.TIAFeedback = 3 
    params.ReadDirection = 1
    params.VoltMode = 1
    ITarget = 4.5
    NumRow = 64
    NumCol = 32
    

    model = torch.load('train_wn_0.03_wc_2_t2.pth',map_location='cpu')
    n_layer = 0

    FILEPATH = '../../Data/20201230_greedy_W_ALB3/'
    if not os.path.exists(FILEPATH):
        os.makedirs(FILEPATH)
    
    w_qbit=4
    
    for idx,layer in enumerate(model['net'].keys()):
        if  layer[0:6] =='linear' or model['net'][layer].dim()==4:
            n_layer += 1
            #print('Start: ', n_layer, layer)
            #continue

            if n_layer>-1 and n_layer<40:
            
                weight = model['net'][layer]
                w_shape = weight.shape

                WHigh = (2**(w_qbit-1)-1) * ReCs[n_layer-1]
                w_max = weight.data.abs().max().item()
                weight = (weight.data.clone()/w_max*WHigh).round()/WHigh

                weight = weight.view(weight.shape[0],-1).transpose(1,0)
                weight_n = weight.clone()
                n_hsplit = math.ceil(weight.shape[0]*1./NumRow)
                n_vsplit = math.ceil(weight.shape[1]*1./NumCol)
                #print(n_hsplit, n_vsplit)
                #print(weight.shape,n_hsplit,n_vsplit)
                
                for i in range(n_hsplit):
                    for j in range(n_vsplit):
                        
                        stime = time.time()
                        w_split = weight[i*NumRow:(i+1)*NumRow, j*NumCol:(j+1)*NumCol]
                        ITargets = torch.ones(w_split.shape[0]*2, w_split.shape[1])
                        w_pos = w_split.clamp(0)*3.6+0.4
                        w_neg = (-w_split).clamp(0)*3.6+0.4
                        ITargets[0::2,:] = w_pos
                        ITargets[1::2,:] = w_neg
                        #ITargets_nA = torch.ones(200, 300)*1.8+0.4

                        params.BLCnt = ITargets.shape[0]
                        params.WLCnt = ITargets.shape[1]

                        for rec in range(-1, ReCs[n_layer-1]*1000):
                            #params.Errorbar = 0.1
                            #start_time = time.time()
                            #if on_chip:
                            #    array.MapOp(ITargets_nA, params)
                            #    rddata = array.GetOpResult(0, 10*params.BLCnt*params.WLCnt)*array.r2i
                            #    rddata = rddata.reshape(params.BLCnt,params.WLCnt,10)
                            #else:
                            #    rddata = np.expand_dims(ITargets_nA,axis=2) + \
                            #        np.clip(np.random.randn(params.BLCnt, params.WLCnt,10),-3,3)*3.6*0.04
                            #w_real = torch.from_numpy(rddata)
                            #torch.save(w_real, FILEPATH+layer+'_'+str(i)+'_'+str(j)+'_re'+str(rec)+'_w_1.pth')
                            #print("1st write Time used: %.3f s" % (time.time()-start_time))
                            READNUM=2
                            #params.Errorbar = 0.2
                            #start_time = time.time()
                            #if on_chip:
                            #    array.MapOp(ITargets_nA, params)
                            #    rddata = array.GetOpResult(0, READNUM*params.BLCnt*params.WLCnt)*array.r2i
                            #    rddata = rddata.reshape(params.BLCnt,params.WLCnt,READNUM)
                            #else:
                            #    rddata = np.expand_dims(ITargets_nA,axis=2) + \
                            #        np.clip(np.random.randn(params.BLCnt, params.WLCnt,READNUM),-3,3)*3.6*0.04
                            #w_real = torch.from_numpy(rddata)
                            #torch.save(w_real, FILEPATH+layer+'_'+str(i)+'_'+str(j)+'_re'+str(rec)+'_w_2.pth')
                            #print("2nd write Time used: %.3f s" % (time.time()-start_time))

                            params.Errorbar = 0.2
                            start_time = time.time()
                            if on_chip:
                                array.MapOp(ITargets, params)
                                rddata = array.GetOpResult(0, READNUM*params.BLCnt*params.WLCnt)*array.r2i[params.TIAFeedback]
                                if rddata.sum()>0:
                                    print('[ERROR]: Receive Data ERROR')
                                rddata = rddata.reshape(params.BLCnt,params.WLCnt,READNUM)
                            else:
                                rddata = np.expand_dims(ITargets,axis=2) + \
                                    np.clip(np.random.randn(params.BLCnt, params.WLCnt,READNUM),-3,3)*params.Errorbar
                            w_real = torch.from_numpy(rddata)
                            #torch.save(w_real, FILEPATH+layer+'_'+str(i)+'_'+str(j)+'_re'+str(rec)+'_w_3.pth')
                            logger.info("%s:%d vsp %d hsp %d remap %d 3rd write, Time used: %.3f s" \
                                % (layer, n_layer, i, j, rec, time.time()-start_time))

                            #blstart = params.BLStart
                            #blcnt = params.BLCnt
                            #slstart = params.WLStart
                            #slcnt = params.WLCnt
                            #rdvolt = 0.2
                            #RdDirection = 1
    #
                            #rdcnt = 10
                            #paraList = [0, params.BLStart, params.BLCnt, slstart, slcnt, rdcnt, gf.VToReg(rdvolt), RdDirection]
                            #start_time = time.time()
                            ##array.ApplyOp(paraList)
                            #print("Read 10 Time used: %.3f s" % (time.time()-start_time))
                            ## Save the output
                            ##rddata = array.GetOpResult(0, blcnt*slcnt*rdcnt)*array.r2i
                            ##w_real = torch.from_numpy(rddata)
                            ##torch.save(w_real, FILEPATH+layer+'_'+str(i)+'_'+str(j)+'_'+str(rec)+'_r_10.pth')
    #
                            #print('%s %s Time: %.2f s %d,%d'% (layer, rec, (time.time()-stime), i, j))
            
    if on_chip:
        array.close()
