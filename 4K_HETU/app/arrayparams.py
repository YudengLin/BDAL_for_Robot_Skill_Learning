class ArrayParams():
    def __init__(self):
        self.OperationType = 0
        self.OpArrayType = 0
        self.OpPlsType = 0
        self.TIAFeedback = 0
        self.AddrArray = 0
        self.BLStart = 0
        self.BLCnt = 0
        self.SLStart = 0
        self.SLCnt = 0
        self.WLStart = 0
        self.WLCnt = 0
        self.ReadV = 0.2
        self.AccessV = 1.8
        self.VReadStart = 0.
        self.VReadEnd = 0.
        self.NumPoints = 0
        self.NumMeans = 0
        self.FormingTarget = 0.
        self.SetTarget = 0.
        self.ResetTarget = 0.
        self.MappingTarget = 0.
        self.MappingHighRatio = 0.
        self.MappingLowRatio = 0.
        self.GRangeHigh = 0.
        self.GRangeLow = 0.

        self.VBLForming = 0.
        self.VBLSet = 0.
        self.VBLReset = 0.
        self.VWLForming = 0.
        self.VWLSet = 0.
        self.VWLReset = 0.
        self.VSLForming = 0.
        self.VSLSet = 0.
        self.VSLReset = 0.

        self.FormVBLStart = 3.
        self.FormVWLStart = 1.
        self.FormVSLStart = 0.
        self.FormVBLStep = 0.1
        self.FormVWLStep = 0.1
        self.FormVSLStep = 0.
        self.FormVBLEnd = 5.
        self.FormVWLEnd = 4.
        self.FormVSLEnd = 0.
        self.SetVBLStart = 2.5
        self.SetVWLStart = 1.
        self.SetVSLStart = 0.
        self.SetVBLStep = 0.1
        self.SetVWLStep = 0.1
        self.SetVSLStep = 0.
        self.SetVBLEnd = 4.4
        self.SetVWLEnd = 0.
        self.SetVSLEnd = 2.8
        self.RstVBLStart = 0.
        self.RstVWLStart = 0.
        self.RstVSLStart = 0.
        self.RstVBLStep = 0.
        self.RstVWLStep = 0.1
        self.RstVSLStep = 0.1
        self.RstVBLEnd = 0.
        self.RstVWLEnd = 5.
        self.RstVSLEnd = 4.4

        self.BLPar = 2
        self.WLPar = 1

        self.PulseWidth = 0
        self.PulseGap = 0
        self.PulseCount = 0
        self.PulseCycle = 0
        self.PulseMax = 0
        #self.Errorbar = 0.
        self.ErrorHigh = 0.
        self.ErrorLow = 0.

        self.RdAfterW = 0

        self.LogLevel = 0
