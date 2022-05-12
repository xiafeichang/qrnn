import numpy as np
from qrnn import trainQuantile, predict

import logging
logger = logging.getLogger(__name__)

class Shifter:

    def __init__(self,mcp0tclf,datap0tclf,mcq_model,X,Y,qs,qweights,final_reg=False):
        self.qs = qs
        self.qweights = qweights

        proba_mc_clf = mcp0tclf.predict_proba(X)
        self.pPeak_mc = proba_mc_clf[:,0]
        self.pTail_mc = proba_mc_clf[:,1]

        proba_data_clf = datap0tclf.predict_proba(X)
        self.pPeak_data = proba_data_clf[:,0]
        self.pTail_data = proba_data_clf[:,1]

        if final_reg:
            self.tailReg = mcq_model
            self.mcqtls = None
            self.X = np.array(X)
        else:
            self.mcqtls   = np.array(predict(X,qs,qweights,mcq_model)).T
            self.tailReg = None

        self.Y = np.array(Y)

        self.Np2t = 0
        self.Nt2p = 0

    def shiftYev(self,iev):

        Y = self.Y[iev]

        r=np.random.uniform()

        drats=self.get_diffrats(self.pPeak_mc[iev],self.pTail_mc[iev],self.pPeak_data[iev],self.pTail_data[iev])

        if Y == 0. and self.pTail_data[iev]>self.pTail_mc[iev] and r<drats[0]:
            Y_corr = self.p2t(iev)
            self.Np2t += 1
        elif Y > 0. and self.pPeak_data[iev]>self.pPeak_mc[iev] and r<drats[1]:
            Y_corr = 0.
            self.Nt2p += 1
        else:
            Y_corr = Y

        return Y_corr

    def p2t(self,iev):

        epsilon = 1.e-5
        r=np.random.uniform(0.01+epsilon,0.99)
        bins = self.qs 

        if self.mcqtls is not None:
            indq = np.searchsorted(bins,r)
            y_tail = np.interp(r,bins[indq-1:indq+1],[self.mcqtls[indq-1,iev],self.mcqtls[indq,iev]])
            if y_tail<=0.:
                logger.info('Warning! Shifting to values <=0. r = {}, bins = {}, qtls = {}'.format(r,bins[indq-1:indq+1],[self.mcqtls[indq-1,iev],self.mcqtls[indq,iev]]))
                print('Warning! Shifting to values <=0. iev = {}, r = {}, bins = {}, qtls = {}'.format(iev,r,bins[indq-1:indq+1],[self.mcqtls[indq-1,iev],self.mcqtls[indq,iev]]))
        elif self.tailReg is not None:
            y_tail = float(self.tailReg.predict(np.hstack((self.X[iev],r)).reshape(1,-1)))

        return y_tail

    def get_diffrats(self,pPeak_mc,pTail_mc,pPeak_data,pTail_data):
        return [np.divide(pTail_data-pTail_mc,pPeak_mc),np.divide(pPeak_data-pPeak_mc,pTail_mc)]

    def __call__(self):
        return np.array([self.shiftYev(iev) for iev in range(self.Y.size)]).ravel()


def applyShift(X,Y,mcp0tclf,datap0tclf,mcq_model,qs=None,qweights=None,final_reg=False):
    if (qs is None) or (qweights is None): 
        qs = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99])
        qweights = np.ones_like(qs)
    shifter = Shifter(mcp0tclf,datap0tclf,mcq_model,X,Y,qs,qweights,final_reg)
    shifted_Y = shifter()
    print('>>>>>>>>>>>>>>> number of events from peak to tail: {}, from tail to peak: {}'.format(shifter.Np2t, shifter.Nt2p))
    return shifted_Y
