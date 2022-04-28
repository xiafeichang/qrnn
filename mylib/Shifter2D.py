import numpy as np
from qrnn import trainQuantile, predict

class Shifter2D:


    def __init__(self,mcp0tclf,datap0tclf,mcq_model0,mcq_model1,X,Y,qs,qweights,final_reg=False):
        self.qs = qs
        self.qweights = qweights
        X = np.array(X)
        Y = np.array(Y)

        proba_mc_clf = mcp0tclf.predict_proba(X)
        self.p00_mc = proba_mc_clf[:,0]
        self.p01_mc = proba_mc_clf[:,1]
        self.p11_mc = proba_mc_clf[:,2]

        proba_data_clf = datap0tclf.predict_proba(X)
        self.p00_data = proba_data_clf[:,0]
        self.p01_data = proba_data_clf[:,1]
        self.p11_data = proba_data_clf[:,2]

        if final_reg:
            self.tailReg0 = mcq_model0
            self.tailReg1 = mcq_model1
            self.X = X
            self.mcqtls0 = None
            self.mcqtls1 = None

        else:
            self.mcqtls0 = np.array(predict(np.hstack((X,Y[:,1].reshape(-1,1))),qs,qweights,mcq_model0)).T
            self.mcqtls1 = np.array(predict(np.hstack((X,Y[:,0].reshape(-1,1))),qs,qweights,mcq_model1)).T
            self.tailReg0 = None
            self.tailReg1 = None

        self.Y = Y


    def shiftYev(self,iev):

        Y = self.Y[iev]
        r=np.random.uniform()
        p=np.random.uniform()

        if Y[0] == 0. and Y[1] == 0. and self.p00_mc[iev] > self.p00_data[iev] and r<=self.w(self.p00_mc[iev],self.p00_data[iev]):
            if self.p01_mc[iev]<self.p01_data[iev] and self.p11_mc[iev]>self.p11_data[iev]:
                Y_shift = np.array([0.,self.p2t(iev)[1]])
            elif self.p01_mc[iev]>self.p01_data[iev] and self.p11_mc[iev]<self.p11_data[iev]:
                Y_shift = self.p2t(iev)
            elif self.p01_mc[iev]<self.p01_data[iev] and self.p11_mc[iev]<self.p11_data[iev]:
                if p<=self.z(self.p01_mc[iev],self.p01_data[iev],self.p00_mc[iev],self.p00_data[iev]):
                    Y_shift = np.array([Y[0],self.p2t(iev)[1]])
                else:
                    Y_shift = self.p2t(iev)

        elif Y[0] == 0. and Y[1] > 0. and self.p01_mc[iev] > self.p01_data[iev] and r<=self.w(self.p01_mc[iev],self.p01_data[iev]):
            if self.p00_mc[iev]<self.p00_data[iev] and self.p11_mc[iev]>self.p11_data[iev]:
                Y_shift = np.zeros(2)
            elif self.p00_mc[iev]>self.p00_data[iev] and self.p11_mc[iev]<self.p11_data[iev]:
                Y_shift = np.array([self.p2t(iev)[0],Y[1]])
            elif self.p00_mc[iev]<self.p00_data[iev] and self.p11_mc[iev]<self.p11_data[iev]:
                if p<=self.z(self.p00_mc[iev],self.p00_data[iev],self.p01_mc[iev],self.p01_data[iev]):
                    Y_shift = np.zeros(2)
                else:
                    Y_shift = np.array([self.p2t(iev)[0],Y[1]])

        elif Y[0] > 0. and Y[1] > 0. and self.p11_mc[iev] > self.p11_data[iev] and r<=self.w(self.p11_mc[iev],self.p11_data[iev]):

            if self.p00_mc[iev]<self.p00_data[iev] and self.p01_mc[iev]>self.p01_data[iev]:
                Y_shift = np.zeros(2)
            elif self.p00_mc[iev]>self.p00_data[iev] and self.p01_mc[iev]<self.p01_data[iev]:
                Y_shift = np.array([0.,Y[1]])
            elif self.p00_mc[iev]<self.p00_data[iev] and self.p01_mc[iev]<self.p01_data[iev]:
                if p<=self.z(self.p00_mc[iev],self.p00_data[iev],self.p11_mc[iev],self.p11_data[iev]):
                    Y_shift=np.zeros(2)
                else:
                    Y_shift=np.array([0.,Y[1]])
        else:
            Y_shift = Y

        return Y_shift

    def w(self,p_mc,p_data):
        return 1.-np.divide(p_data,p_mc)

    def z(self,pj_mc,pj_data,pi_mc,pi_data):
        return np.divide(pj_data-pj_mc,pi_mc-pi_data)

    def p2t(self,iev):

        epsilon = 1.e-5
        r=np.random.uniform(0.01+epsilon,0.99)
        p=np.random.uniform(0.01+epsilon,0.99)
        bins = self.qs 

        if self.mcqtls0 is not None and self.mcqtls1 is not None:
            indqr = np.searchsorted(bins,r)
            indqp = np.searchsorted(bins,p)
            y_tail = np.array([np.interp(r,bins[indqr-1:indqr+1],[self.mcqtls0[indqr-1,iev],self.mcqtls0[indqr,iev]]),np.interp(p,bins[indqp-1:indqp+1],[self.mcqtls1[indqp-1,iev],self.mcqtls1[indqp,iev]])])

        elif self.tailReg0 is not None and self.tailReg1 is not None:
            y_tail = np.hstack((self.tailReg0.predict(np.hstack((self.X[iev],self.Y[iev][1],r)).reshape(1,-1)),self.tailReg1.predict(np.hstack((self.X[iev],self.Y[iev][0],p)).reshape(1,-1))))

        return y_tail


    def __call__(self):
        return np.array([self.shiftYev(iev) for iev in range(self.Y.shape[0])]).reshape(-1,2)


def apply2DShift(X,Y,mcp0tclf,datap0tclf,mcq_model0,mcq_model1,qs=None,qweights=None,final_reg=False):
    if (qs is None) or (qweights is None): 
        qs = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99])
        qweights = np.ones_like(qs)
    return Shifter2D(mcp0tclf,datap0tclf,mcq_model0,mcq_model1,X,Y,qs,qweights,final_reg)()
