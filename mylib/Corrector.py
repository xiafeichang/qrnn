import numpy as np
from qrnn import trainQuantile, predict

class Corrector:

   # store regressors
   def __init__(self,mc_model,data_model,X,Y,qs,qweights,scale_par=None,diz=False):
      self.diz=diz #Flag for distribution with discrete 0, i.e. Isolation
      self.mcqtls   = np.array(predict(X,qs,qweights,mc_model,scale_par))
      self.dataqtls = np.array(predict(X,qs,qweights,data_model,scale_par))

      self.Y = np.array(Y)

   # correction is actually done here
   def correctEvent(self,iev):

#      mcqtls = self.mcqtls[:,iev]
#      dataqtls = self.dataqtls[:,iev]

      mcqtls = self.mcqtls[iev]
      dataqtls = self.dataqtls[iev]
      Y = self.Y[iev]

      if self.diz and Y == 0.:
         return 0.

      qmc =0

      while qmc < len(mcqtls): # while + if, to avoid bumping the range
         if mcqtls[qmc] < Y:
            qmc+=1
         else:
            break

      if qmc == 0:
         qmc_low,qdata_low   = 0,0                              # all shower shapes have a lower bound at 0
         qmc_high,qdata_high = mcqtls[qmc],dataqtls[qmc]
      elif qmc < len(mcqtls):
         qmc_low,qdata_low   = mcqtls[qmc-1],dataqtls[qmc-1]
         qmc_high,qdata_high = mcqtls[qmc],dataqtls[qmc]
      else:
         qmc_low,qdata_low   = mcqtls[qmc-1],dataqtls[qmc-1]
         qmc_high,qdata_high = mcqtls[len(mcqtls)-1]*1.2,dataqtls[len(dataqtls)-1]*1.2
         # to set the value for the highest quantile 20% higher

      return (qdata_high-qdata_low)/(qmc_high-qmc_low) * (Y - qmc_low) + qdata_low

   def __call__(self):
      return np.array([ self.correctEvent(iev) for iev in range(self.Y.size) ]).ravel()

def applyCorrection(X,Y,mc_model,data_model,qs=None,qweights=None,scale_par=None,diz=False):
    if (qs is None) or (qweights is None): 
        qs = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99])
        qweights = np.ones_like(qs)
    return Corrector(mc_model,data_model,X,Y,qs,qweights,scale_par,diz)()
