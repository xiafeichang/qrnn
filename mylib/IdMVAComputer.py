import numpy as np
import ROOT as rt
from joblib import delayed, Parallel

class IdMvaComputer:

   def __init__(self,weightsEB,weightsEE,correct=[],tpC='qr',leg2016=False):
      rt.gROOT.LoadMacro("./phoIDMVAonthefly.C")
      
      self.rhoSubtraction = False
      # if type(correct) == dict:
      #    self.rhoSubtraction = correct["rhoSubtraction"]
      #    correct = correct["correct"]
         
      
      self.tpC = tpC
      self.leg2016 = leg2016
      self.X = rt.phoIDInput()
      self.readerEB = rt.bookReadersEB(weightsEB, self.X)
      
      self.readerEE = rt.bookReadersEE(weightsEE, self.X, self.rhoSubtraction, self.leg2016)
      
      # print ("IdMvaComputer.__init__")
      if leg2016:
         columns = ["probeScEnergy","probeScEta","rho","probeR9","probeSigmaIeIe","probePhiWidth","probeEtaWidth","probeCovarianceIetaIphi","probeS4","probePhoIso","probeChIso03","probeChIso03worst","probeSigmaRR","probeScPreshowerEnergy","probePt"]
      else:
         columns = ["probeScEnergy","probeScEta","rho","probeR9","probeSigmaIeIe","probePhiWidth","probeEtaWidth","probeCovarianceIeIp","probeS4","probePhoIso","probeChIso03","probeChIso03worst","probeSigmaRR","probeScPreshowerEnergy","probePt"]

      if self.rhoSubtraction:
         self.effareas = np.array([[0.0000, 0.1210],   
                                   [1.0000, 0.1107],
                                   [1.4790, 0.0699],
                                   [2.0000, 0.1056],
                                   [2.2000, 0.1457],
                                   [2.3000, 0.1719],
                                   [2.4000, 0.1998],
         ])
         
      # make list of input columns
      if self.tpC=="qr":
         print("Using variables corrected by quantile regression")
         self.columns = map(lambda x: x+"_corr" if x in correct else x, columns)
         print(self.columns)
         
      elif self.tpC=="final":
         print("Using variables corrected by final regression")
         self.columns = map(lambda x: x+"_corr_1Reg" if x in correct else x, columns)
         print(self.columns)

      elif self.tpC=="uncorr":
         print("Using uncorrected variables from flashgg")
         self.columns = ['{}_uncorr'.format(x) if x in correct else x for x in columns]
         # self.columns = map(lambda x: x+"_uncorr" if x in correct else x, columns)
         print(self.columns)

      elif self.tpC=="old":
         print("Using variables corrected by old method")
         self.columns = map(lambda x: x+"_old_corr" if x in correct else x, columns)
         print(self.columns)

      elif self.tpC=="data":
         print("Using uncorrected variables")
         self.columns = columns
         print(self.columns)

      elif self.tpC=="n-1qr":
         print("Using variables corrected by N-1 quantile regression")
         self.columns = map(lambda x: x+"_corr_corrn-1" if x in correct else x, columns)
         print(self.columns)

      elif self.tpC=="n-1qrnc":
         print("Using variables corrected by N-1 nc quantile regression")
         self.columns = map(lambda x: x+"_corrn-1" if x in correct else x, columns)
         print(self.columns)

      elif self.tpC=="I2qr":
         print("Using variables corrected by I2 quantile regression")
         self.columns = map(lambda x: x+"_corr_corrn-1_corr" if x in correct else x, columns)
         print(self.columns)
      
      elif self.tpC=="I2n-1qr":
         print("Using variables corrected by I2 N-1 quantile regression")
         self.columns = map(lambda x: x+"_corr_corrn-1_corr_corrn-1" if x in correct else x, columns)
         print(self.columns)
      
   def __call__(self,X):

      # make sure of order of the input columns and convert to a numpy array
      Xvals = X[self.columns ].values
      #print self.columns
      #print Xvals[0]
 
      return np.apply_along_axis( self.predict, 1, Xvals ).ravel()
      
   def predict(self,row):
      return self.predictEB(row) if np.abs(row[1]) < 1.5 else self.predictEE(row)
      # return self.predictEB(row)
      
   def predictEB(self,row):
      # use numeric indexes to speed up
      #print ("IdMvaComputer.predictEB")
      self.X.phoIdMva_SCRawE_          = row[0]
      self.X.phoIdMva_ScEta_           = row[1]
      self.X.phoIdMva_rho_             = row[2]
      self.X.phoIdMva_R9_              = row[3]
      self.X.phoIdMva_covIEtaIEta_     = row[4] # this is really sieie
      self.X.phoIdMva_PhiWidth_        = row[5]
      self.X.phoIdMva_EtaWidth_        = row[6]
      self.X.phoIdMva_covIEtaIPhi_     = row[7]
      self.X.phoIdMva_S4_              = row[8]
      self.X.phoIdMva_pfPhoIso03_      = row[9]
      self.X.phoIdMva_pfChgIso03_      = row[10]
      self.X.phoIdMva_pfChgIso03worst_ = row[11]
      return self.readerEB.EvaluateMVA("BDT")

   def effArea(self,eta):
      ibin = min(self.effareas.shape[0]-1,bisect.bisect_left(self.effareas[:,0],eta))
      return self.effareas[ibin,1]
   
   def predictEE(self,row):
      #print "IdMvaComputer.predictEE"
      self.X.phoIdMva_SCRawE_          = row[0]
      self.X.phoIdMva_ScEta_           = row[1]
      self.X.phoIdMva_rho_             = row[2]
      self.X.phoIdMva_R9_              = row[3]
      self.X.phoIdMva_covIEtaIEta_     = row[4] # this is really sieie
      self.X.phoIdMva_PhiWidth_        = row[5]
      self.X.phoIdMva_EtaWidth_        = row[6]
      self.X.phoIdMva_covIEtaIPhi_     = row[7]
      self.X.phoIdMva_S4_              = row[8]
      self.X.phoIdMva_pfPhoIso03_      = row[9]
      if self.rhoSubtraction: self.X.phoIdMva_pfPhoIso03_ = max(2.5, self.X.phoIdMva_pfPhoIso03_ - self.effArea( np.abs(self.X.phoIdMva_ScEta_))*self.X.phoIdMva_rho_ - 0.0034*row[14] )
      self.X.phoIdMva_pfChgIso03_      = row[10]
      self.X.phoIdMva_pfChgIso03worst_ = row[11]
      self.X.phoIdMva_ESEffSigmaRR_    = row[12]
      esEn                             = row[13]
      ScEn                             = row[0]
      self.X.phoIdMva_esEnovSCRawEn_ = esEn/ScEn
      return self.readerEE.EvaluateMVA("BDT")


def helpComputeIdMva(weightsEB,weightsEE,correct,X,tpC,leg2016):
   return IdMvaComputer(weightsEB,weightsEE,correct,tpC,leg2016)(X)
