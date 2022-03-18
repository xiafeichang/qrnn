# Log of NN setup
## This file is to record the trials of parameters and their performance during the optimization of the neural net

*17.03.2021:* 
for ```probeS4```, give up regularization, use ```SGD``` optimizer: performs well in most quantiles, but a little bit larger for the quantiles around 0.3
*18.03.2021:* 
| variable | N_layers | N_units | acitivation | batch_size | optimizer | regularization | val_loss | comments |
| -------- | -------- | ------- | ----------- | ---------- | --------- | -------------- | -------- | -------- |
| probePhiWidth| 5 | 500 each | tanh each | $2^10$ | Nadam, 0.1 | l2, 1e-6 | 0.3959 | globally good, but bad in $p_T$ dependence |

