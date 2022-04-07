# qrnn
quantile regression with neural network

run ```weight_to_uniform.py``` to compute the sample weight before training

run ```transform.py``` to scale/transform the features and targets

read files and setup the structure of the neural net in ```try_qrnn.py```

```qrnn.py``` provides the method to train and test

draw histograms of data and (un)correted MC with ```compare_data_mc.py``` and check the dependency on pT, eta, phi and rho with ```check_results.py```

use ```submit.sh``` to submit the job. use ```submitGPU.sh``` if it's a GPU job
