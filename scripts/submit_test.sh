#! /bin/bash 

for var in "probeCovarianceIeIp" "probeSigmaIeIe" "probeEtaWidth" "probePhiWidth" "probeR9" "probeS4";
do
    echo Submitting training jobs for variable $var on data
    for q in 0.01 0.99 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95;
    do
	echo Submitting training job for quantile $q on data
	sbatch train.sh ${var} ${q}
    done
done
 