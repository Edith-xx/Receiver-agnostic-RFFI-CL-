#!/bin/bash

lams=(1.0 0.5 0.1 0.05 0.01 0.005 0.001)
#lams=(1.0)

for lam in "${lams[@]}"
do
    echo "Running train_DIFEX1 with lam=$lam"
    python train_DIFEX1.py --beta 1.0 --lam "$lam"
    python train_DIFEX1.py --beta 0.5 --lam "$lam"
    python train_DIFEX1.py --beta 0.1 --lam "$lam"
    python train_DIFEX1.py --beta 0.05 --lam "$lam"
    python train_DIFEX1.py --beta 0.01 --lam "$lam"
    python train_DIFEX1.py --beta 0.005 --lam "$lam"
    python train_DIFEX1.py --beta 0.001 --lam "$lam"
done
#左边文件夹的参数第一个是lam,第二个是beta
#day1Rx14_Rx512_F1256_F2256里面跑了两组实验，下面一组的256_512,上面256_256