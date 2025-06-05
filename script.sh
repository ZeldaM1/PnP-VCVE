 
 sudo docker run --gpus all  -it -v /home/public/huimin:/home/public/huimin -v /home/public/USB-4TB-B/huimin:/home/public/USB-4TB-B/huimin  -v /mnt/nvme1/huimin:/mnt/nvme1/huimin  --shm-size 64g registry.cn-hangzhou.aliyuncs.com/zenghuimin/zhm_docker:py37-torch18 /bin/bash

cd /mnt/nvme1/huimin/CompressedVSR/CompressedVSR_ablation_copy/code_save/

 
./tools/dist_train.sh  configs/HR_davis_LR_128x128.py  1   --exp_name HR_davis_LR_128x128
./tools/dist_train.sh  configs/HR_davis_LR_128x128_IPB.py  1   --exp_name HR_davis_LR_128x128_IPB




# ---------- testing 
# ------ HR REDS
  ./tools/dist_test.sh  configs/HR_davis_LR_128x128.py   checkpoint/HR_davis_LR_128x128.pth   1   \
 --testdir_lr  dataset/REDS_test_HR/crf15/png    --testdir_gt dataset/REDS_test_HR/sharp/png   --save-path ./HR_davis_LR_128x128/REDS_test_HR/crf15
 
  ./tools/dist_test.sh  configs/HR_davis_LR_128x128_IPB.py checkpoint/HR_davis_LR_128x128_IPB.pth  1   \
 --testdir_lr  dataset/REDS_test_HR/crf15/png    --testdir_gt dataset/REDS_test_HR/sharp/png --save-path ./HR_davis_LR_128x128_IPB/REDS_test_HR/crf15
 
# ------ VSR: LR REDS 
  ./tools/dist_test.sh  configs/HR_davis_LR_128x128_IPB_LR_test.py checkpoint/HR_davis_LR_128x128_IPB.pth  1   \
 --testdir_lr  dataset/REDS_test_LR/crf15/png    --testdir_gt dataset/REDS_test_LR/X4/png --save-path ./HR_davis_LR_128x128_IPB/REDS_test_HR/crf15


# ------ FLOW: HR KITII

 
# ------ VOS/Inpainting: HR DAVIS 
 




  ./tools/dist_test.sh  configs/HR_davis_LR_128x128.py   checkpoint/HR_davis_LR_128x128.pth   1   \
 --testdir_lr  dataset/REDS_test_HR/crf15/png    --testdir_gt dataset/REDS_test_HR/sharp/png 



  ./tools/dist_test.sh  configs/HR_davis_LR_128x128_IPB.py checkpoint/HR_davis_LR_128x128_IPB.pth  1   \
 --testdir_lr  dataset/REDS_test_HR/crf15/png    --testdir_gt dataset/REDS_test_HR/sharp/png