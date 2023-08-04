cd  /home/dell/wh/code/HF_Trainer/
torchrun --nproc-per-node 4 \
        main.py --data_load_from_disk True --debug True
