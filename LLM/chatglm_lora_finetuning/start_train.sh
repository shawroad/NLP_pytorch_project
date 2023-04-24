CUDA_VISIBLE_DEVICES=0 deepspeed run_finetuning_lora.py --num_train_epochs 5 --train_batch_size 1 --lora_r 8
