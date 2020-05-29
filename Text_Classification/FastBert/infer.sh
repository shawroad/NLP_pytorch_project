python3 infer.py \
    --model_config_file='config/fastbert_cls.json' \
    --save_model_path='save_model/fastbert_test_distill' \
    --inference_speed=0.5 \
    --infer_data='./data/weibo/dev.tsv' \
    --dump_info_file='infer_info.txt' \
    --data_load_num_workers=2 \
    --gpu_ids=1 \
    --debug_break=0
