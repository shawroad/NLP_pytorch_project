python data_process1.py \
    --full_data=./data/train.json \
    --example_output=./data/train_example.pkl.gz \
    --feature_output=./data/train_feature.pkl.gz

python data_process1.py \
    --full_data=./data/test.json \
    --example_output=./data/dev_example.pkl.gz \
    --feature_output=./data/dev_feature.pkl.gz