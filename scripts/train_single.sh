export CUDA_VISIBLE_DEVICES=0

python run_soul.py \
    --workers 4 \
    --seed 42 \
    --log_dir ./logs \
    --data_dir ./data/cifar10 \
    --model_dir ./saved_models \
    --epochs 150 \
    --batch_size 128 \
    --optimizer adam \
    --scheduler cosine \
    --learning_rate 0.0001 \
    --weight_decay 0.0 \
    --momentum 0.9 \
    --model spikingvgg9 \
    --neuron_type lif \
    --time_step 4 \
    --dataset_name cifar10