GPUS=(3 1 2 0) # 指定可用的 GPU 设备编号
NUM_GPUS=${#GPUS[@]}  # 计算可用 GPU 的数量
TASK_COUNT=0  # 任务计数器

for neuron_type in "LIF"; do
    for seed in 42; do
        GPU_ID=${GPUS[$((TASK_COUNT % NUM_GPUS))]}  # 轮流分配 GPU
        echo "Running training with neuron_type=$neuron_type on GPU $GPU_ID"

        CUDA_VISIBLE_DEVICES=$GPU_ID python train.py --T 4 \
                                                     --num_classes 200 \
                                                     --epochs 100 \
                                                     --batch_size 96 \
                                                     --dataset "TinyImageNet" \
                                                     --data_dir "data/TinyImageNet" \
                                                     --lr 5e-4 \
                                                     --neuron_type "$neuron_type" \
                                                     --seed "$seed" \
                                                     --model_type "SpikingVGG9" > "tinyimagenet_logs/tinyimagenet_spikingvgg9_${neuron_type}_${seed}.log" &
        
        ((TASK_COUNT++))  # 增加任务计数

        # 控制并行任务数量，确保不会超出 GPU 资源
        if ((TASK_COUNT % NUM_GPUS == 0)); then
            wait  # 等待当前任务完成后再继续
        fi
    done
done

wait  # 等待所有后台任务完成
