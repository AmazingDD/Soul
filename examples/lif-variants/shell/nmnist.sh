GPUS=(0 1 2 3) # 指定可用的 GPU 设备编号
NUM_GPUS=${#GPUS[@]}  # 计算可用 GPU 的数量
TASK_COUNT=0  # 任务计数器

for neuron_type in "LIF" "PLIF" "GatedLIF" "CLIF" "PSN"; do
    for seed in 41 42 43; do
        GPU_ID=${GPUS[$((TASK_COUNT % NUM_GPUS))]}  # 轮流分配 GPU
        echo "Running training with neuron_type=$neuron_type on GPU $GPU_ID"

        CUDA_VISIBLE_DEVICES=$GPU_ID python train.py --T 20 \
                                                     --num_classes 10 \
                                                     --epochs 50 \
                                                     --batch_size 32 \
                                                     --dataset "NMNIST" \
                                                     --data_dir "./data/NMNIST" \
                                                     --lr 5e-4 \
                                                     --neuron_type "$neuron_type" \
                                                     --seed "$seed" \
                                                     --model_type "SpikingVGG9" > "nmnist_logs/nmnist_${neuron_type}_${seed}.log" &
        
        ((TASK_COUNT++))  # 增加任务计数

        # 控制并行任务数量，确保不会超出 GPU 资源
        if ((TASK_COUNT % NUM_GPUS == 0)); then
            wait  # 等待当前任务完成后再继续
        fi
    done
done

wait  # 等待所有后台任务完成