#!/bin/bash

datasets=("CIFAR10" "CIFAR10DVS" "NMNIST" "Caltech101" "NCaltech101")
num_classes=(10 10 10 101 101)
T_values=(4 20 20 4 8)
gpus=(0 1 2 3)  # 4张GPU
seeds=(41 42 43)

NUM_GPUS=${#gpus[@]}
TASK_COUNT=0

for i in "${!datasets[@]}"; do
    dataset="${datasets[$i]}"
    classes="${num_classes[$i]}"
    T="${T_values[$i]}"
    dataset_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
    
    for seed in "${seeds[@]}"; do
        gpu="${gpus[$((TASK_COUNT % NUM_GPUS))]}"
        log_dir="${dataset_lower}_logs"
        log_file="${log_dir}/${dataset_lower}_ILIF_${seed}.log"
        mkdir -p "$log_dir"
        
        echo "Running on GPU $gpu: $dataset with $classes classes, T=$T, and seed=$seed"
        CUDA_VISIBLE_DEVICES=$gpu python train.py --T $T --num_classes $classes \
            --batch_size 16 --dataset "$dataset" --data_dir "data/$dataset" --epochs 100 \
            --lr 5e-4 --model_type "SpikingVGG9" --neuron_type ILIF --seed "$seed" \
            > "$log_file" &
        
        ((TASK_COUNT++))  # 增加任务计数
        
        # 控制并行任务数量，确保不会超出 GPU 资源
        if ((TASK_COUNT % NUM_GPUS == 0)); then
            wait  # 等待当前批次任务完成后再继续
        fi
    done
done

# 确保所有任务都完成
wait
