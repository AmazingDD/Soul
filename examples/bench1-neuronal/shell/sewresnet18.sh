#!/bin/bash

datasets=("TinyImageNet" "DVSGesture")
num_classes=(200 11)
T_values=(4 16)
gpus=(0 1 2 3)  # 4张GPU
seeds=(41 42 43)

NUM_GPUS=${#gpus[@]}
TASK_COUNT=0

for i in "${!datasets[@]}"; do
    dataset="${datasets[$i]}"
    classes="${num_classes[$i]}"
    T="${T_values[$i]}"
    dataset_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
    for neuron_type in "LIF" "PLIF" "GatedLIF" "CLIF" "ILIF"; do
        for seed in "${seeds[@]}"; do
            gpu="${gpus[$((TASK_COUNT % NUM_GPUS))]}"
            log_dir="${dataset_lower}_logs"
            log_file="${log_dir}/${dataset_lower}_${neuron_type}_${seed}.log"
            mkdir -p "$log_dir"
            
            echo "Running on GPU $gpu: $dataset with $classes classes, T=$T, and seed=$seed"
            CUDA_VISIBLE_DEVICES=$gpu python train.py --T $T --num_classes $classes \
                --batch_size 16 --dataset "$dataset" --data_dir "data/$dataset" --epochs 100 \
                --lr 5e-4 --model_type "SEWResNet18" --neuron_type ${neuron_type} --seed "$seed" \
                > "$log_file" &
            
            ((TASK_COUNT++))  # 增加任务计数
            
            # 控制并行任务数量，确保不会超出 GPU 资源
            if ((TASK_COUNT % NUM_GPUS == 0)); then
                wait  # 等待当前批次任务完成后再继续
            fi
        done
    done
done

# 确保所有任务都完成
wait
