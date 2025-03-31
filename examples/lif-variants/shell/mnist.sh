for neuron_type in "LIF" "PLIF" "GatedLIF" "PSN"
do
    for seed in 41 42 43
        do
            echo "Running training with neuron_type=$neuron_type"
            python train.py --T 4 \
                            --num_classes 10 \
                            --epochs 50 \
                            --batch_size 32 \
                            --dataset "MNIST" \
                            --data_dir "data/MNIST" \
                            --lr 5e-4 \
                            --neuron_type "$neuron_type" \
                            --seed "$seed" \
                            --model_type "SpikingVGG9"  > "mnist_logs/mnist_${neuron_type}_${seed}.log"
        done
done