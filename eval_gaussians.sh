source venv/bin/activate

# This will produce an estimates.tsv file for each of the Gaussian encoders.
# The files will contain the MI estimates for every trial and varying batch sizes.
for alpha_dir in experiments/gaussians/*; do
    output_path=experiments/gaussians/$alpha_dir/estimates.tsv
    echo "model\ttrial\tbatch_size\tscore" > $output_path

    for model in $1/alpha_dir/*; do
        if [ "$model" == "encoder" ]; then
            continue
        fi

        for trial in $1/$alpha_dir/$model/*; do
            if [ "$trial" == ".kerastuner" ]; then
                continue
            elif [ "$trial" == "config.json" ]; then
                continue
            fi

            save_dir=experiments/gaussians/$alpha_dir/$model/$trial
            batch_sizes=( 10 100 1000 10000 )
            for bs in "${batch_sizes[@]}"; do
                printf "$model\t$trial\t$bs\t" >> $output_path
                rf estimate -s $save_dir --batch_size $bs >> $output_path
            done
        done
    done
done
