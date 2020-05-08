source venv/bin/activate

# This will produce an estimates.tsv file for each of the Gaussian encoders.
# The files will contain the MI estimates for every trial and varying batch sizes.
for alpha_path in experiments/gaussians/*; do
    output_path=$alpha_path/estimates.tsv
    echo 'model\ttrial\tbatch_size\tscore' > $output_path

    for model_path in $alpha_path/*; do
        model="${model_path##*/}"
        if [ "$model" == "encoder" ]; then
            continue
        elif [ "$model" == "estimates.tsv" ]; then
            continue
        fi

        for trial_path in $model_path/*; do
            trial="${trial_path##*/}"
            if [ "$trial" == ".kerastuner" ]; then
                continue
            elif [ "$trial" == "config.json" ]; then
                continue
            fi

            batch_sizes=( 10 100 1000 10000 )
            for bs in "${batch_sizes[@]}"; do
                echo "Evaluating $model (trial=$trial bs=$bs)"
                echo "  saved at $trial_path"
                printf "$model\t$trial\t$bs\t" >> $output_path
                rf estimate -s $trial_path -m models.$model -d data_loaders.gaussian --batch_size $bs >> $output_path
            done
        done
    done
done
