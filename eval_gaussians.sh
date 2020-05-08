source venv/bin/activate


# Evaluate Gaussian MI estimator.
#     args: [trial_path] [model] [trial]
eval_trial () {
    batch_sizes=( 10 100 1000 10000 )
    for bs in "${batch_sizes[@]}"; do
        printf "$model\t$trial\t$bs\t"
        rf estimate -s $trial_path -m models.$model -d data_loaders.gaussian --batch_size $bs
    done
}


# This will produce an estimates.tsv file for each of the Gaussian encoders.
# The files will contain the MI estimates for every trial and varying batch sizes.
for alpha_path in experiments/gaussians/*; do
    output_path=$alpha_path/estimates.tsv
    echo 'model\ttrial\tbatch_size\tscore' > $output_path

    baseline_models=("lb_nce" "ub_loo")
    for model in "${baseline_models[@]}"; do
        eval_trial $alpha_path/$model $model "n/a" >> $output_path
    done

    neural_models=("lb_dv" "lb_nwj" "ub_dv" "ub_nwj" "ub_variational")
    for model in "${neural_models[@]}"; do
        for trial_path in $alpha_path/$model/*; do
            trial="${trial_path##*/}"
            if [ "$trial" == ".kerastuner" ]; then
                continue
            elif [ "$trial" == "config.json" ]; then
                continue
            fi

            eval_trial $trial_path $model $trial >> $output_path
        done
    done
done
