source venv/bin/activate


# Evaluate Gaussian MI estimator.
#     args: [trial_path] [model] [trial]
eval_neural () {
    batch_sizes=( 10 100 1000 )
    for bs in "${batch_sizes[@]}"; do
        printf "$2\t$3\t$bs\t"
        rf estimate -s $1 -m models.$2 -d data_loaders.gaussian --batch_size $bs
    done
}
eval_structured () {
    batch_sizes=( 10 100 1000 )
    for bs in "${batch_sizes[@]}"; do
        printf "$2\tno_divadd\t$bs\t"
        rf estimate -s $1 -m models.$2 -d data_loaders.gaussian --batch_size $bs --gaussian $3
        printf "$2\tdivadd\t$bs\t"
        rf estimate -s $1 -m models.$2 -d data_loaders.gaussian --batch_size $bs --gaussian $3 --div_add 1e-12
    done
}


# This will produce an estimates.tsv file for each of the Gaussian encoders.
# The files will contain the MI estimates for every trial and varying batch sizes.
for alpha_path in experiments/gaussians/*; do
    output_path=$alpha_path/estimates.tsv
    # echo "model\ttrial\tbatch_size\tscore" > $output_path

    alpha="${alpha_path##*/alpha_}"
    baseline_models=("lb_nce" "ub_loo")
    for model in "${baseline_models[@]}"; do
        eval_structured $alpha_path/$model $model $alpha  >> $output_path
    done

    neural_models=("lb_dv" "lb_nwj" "lb_mine" "ub_dv" "ub_nwj" "ub_mine" "ub_variational")
    for model in "${neural_models[@]}"; do
        for trial_path in $alpha_path/$model/*; do
            trial="${trial_path##*/}"
            if [ "$trial" == ".kerastuner" ]; then
                continue
            elif [ "$trial" == "config.json" ]; then
                continue
            fi

            # eval_neural $trial_path $model $trial >> $output_path
        done
    done
done
