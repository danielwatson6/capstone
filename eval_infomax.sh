source venv/bin/activate


# Evaluate MI estimator.
#     args: [trial_path] [model] [trial]
eval_neural () {
    printf "$2\t$3\t"
    rf estimate -s $1 -m models.$2 -d data_loaders.mnist --batch_size 1000
}
eval_structured () {
    printf "$2\tno_divadd\t$bs\t"
    rf estimate -s $1 -m models.$2 -d data_loaders.mnist --batch_size 1000
    printf "$2\tdivadd\t"
    rf estimate -s $1 -m models.$2 -d data_loaders.mnist --batch_size 1000
}


# This will produce an estimates.tsv file for each of the InfoMax encoders.
# The files will contain the MI estimates for every trial.
for im_path in experiments/infomax/*; do
    output_path=$im_path/estimates.tsv
    echo "model\ttrial\tscore" > $output_path

    im_model="${im_path##*/}"
    baseline_models=("lb_nce" "ub_loo")
    for model in "${baseline_models[@]}"; do
        eval_structured $im_path/$model $model >> $output_path
    done

    neural_models=("lb_dv" "lb_nwj" "lb_mine" "ub_dv" "ub_nwj" "ub_mine" "ub_variational")
    for model in "${neural_models[@]}"; do
        for trial_path in $im_path/$model/*; do
            trial="${trial_path##*/}"
            if [ "$trial" == ".kerastuner" ]; then
                continue
            elif [ "$trial" == "config.json" ]; then
                continue
            fi
            eval_neural $trial_path $model $trial >> $output_path
        done
    done
done
