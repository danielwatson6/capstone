source venv/bin/activate


# Tune estimators for Gaussians with fixed MI.
# Saved at "experiments/gaussians/alpha_[mi_value]".
#     args: [mi_value]
tune_gaussian () {
    # Save the Gaussian encoder.
    rf create -s experiments/gaussians/alpha_$1/encoder -m models.encoder -d data_loaders.gaussian --gaussian=$1

    # Save the non-trainable models.
    rf create -s experiments/gaussians/alpha_$1/lb_nce -m models.lb_nce -d data_loaders.gaussian --gaussian=$1
    rf create -s experiments/gaussians/alpha_$1/ub_loo -m models.ub_loo -d data_loaders.gaussian --gaussian=$1

    lb=("lb_dv" "lb_nwj")
    for model in "${lb[@]}"
    do
        rf tune -s experiments/gaussians/alpha_$1/$model -m models.$model -d data_loaders.gaussian -t train -e estimate max -a bayes -f gaussian=$1 enc_activation var_eps encoder=../../encoder --max_trials 16
    done

    ub=("ub_dv" "ub_nwj" "ub_variational")
    for model in "${ub[@]}"
    do
        rf tune -s experiments/gaussians/alpha_$1/$model -m models.$model -d data_loaders.gaussian -t train -e estimate min -a bayes -f gaussian=$1 enc_activation var_eps encoder=../../encoder --max_trials 16
    done
}


mi_values=( 1.0 2.0 4.0 8.0 16.0 )
for alpha in "${mi_values[@]}"
do
    tune_gaussian $alpha
done
