source venv/bin/activate


tune_encoder_mi () {
    lb=("lb_dv" "lb_nwj" "lb_mine")
    for model in "${lb[@]}"; do
        rf tune -s experiments/infomax/$1/$model -m models.$model -d data_loaders.$2 -t train -e estimate max -a bayes -f enc_activation var_eps encoder=../../encoder --max_trials 16
    done

    ub=("ub_dv" "ub_nwj" "ub_mine" "ub_variational")
    for model in "${ub[@]}"; do
        rf tune -s experiments/infomax/$1/$model -m models.$model -d data_loaders.$2 -t train -e estimate min -a bayes -f enc_activation var_eps encoder=../../encoder --max_trials 16
    done
}


# Autoencoder.
tune_encoder_mi ae mnist

# NAT encoder.
tune_encoder_mi nat mnist_nat

# GAN encoder.
tune_encoder_mi gan mnist
