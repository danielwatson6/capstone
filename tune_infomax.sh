source venv/bin/activate


eval_encoder_loop () {
    for trial_path in experiments/infomax/$1/*; do
        trial="${trial_path##*/}"
        if [ "$trial" == ".kerastuner" ]; then
            continue
        elif [ "$trial" == "config.json" ]; then
            continue
        elif [ "$trial" == "losses.tsv" ]; then
            continue
        fi
        printf "$1\t$trial\t"
        rf evaluate -s $trial_path -m models.infomax_$1 -d data_loaders.$2
    done
}

eval_encoder () {
    echo "model\ttrial\tloss" > experiments/infomax/$1/losses.tsv
    eval_encoder_loop $1 $2 >> experiments/infomax/$1/losses.tsv
}

# Autoencoder.
# rf tune -s experiments/infomax/ae -m models.infomax_ae -d data_loaders.mnist -t train -e evaluate min -a bayes --max_trials 16
eval_encoder ae mnist


# NAT encoder.
# rf tune -s experiments/infomax/nat -m models.infomax_nat -d data_loaders.mnist_nat -t train -e evaluate min -a bayes --max_trials 16
eval_encoder nat mnist_nat


# GAN encoder.
# rf tune -s experiments/infomax/gan -m models.infomax_gan -d data_loaders.mnist -t train -e evaluate min -a bayes --max_trials 16
eval_encoder gan mnist
