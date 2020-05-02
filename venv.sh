source venv/bin/activate
alias run='python run.py'
alias tune='python tune.py'

# Tune an encoder to maximize its MI estimate.
#     args: [save_dir] [lb_module]
tune_encoder_mnist () {
    # tune [save_dir] [model] [data_loader] -t [train_method] -e [eval_method] [min|max]
    #      -a [algorithm] [tuning_options...]"
    tune $1/self $2 mnist -t train -e estimate max -a bayes --max_trials 16 --exclude gaussian
}

# Tune evaluators for a pre-trained encoder.
#    args: [save_dir]/[best_run]
tune_eval_mnist () {
    tune $1/lb_ae mnist -t train -e estimate max -a bayes --max_trials 16 --exclude gaussian
}
