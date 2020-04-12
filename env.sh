source env/bin/activate
alias run='python run.py'

# Args: save_dir, data_loader
run_estimate () {
    printf 'model\tavgmode\tmi\n'

    # Estimates based on MI estimator trained with encoder.
    printf 'self\tmicro\t'
    run estimate $1/self $2
    printf 'self\tmacro\t'
    run estimate $1/self $2 --macro=True

    # New AE based estimates.
    run train $1/lb_ae lb_ae $2
    printf 'lb_ae\tmicro\t'
    run estimate $1/lb_ae $2
    printf 'lb_ae\tmacro\t'
    run estimate $1/lb_ae $2 --macro=True

    # New DV based estimates.
    run train $1/lb_dv lb_dv $2
    printf 'lb_dv\tmicro\t'
    run estimate $1/lb_dv $2
    printf 'lb_dv\tmacro\t'
    run estimate $1/lb_dv $2 --macro=True

    # NCE based estimates.
    printf 'lb_nce\tmicro\t'
    run estimate $1/lb_nce lb_nce $2
    printf 'lb_nce\tmacro\t'
    run estimate $1/lb_nce $2 --macro=True

    # New NWJ based (lower bound) estimates.
    run train $1/lb_nwj lb_nwj $2
    printf 'lb_nwj\tmicro\t'
    run estimate $1/lb_nwj $2
    printf 'lb_nwj\tmacro\t'
    run estimate $1/lb_nwj $2 --macro=True

    # New GAN based estimates.
    run train $1/ub_gan ub_gan $2
    printf 'ub_gan\tmicro\t'
    run estimate $1/ub_gan $2
    printf 'ub_gan\tmacro\t'
    run estimate $1/ub_gan $2 --macro=True

    # LOO based estimates.
    printf 'ub_loo\tmicro\t'
    run estimate $1/ub_loo ub_loo $2
    printf 'ub_loo\tmacro\t'
    run estimate $1/ub_loo $2 --macro=True

    # New NWJ based (upper bound) estimates.
    run train $1/ub_nwj ub_nwj $2
    printf 'ub_nwj\tmicro\t'
    run estimate $1/ub_nwj $2
    printf 'ub_nwj\tmacro\t'
    run estimate $1/ub_nwj $2 --macro=True
}

# Args: save_dir, model, data_loader
run_all () {
    run train $1/self $2 $3
    run_estimate $1 $3 > experiments/$1/estimates.txt
}

# Args: save_dir, mi
run_gauss_estimate () {
    printf 'model\tavgmode\tmi\n'
    printf 'true\ttrue\t'
    printf $2
    printf '\n'

    # New DV based estimates.
    run train $1/lb_dv lb_dv gaussian --gaussian=$2
    printf 'lb_dv\tmicro\t'
    run estimate $1/lb_dv gaussian
    printf 'lb_dv\tmacro\t'
    run estimate $1/lb_dv gaussian --macro=True

    # NCE based estimates.
    printf 'lb_nce\tmicro\t'
    run estimate $1/lb_nce lb_nce gaussian --gaussian=$2
    printf 'lb_nce\tmacro\t'
    run estimate $1/lb_nce gaussian --macro=True

    # New NWJ based (lower bound) estimates.
    run train $1/lb_nwj lb_nwj gaussian --gaussian=$2
    printf 'lb_nwj\tmicro\t'
    run estimate $1/lb_nwj gaussian
    printf 'lb_nwj\tmacro\t'
    run estimate $1/lb_nwj gaussian --macro=True

    # New GAN based estimates.
    run train $1/ub_gan ub_gan gaussian --gaussian=$2
    printf 'ub_gan\tmicro\t'
    run estimate $1/ub_gan gaussian
    printf 'ub_gan\tmacro\t'
    run estimate $1/ub_gan gaussian --macro=True

    # LOO based estimates.
    printf 'ub_loo\tmicro\t'
    run estimate $1/ub_loo ub_loo gaussian --gaussian=$2
    printf 'ub_loo\tmacro\t'
    run estimate $1/ub_loo gaussian --macro=True

    # New NWJ based (upper bound) estimates.
    run train $1/ub_nwj ub_nwj gaussian --gaussian=$2
    printf 'ub_nwj\tmicro\t'
    run estimate $1/ub_nwj gaussian
    printf 'ub_nwj\tmacro\t'
    run estimate $1/ub_nwj gaussian --macro=True
}

# Args: save_dir, mi
run_gauss () {
    run train $1/dummy lb_nce gaussian --gaussian=$2
    run_gauss_estimate $1 $2 > experiments/$1/estimates.txt
}
