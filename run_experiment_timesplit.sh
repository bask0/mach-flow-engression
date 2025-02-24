#!/bin/bash

# Define the script path
SCRIPT_PATH="cli/cli.py"
COMMAND="python ${SCRIPT_PATH} --model LSTM -c cli/config_timesplit.yaml --overwrite"

# Define the arrays for beta and alpha
betas=(0.3333 0.6666 1.0 1.3333 1.6666)
es_lengths=(1 5)
noise_dims=(5 20 100)
# betas=(0.3333)
# es_lengths=(1)
# noise_dims=(20)

# Loop over each beta
for beta in "${betas[@]}"; do
    # Loop over each es_length
    for es_length in "${es_lengths[@]}"; do
        # Loop over each alpha
        for noise_dim in "${noise_dims[@]}"; do
            # Print or use the current combination of beta and alpha
            args="--criterion.es_beta ${beta} --criterion.es_length ${es_length} --criterion.noise_dim ${noise_dim}"
            beta_name=$(printf "beta%.2f" $beta)
            es_length_name=$(printf "esl%03d" $es_length)
            noise_dim_name=$(printf "nl%03d" $noise_dim)
            config_name="--config_name ${beta_name}_${es_length_name}_${noise_dim_name}"
            CMD="${COMMAND} ${config_name} ${args}"
            echo "Running: ${CMD}"
            eval $CMD
        done
    done
done

# COMMAND="python model_summary.py"
# eval $CMD
