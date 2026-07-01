#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "${REPO_ROOT}"

train_data_list=(mvtec visa)
test_data_list=("visa" "mvtec")

sae_epoch=50
image_size=448
input_norm=none

guide_epoch=15

for sae_topk in 8 16 32 64 128; do
    echo "sae_topk=${sae_topk}"

    for sae_hidden_dim in 2048 3072 4096 6144 8192; do
        echo "sae_hidden_dim=${sae_hidden_dim}"

        for i in "${!train_data_list[@]}"; do
            train_data=${train_data_list[i]}
            echo "train=${train_data}" 

            # train sae
            python run.py mode=train_sae model=facebook_dinov3-vitl16 data=single_dataset/${train_data} train.epoch=${sae_epoch} \
                model.image_size=${image_size} \
                sae.0.use_cls=false \
                sae.0.input_norm=${input_norm} \
                sae.0.hidden_dim=${sae_hidden_dim} \
                sae.0.sparsifier_params.topk=${sae_topk} \
                save_freq=${sae_epoch} 

            run_dir=$(ls -td outputs/*/* 2>/dev/null | head -n 1)
            rel=${run_dir#outputs/}
            timestamp=$(echo "$rel" | sed 's|_.*||')

            python run.py mode=train model=facebook_dinov3-vitl16 experiment=guide_sae data=single_dataset/${train_data} train.epoch=${guide_epoch} \
                model.image_size=${image_size} \
                model.method_config.sae.0.hidden_dim=${sae_hidden_dim} \
                save_freq=1 \
                train.learning_rate=0.01 \
                train.ema.warmup_steps=0 \
                model.method_config.sae.0.auxk=512 \
                model.method_config.sae.0.use_cls=false \
                model.method_config.sae.0.input_norm=${input_norm} \
                model.method_config.sae.0.sparsifier_params.topk=${sae_topk} \
                model.method_config.guide_sae.datetime=${timestamp} \
                model.method_config.guide_sae.checkpoint_epoch=${sae_epoch}

            run_dir=$(ls -td outputs/*/* 2>/dev/null | head -n 1)

            # evaluate
            for test_data in ${test_data_list[i]}; do
                echo "test=${test_data}"

                python run.py mode=eval data@test_data=single_dataset/${test_data} train_dir=${run_dir} evaluate.epoch=${guide_epoch} \
                    evaluate.use_ema=true \
                    evaluate.image_score.mode="map" \
                    evaluate.image_score.map_pool="max" \
                    evaluate.pro_use_fast=true

                mv "${run_dir}/metrics/metrics_${test_data}_${guide_epoch}_ema.csv" \
                    "${run_dir}/metrics/metrics_${test_data}_${guide_epoch}_ema_map_max.csv"

            done
        done
    done
done
