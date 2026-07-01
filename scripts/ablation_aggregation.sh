#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "${REPO_ROOT}"

train_data_list=(mvtec visa)
test_data_list=("visa" "mvtec")

sae_epoch=50
sae_hidden_dim=4096
sae_topk=32
image_size=448
input_norm=none

guide_epoch=15

# Set each RUNS array entry to the actual run_dir name
# (e.g. outputs/202X-XX-XX/XX-XX-XX_guide_sae_facebook_dinov3-vitl16).
# The examples below cover mvtec->visa and visa->mvtec.
# If you rerun a model or experiment, replace these with the newly generated directory names.
RUNS=(
  "outputs/2026-02-20/13-57-36_guide_sae_facebook_dinov3-vitl16" # mvtec->visa
  "outputs/2026-02-20/14-39-38_guide_sae_facebook_dinov3-vitl16" # visa->mvtec
)

aggregation="log_sum_exp"
for tau in 0.001 0.005 0.01 0.05 0.1 0.5 1.0 5.0 10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0 90.0 100.0; do
    run_dir=${RUNS[0]}
    test_data="visa"
    python run.py mode=eval data@test_data=single_dataset/${test_data} train_dir=${run_dir} evaluate.epoch=${guide_epoch} \
        evaluate.use_ema=true \
        evaluate.image_score.mode="map" \
        evaluate.image_score.map_pool=${aggregation} \
        evaluate.image_score.map_pool_tau=${tau} \
        evaluate.pro_use_fast=true
    mv "${run_dir}/metrics/metrics_${test_data}_${guide_epoch}_ema.csv" \
        "${run_dir}/metrics/metrics_${test_data}_${guide_epoch}_ema_map_${aggregation}_tau_${tau}.csv"
    
    run_dir=${RUNS[1]}
    test_data="mvtec"
    python run.py mode=eval data@test_data=single_dataset/${test_data} train_dir=${run_dir} evaluate.epoch=${guide_epoch} \
        evaluate.use_ema=true \
        evaluate.image_score.mode="map" \
        evaluate.image_score.map_pool=${aggregation} \
        evaluate.image_score.map_pool_tau=${tau} \
        evaluate.pro_use_fast=true
    mv "${run_dir}/metrics/metrics_${test_data}_${guide_epoch}_ema.csv" \
        "${run_dir}/metrics/metrics_${test_data}_${guide_epoch}_ema_map_${aggregation}_tau_${tau}.csv"
done
