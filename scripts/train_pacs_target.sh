SRC_DOMAIN=$1
TGT_DOMAIN=$2
SRC_MODEL_DIR=$3
PORT=$4

MEMO="target"
SEED="2022"

export CUBLAS_WORKSPACE_CONFIG=:4096:8

python main.py \
seed=${SEED} port=${PORT} memo=${MEMO} project="PACS" \
learn.epochs=100 \
learn.num_neighbors=3 \
learn.mix_prob=0.8 \
learn.patch_height=56 \
learn.alpha_spm=8.0 \
learn.reweighting=true \
data.batch_size=128 \
data.aug_type="shuffle_patch_mix_o_all" \
data.data_root="datasets" data.workers=8 \
data.dataset="PACS" data.source_domains="[${SRC_DOMAIN}]" data.target_domains="[${TGT_DOMAIN}]" \
model_src.arch="resnet18" \
model_tta.src_log_dir=${SRC_MODEL_DIR} \
data.ttd=false \
data.test_target_domain="[photo,art_painting,cartoon,sketch]" \
optim.lr=2e-4
