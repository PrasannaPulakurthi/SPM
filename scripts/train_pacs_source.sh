SRC_DOMAIN=$1
TGT_DOMAIN=$2
SRC_MODEL_DIR=$3
PORT=${4:-10000}

MEMO="source"
SEED="2022"

export CUBLAS_WORKSPACE_CONFIG=:4096:8

python main.py \
seed=${SEED} port=${PORT} memo=${MEMO} project="PACS" \
learn.epochs=20 \
data.batch_size=16 \
data.aug_type=${AUG} \
data.data_root="datasets" data.workers=8 \
data.dataset="PACS" \
data.source_domains="[${SRC_DOMAIN}]" data.target_domains="[photo,art_painting,cartoon,sketch]" \
model_src.arch="resnet18" \
optim.lr=2e-4
