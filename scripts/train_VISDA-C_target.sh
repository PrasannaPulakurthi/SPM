SRC_MODEL_DIR=$1
PORT=$2

MEMO="target"
SEED="2022"

export CUBLAS_WORKSPACE_CONFIG=:4096:8

python main.py \
seed=${SEED} port=${PORT} memo=${MEMO} project="VISDA-C" \
learn.epochs=50 \
learn.num_neighbors=3 \
learn.mix_prob=0.8 \
learn.patch_height=56 \
learn.alpha_spm=4.0 \
data.batch_size=128 \
data.aug_type="shuffle_patch_mix_o_all" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir=${SRC_MODEL_DIR} \
optim.lr=2e-4
