SRC_DOMAIN=$1
TGT_DOMAIN=$2
SRC_MODEL_DIR=$3
PORT=$4
RW=$5
NN=$6
C=$7
PS=$8
AUG=$9
MIXPROB=${10}
NEGAUG=${11}
CHG_ALP=${12}
ALPHASTART=${13}
RW_T=${14}

MEMO="target"
SEED="2022"

export CUBLAS_WORKSPACE_CONFIG=:4096:8

python main_adacontrast.py \
seed=${SEED} port=${PORT} memo=${MEMO} project="PACS" \
learn.epochs=100 \
learn.num_neighbors=${NN} \
learn.mix_prob=${MIXPROB} \
learn.patch_height=${PS} \
learn.alpha_spm=${ALPHASTART} \
learn.reweighting=${RW} \
learn.reweighting_type=${RW_T} \
learn.negative_aug=${NEGAUG} \
learn.change_alpha=${CHG_ALP} \
data.batch_size=128 \
data.aug_type=${AUG} \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="PACS" data.source_domains="[${SRC_DOMAIN}]" data.target_domains="[${TGT_DOMAIN}]" \
model_src.arch="resnet18" \
model_tta.src_log_dir=${SRC_MODEL_DIR} \
data.ttd=false \
data.test_target_domain="[photo,art_painting,cartoon,sketch]" \
optim.lr=2e-4
