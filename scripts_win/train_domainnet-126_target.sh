set CUBLAS_WORKSPACE_CONFIG=:4096:8

python main_adacontrast_win.py seed=2022 port=10001 memo="target" project="domainnet-126" learn.epochs=30 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=28 learn.num_neighbors=3 data.batch_size=64 data.aug_type="shuffle_patch_mix" data.data_root="datasets" data.workers=4 data.dataset="domainnet-126" data.source_domains="[real]" data.target_domains="[clipart]" model_src.arch="resnet50" model_tta.src_log_dir="output/domainnet-126/source" optim.lr=2e-4 learn.reweighting=true learn.c=0.25

python main_adacontrast_win.py seed=2022 port=10002 memo="target" project="domainnet-126" learn.epochs=30 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=28 learn.num_neighbors=3 data.batch_size=64 data.aug_type="shuffle_patch_mix" data.data_root="datasets" data.workers=8 data.dataset="domainnet-126" data.source_domains="[real]" data.target_domains="[painting]" model_src.arch="resnet50" model_tta.src_log_dir="output/domainnet-126/source" optim.lr=2e-4 learn.reweighting=true learn.c=0.25

python main_adacontrast_win.py seed=2022 port=10003 memo="target" project="domainnet-126" learn.epochs=30 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=28 learn.num_neighbors=3 data.batch_size=64 data.aug_type="shuffle_patch_mix" data.data_root="datasets" data.workers=4 data.dataset="domainnet-126" data.source_domains="[painting]" data.target_domains="[clipart]" model_src.arch="resnet50" model_tta.src_log_dir="output/domainnet-126/source" optim.lr=2e-4 learn.reweighting=true learn.c=0.25

python main_adacontrast_win.py seed=2022 port=10004 memo="target" project="domainnet-126" learn.epochs=30 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=28 learn.num_neighbors=3 data.batch_size=64 data.aug_type="shuffle_patch_mix" data.data_root="datasets" data.workers=4 data.dataset="domainnet-126" data.source_domains="[clipart]" data.target_domains="[sketch]" model_src.arch="resnet50" model_tta.src_log_dir="output/domainnet-126/source" optim.lr=2e-4 learn.reweighting=true learn.c=0.25 model_src.act="PMish"

python main_adacontrast_win.py seed=2022 port=10005 memo="target" project="domainnet-126" learn.epochs=30 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=112 learn.num_neighbors=3 data.batch_size=64 data.aug_type="shuffle_patch_mix" data.data_root="datasets" data.workers=4 data.dataset="domainnet-126" data.source_domains="[sketch]" data.target_domains="[painting]" model_src.arch="resnet50" model_tta.src_log_dir="output/domainnet-126/source" optim.lr=2e-4 learn.reweighting=true learn.c=0.25

python main_adacontrast_win.py seed=2022 port=10006 memo="target" project="domainnet-126" learn.epochs=30 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=28 learn.num_neighbors=3 data.batch_size=64 data.aug_type="shuffle_patch_mix" data.data_root="datasets" data.workers=4 data.dataset="domainnet-126" data.source_domains="[real]" data.target_domains="[sketch]" model_src.arch="resnet50" model_tta.src_log_dir="output/domainnet-126/source" optim.lr=2e-4 learn.reweighting=true learn.c=0.25

python main_adacontrast_win.py seed=2022 port=10007 memo="target" project="domainnet-126" learn.epochs=30 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=28 learn.num_neighbors=3 data.batch_size=64 data.aug_type="shuffle_patch_mix" data.data_root="datasets" data.workers=4 data.dataset="domainnet-126" data.source_domains="[painting]" data.target_domains="[real]" model_src.arch="resnet50" model_tta.src_log_dir="output/domainnet-126/source" optim.lr=2e-4 learn.reweighting=true learn.c=0.25



# Adacontrast
python main_adacontrast_win.py seed=2022 port=10002 memo="target" project="domainnet-126" learn.epochs=5 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=28 learn.num_neighbors=10 data.batch_size=64 data.aug_type="moco-v2" data.data_root="datasets" data.workers=8 data.dataset="domainnet-126" data.source_domains="[real]" data.target_domains="[painting]" model_src.arch="resnet50" model_tta.src_log_dir="output/domainnet-126/source" optim.lr=2e-4

# SPM (ICASSP)
python main_adacontrast_win.py seed=2022 port=10002 memo="target" project="domainnet-126" learn.epochs=5 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=28 learn.num_neighbors=3 data.batch_size=64 data.aug_type="shuffle_patch_mix" data.data_root="datasets" data.workers=8 data.dataset="domainnet-126" data.source_domains="[real]" data.target_domains="[painting]" model_src.arch="resnet50" model_tta.src_log_dir="output/domainnet-126/source" optim.lr=2e-4

# SPM_l (ICASSP)
python main_adacontrast_win.py seed=2022 port=10002 memo="target" project="domainnet-126" learn.epochs=5 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=28 learn.num_neighbors=3 data.batch_size=64 data.aug_type="shuffle_patch_mix_l" data.data_root="datasets" data.workers=8 data.dataset="domainnet-126" data.source_domains="[real]" data.target_domains="[painting]" model_src.arch="resnet50" model_tta.src_log_dir="output/domainnet-126/source" optim.lr=2e-4

# SPM_l + Pseudo-label reweighting 
python main_adacontrast_win.py seed=2022 port=10002 memo="target" project="domainnet-126" learn.epochs=5 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=28 learn.num_neighbors=3 data.batch_size=64 data.aug_type="shuffle_patch_mix_l" data.data_root="datasets" data.workers=8 data.dataset="domainnet-126" data.source_domains="[real]" data.target_domains="[painting]" model_src.arch="resnet50" model_tta.src_log_dir="output/domainnet-126/source" optim.lr=2e-4 learn.reweighting=true learn.c=0.25

# SPM_l + increase alpha (BAD)
python main_adacontrast_win.py seed=2022 port=10002 memo="target" project="domainnet-126" learn.epochs=5 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=28 learn.num_neighbors=3 data.batch_size=64 data.aug_type="shuffle_patch_mix_l" data.data_root="datasets" data.workers=8 data.dataset="domainnet-126" data.source_domains="[real]" data.target_domains="[painting]" model_src.arch="resnet50" model_tta.src_log_dir="output/domainnet-126/source" optim.lr=2e-4 learn.change_alpha=true

# SPM_l + Pseudo-label reweighting + Neg augmentation
python main_adacontrast_win.py seed=2022 port=10002 memo="target" project="domainnet-126" learn.epochs=5 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=28 learn.num_neighbors=3 data.batch_size=64 data.aug_type="shuffle_patch_mix_l" data.data_root="datasets" data.workers=8 data.dataset="domainnet-126" data.source_domains="[real]" data.target_domains="[painting]" model_src.arch="resnet50" model_tta.src_log_dir="output/domainnet-126/source" optim.lr=2e-4 learn.reweighting=true learn.c=0.25 learn.negative_aug=true

# SPM_l + Pseudo-label reweighting + Neg augmentation + MCC
python main_adacontrast_win.py seed=2022 port=10002 memo="target" project="domainnet-126" learn.epochs=5 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=28 learn.num_neighbors=3 data.batch_size=64 data.aug_type="shuffle_patch_mix_l" data.data_root="datasets" data.workers=8 data.dataset="domainnet-126" data.source_domains="[real]" data.target_domains="[painting]" model_src.arch="resnet50" model_tta.src_log_dir="output/domainnet-126/source" optim.lr=2e-4 learn.reweighting=true learn.c=0.25 learn.negative_aug=true learn.mcc=true

# SPM_o_l + Pseudo-label reweighting + Neg augmentation + MCC
python main_adacontrast_win.py seed=2022 port=10002 memo="target" project="domainnet-126" learn.epochs=5 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=28 learn.num_neighbors=3 data.batch_size=64 data.aug_type="shuffle_patch_mix_o_l" data.data_root="datasets" data.workers=8 data.dataset="domainnet-126" data.source_domains="[real]" data.target_domains="[painting]" model_src.arch="resnet50" model_tta.src_log_dir="output/domainnet-126/source" optim.lr=2e-4 learn.reweighting=true learn.c=0.25 learn.negative_aug=true learn.mcc=true


# SPM + Pseudo-label
python main_adacontrast_win.py seed=2022 port=10002 memo="target" project="domainnet-126" learn.epochs=5 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=28 learn.num_neighbors=3 data.batch_size=64 data.aug_type="shuffle_patch_mix" data.data_root="datasets" data.workers=8 data.dataset="domainnet-126" data.source_domains="[real]" data.target_domains="[painting]" model_src.arch="resnet50" model_tta.src_log_dir="output/domainnet-126/source" optim.lr=2e-4 learn.reweighting=true
