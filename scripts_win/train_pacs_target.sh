set CUBLAS_WORKSPACE_CONFIG=:4096:8

# P to A
python main_win.py seed=2022 port=10001 memo="target" project="PACS" learn.epochs=100 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_o_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[photo]" data.target_domains="[art_painting]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true learn.reweighting=true

# P to C
python main_win.py seed=2022 port=10002 memo="target" project="PACS" learn.epochs=100 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_o_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[photo]" data.target_domains="[cartoon]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true learn.reweighting=true

# P to S
python main_win.py seed=2022 port=10003 memo="target" project="PACS" learn.epochs=100 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_o_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[photo]" data.target_domains="[sketch]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true learn.reweighting=true

# A to P
python main_win.py seed=2022 port=10004 memo="target" project="PACS" learn.epochs=50 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_o_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[art_painting]" data.target_domains="[photo]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true learn.reweighting=true

# A to C
python main_win.py seed=2022 port=10005 memo="target" project="PACS" learn.epochs=100 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_o_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[art_painting]" data.target_domains="[cartoon]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true learn.reweighting=true

# A to S
python main_win.py seed=2022 port=10006 memo="target" project="PACS" learn.epochs=100 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_o_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[art_painting]" data.target_domains="[sketch]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true learn.reweighting=true

# P to ACS
python main_win.py seed=2022 port=10007 memo="target" project="PACS" learn.epochs=100 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_o_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[photo]" data.target_domains="[acs]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true learn.reweighting=true data.ttd=true data.test_target_domain="[art_painting,cartoon,sketch]"

# A to PCS
python main_win.py seed=2022 port=10008 memo="target" project="PACS" learn.epochs=100 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_o_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[art_painting]" data.target_domains="[pcs]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true learn.reweighting=true data.ttd=true data.test_target_domain="[photo,cartoon,sketch]"
