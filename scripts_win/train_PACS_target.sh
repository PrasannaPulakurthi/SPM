set CUBLAS_WORKSPACE_CONFIG=:4096:8

# P to A
python main_adacontrast_win.py seed=2022 port=10001 memo="target" project="PACS" learn.epochs=100 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_o_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[photo]" data.target_domains="[art_painting]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true learn.reweighting=true

# P to C
python main_adacontrast_win.py seed=2022 port=10002 memo="target" project="PACS" learn.epochs=100 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_o_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[photo]" data.target_domains="[cartoon]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true learn.reweighting=true

# P to S
python main_adacontrast_win.py seed=2022 port=10003 memo="target" project="PACS" learn.epochs=100 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_o_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[photo]" data.target_domains="[sketch]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true learn.reweighting=true

# A to P
python main_adacontrast_win.py seed=2022 port=10004 memo="target" project="PACS" learn.epochs=100 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_o_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[art_painting]" data.target_domains="[photo]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true learn.reweighting=true

# A to C
python main_adacontrast_win.py seed=2022 port=10005 memo="target" project="PACS" learn.epochs=100 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_o_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[art_painting]" data.target_domains="[cartoon]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true learn.reweighting=true

# A to S
python main_adacontrast_win.py seed=2022 port=10006 memo="target" project="PACS" learn.epochs=100 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_o_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[art_painting]" data.target_domains="[sketch]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true learn.reweighting=true

# P to ACS
python main_adacontrast_win.py seed=2022 port=10007 memo="target" project="PACS" learn.epochs=100 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_o_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[photo]" data.target_domains="[acs]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true learn.reweighting=true data.ttd=true data.test_target_domain="[art_painting,cartoon,sketch]"

# A to PCS
python main_adacontrast_win.py seed=2022 port=10008 memo="target" project="PACS" learn.epochs=100 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_o_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[art_painting]" data.target_domains="[pcs]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true learn.reweighting=true data.ttd=true data.test_target_domain="[photo,cartoon,sketch]"



# P to A
# SPM
python main_adacontrast_win.py seed=2022 port=10001 memo="target" project="PACS" learn.epochs=50 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[photo]" data.target_domains="[art_painting]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4

# SPM Change Alpha
python main_adacontrast_win.py seed=2022 port=10001 memo="target" project="PACS" learn.epochs=50 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[photo]" data.target_domains="[art_painting]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true

# SPM_all Change Alpha
python main_adacontrast_win.py seed=2022 port=10001 memo="target" project="PACS" learn.epochs=50 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[photo]" data.target_domains="[art_painting]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true

# SPM_l_all Change Alpha
python main_adacontrast_win.py seed=2022 port=10001 memo="target" project="PACS" learn.epochs=50 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_l_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[photo]" data.target_domains="[art_painting]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true

# SPM_o_l_all Change Alpha
python main_adacontrast_win.py seed=2022 port=10001 memo="target" project="PACS" learn.epochs=50 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_o_l_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[photo]" data.target_domains="[art_painting]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true


# A to C 
# SPM_o_l_all Change Alpha
python main_adacontrast_win.py seed=2022 port=10005 memo="target" project="PACS" learn.epochs=30 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_o_l_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[art_painting]" data.target_domains="[cartoon]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true

# SPM_o_l_all Change Alpha reweight
python main_adacontrast_win.py seed=2022 port=10005 memo="target" project="PACS" learn.epochs=30 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_o_l_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[art_painting]" data.target_domains="[cartoon]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true learn.reweighting=true

# SPM_o_l_all Change Alpha reweight Lambda_finetune
python main_adacontrast_win.py seed=2022 port=10005 memo="target" project="PACS" learn.epochs=30 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_o_l_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[art_painting]" data.target_domains="[cartoon]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true learn.reweighting=true learn.lambda_cls=2 learn.lambda_ins=0.5

# SPM_o_l_all Change Alpha reweight Lambda_finetune
python main_adacontrast_win.py seed=2022 port=10005 memo="target" project="PACS" learn.epochs=50 learn.alpha_spm=8.0 learn.alpha_spm_end=3.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_o_l_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[art_painting]" data.target_domains="[cartoon]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true learn.reweighting=true learn.lambda_cls=2


# SPM_all + change alpha + reweight
python main_adacontrast_win.py seed=2022 port=10005 memo="target" project="PACS" learn.epochs=50 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[art_painting]" data.target_domains="[cartoon]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true learn.reweighting=true

# SPM_o_all + change alpha + reweight
python main_adacontrast_win.py seed=2022 port=10005 memo="target" project="PACS" learn.epochs=50 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_o_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[art_painting]" data.target_domains="[cartoon]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true learn.reweighting=true

# SPM_o_l_all + change alpha + reweight
python main_adacontrast_win.py seed=2022 port=10005 memo="target" project="PACS" learn.epochs=50 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_o_l_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[art_painting]" data.target_domains="[cartoon]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true learn.reweighting=true


# P to C 
# SPM_all
python main_adacontrast_win.py seed=2022 port=10002 memo="target" project="PACS" learn.epochs=50 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[photo]" data.target_domains="[cartoon]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true learn.reweighting=true

# SPM_o_all
python main_adacontrast_win.py seed=2022 port=10002 memo="target" project="PACS" learn.epochs=50 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_o_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[photo]" data.target_domains="[cartoon]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true learn.reweighting=true

# SPM_o_l_all
python main_adacontrast_win.py seed=2022 port=10002 memo="target" project="PACS" learn.epochs=50 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_o_l_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[photo]" data.target_domains="[cartoon]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true learn.reweighting=true



# SPM_all [7, 14, 28, 56, 112]          75.63%
python main_adacontrast_win.py seed=2022 port=10002 memo="target" project="PACS" learn.epochs=50 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[photo]" data.target_domains="[cartoon]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true learn.reweighting=true

# SPM_all [14, 28, 56, 112]             75.93%
python main_adacontrast_win.py seed=2022 port=10002 memo="target" project="PACS" learn.epochs=50 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[photo]" data.target_domains="[cartoon]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true learn.reweighting=true

# SPM_all [28, 56, 112]                 75.65%
python main_adacontrast_win.py seed=2022 port=10002 memo="target" project="PACS" learn.epochs=50 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[photo]" data.target_domains="[cartoon]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true learn.reweighting=true

# SPM_pre_all                           75.68%
python main_adacontrast_win.py seed=2022 port=10002 memo="target" project="PACS" learn.epochs=50 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_pre_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[photo]" data.target_domains="[cartoon]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true learn.reweighting=true


#
set CUBLAS_WORKSPACE_CONFIG=:4096:8

# SPM_all [14, 28, 56, 112]             76.07%
python main_adacontrast_win.py seed=2022 port=10002 memo="target" project="PACS" learn.epochs=50 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[photo]" data.target_domains="[cartoon]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true learn.reweighting=true

# SPM_l_all                             75.51%
python main_adacontrast_win.py seed=2022 port=10002 memo="target" project="PACS" learn.epochs=50 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_l_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[photo]" data.target_domains="[cartoon]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true learn.reweighting=true

# SPM_o_all                             76.19%
python main_adacontrast_win.py seed=2022 port=10002 memo="target" project="PACS" learn.epochs=50 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_o_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[photo]" data.target_domains="[cartoon]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true learn.reweighting=true

# SPM_o_l_all                           75.60%
python main_adacontrast_win.py seed=2022 port=10002 memo="target" project="PACS" learn.epochs=50 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_o_l_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[photo]" data.target_domains="[cartoon]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true learn.reweighting=true

# SPM_o_all   60 epochs                 76.57%
python main_adacontrast_win.py seed=2022 port=10002 memo="target" project="PACS" learn.epochs=60 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_o_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[photo]" data.target_domains="[cartoon]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true learn.reweighting=true

# SPM_o_all   80 epochs       8 to 2 (0 to 60)   same as epochs 70 improvemt seen before making contsant spm_end      77.52%
python main_adacontrast_win.py seed=2022 port=10002 memo="target" project="PACS" learn.epochs=80 learn.alpha_spm=8.0 learn.alpha_spm_end=1.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_o_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[photo]" data.target_domains="[cartoon]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true learn.reweighting=true

# SPM_o_all   100 epochs                77.82%
python main_adacontrast_win.py seed=2022 port=10002 memo="target" project="PACS" learn.epochs=100 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_o_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[photo]" data.target_domains="[cartoon]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true learn.reweighting=true

# SPM_o_all confidance_margin_reweighting   100 epochs                81.48%
python main_adacontrast_win.py seed=2022 port=10002 memo="target" project="PACS" learn.epochs=100 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_o_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[photo]" data.target_domains="[cartoon]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true learn.reweighting=true

# P to S
python main_adacontrast_win.py seed=2022 port=10003 memo="target" project="PACS" learn.epochs=100 learn.alpha_spm=8.0 learn.beta_spm=2.0 learn.patch_height=56 learn.num_neighbors=3 data.aug_type="shuffle_patch_mix_o_all" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[photo]" data.target_domains="[sketch]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 learn.change_alpha=true learn.reweighting=true
