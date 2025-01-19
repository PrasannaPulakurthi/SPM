set CUBLAS_WORKSPACE_CONFIG=:4096:8

python main_adacontrast_win.py train_source=true learn=source seed=2022 port=10000 memo="source" project="domainnet-126" data.data_root="datasets" data.workers=4 data.dataset="domainnet-126" data.source_domains="[real]" data.target_domains="[real,sketch,clipart,painting]" learn.epochs=60 model_src.arch="resnet50" optim.lr=2e-4

python main_adacontrast_win.py train_source=true learn=source seed=2022 port=10000 memo="source" project="domainnet-126" data.data_root="datasets" data.workers=4 data.dataset="domainnet-126" data.source_domains="[sketch]" data.target_domains="[real,sketch,clipart,painting]" learn.epochs=60 model_src.arch="resnet50" optim.lr=2e-4

python main_adacontrast_win.py train_source=true learn=source seed=2022 port=10000 memo="source" project="domainnet-126" data.data_root="datasets" data.workers=4 data.dataset="domainnet-126" data.source_domains="[clipart]" data.target_domains="[real,sketch,clipart,painting]" learn.epochs=60 model_src.arch="resnet50" optim.lr=2e-4

python main_adacontrast_win.py train_source=true learn=source seed=2022 port=10000 memo="source" project="domainnet-126" data.data_root="datasets" data.workers=4 data.dataset="domainnet-126" data.source_domains="[painting]" data.target_domains="[real,sketch,clipart,painting]" learn.epochs=60 model_src.arch="resnet50" optim.lr=2e-4
