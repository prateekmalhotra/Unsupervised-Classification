python3 moco.py --config_env configs/env.yml --config_exp configs/pretext/moco_custom.yml
python3 scan.py --config_env configs/env.yml --config_exp configs/scan/scan_custom.yml
python3 selflabel.py --config_env configs/env.yml --config_exp configs/selflabel/selflabel_custom.yml
python3 eval.py --config_exp configs/selflabel/selflabel_custom.yml --model results/custom/selflabel/model.pth.tar
