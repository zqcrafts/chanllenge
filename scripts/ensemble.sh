./tools/dist_test.sh widerface_cfg/$1_500.py work_dirs/$1/epoch_600.pth 6 --out work_dirs/$1/result_500.pkl
./tools/dist_test.sh widerface_cfg/$1_800.py work_dirs/$1/epoch_600.pth 6 --out work_dirs/$1/result_800.pkl
./tools/dist_test.sh widerface_cfg/$1_1100.py work_dirs/$1/epoch_600.pth 6 --out work_dirs/$1/result_1100.pkl
./tools/dist_test.sh widerface_cfg/$1_1400.py work_dirs/$1/epoch_600.pth 6 --out work_dirs/$1/result_1400.pkl
./tools/dist_test.sh widerface_cfg/$1_1700.py work_dirs/$1/epoch_600.pth 6 --out work_dirs/$1/result_1700.pkl

python widerface_evaluate/group_ensemble_all.py widerface_cfg/$1.py
cd widerface_evaluate
python evaluation.py
cd ..