# param1 = config name; param2 model dir; param3 output pkl dir
./tools/dist_test.sh $1 $2 1 --out $3
python widerface_evaluate/convert_anno.py $1
cd widerface_evaluate
python evaluation.py
cd ..