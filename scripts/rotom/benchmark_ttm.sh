mkdir -p logs/rotom/ttm

for run in {1..5}; do
python main.py --backend ckks --rolls --benchmark ttm --n 8192 --not-secure > logs/rotom/ttm/ttm_8192_$run.txt
done 

for run in {1..5}; do
python main.py --backend ckks --rolls --benchmark ttm_32 --n 32768 > logs/rotom/ttm/ttm_32768_$run.txt
done 