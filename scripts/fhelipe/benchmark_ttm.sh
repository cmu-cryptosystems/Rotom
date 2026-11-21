mkdir -p logs/fhelipe/ttm/

for run in {1..5}; do
python main.py --fhelipe benchmarks/fhelipe_benchmarks/ttm_8192 --n 8192 --backend ckks > logs/fhelipe/ttm/ttm_8192_$run.txt
done 

for run in {1..5}; do
python main.py --fhelipe benchmarks/fhelipe_benchmarks/ttm_32768 --n 32768 --backend ckks > logs/fhelipe/ttm/ttm_32768_$run.txt
done 

