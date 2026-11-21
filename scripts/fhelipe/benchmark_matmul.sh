mkdir -p logs/fhelipe/matmul/

for run in {1..5}; do
python main.py --fhelipe benchmarks/fhelipe_benchmarks/matmul_128_64 --n 8192 --backend ckks > logs/fhelipe/matmul/matmul_128_64_$run.txt
done 

for run in {1..5}; do
python main.py --fhelipe benchmarks/fhelipe_benchmarks/matmul_256_128 --n 32768 --backend ckks > logs/fhelipe/matmul/matmul_256_128_$run.txt
done 