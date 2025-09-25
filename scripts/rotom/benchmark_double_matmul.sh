mkdir -p logs/rotom/double_matmul

# double-matmul-128-64
for run in {1..5}; do
python main.py --n 8192 --backend ckks --serialize --rolls --microbenchmark double_matmul_128_64_micro > logs/rotom/double_matmul/double_matmul_128_64_$run.txt
done 

# double-matmul-256-128
for run in {1..5}; do
python main.py --n 32768 --backend ckks --serialize --rolls --microbenchmark double_matmul_256_128_micro > logs/rotom/double_matmul/double_matmul_256_128_$run.txt
done 
