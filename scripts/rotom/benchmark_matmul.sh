mkdir -p logs/rotom/matmul

# matmul-128-64
for run in {1..5}; do
python main.py --n 8192 --backend ckks --rolls --benchmark matmul_128_64 --not-secure > logs/rotom/matmul/matmul_128_64_$run.txt
done 

# matmul-256-128
for run in {1..5}; do
python main.py --n 32768 --backend ckks --rolls --benchmark matmul_256_128 > logs/rotom/matmul/matmul_256_128_$run.txt
done 