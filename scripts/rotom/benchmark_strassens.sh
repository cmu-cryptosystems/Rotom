mkdir -p logs/rotom/strassens

for run in {1..5}; do
python main.py --n 4096 --backend ckks --serialize --rolls --benchmark strassens > logs/rotom/strassens/strassens_$run.txt
done 

for run in {1..5}; do
python main.py --n 4096 --backend ckks --serialize --rolls --benchmark matmul_128_128 > logs/rotom/strassens/matmul_128_128_$run.txt
done 

