mkdir -p logs/rotom/distance

# double-matmul-128-64
for run in {1..5}; do
python main.py --n 8192 --backend ckks --serialize --rolls --benchmark distance > logs/rotom/distance/distance_$run.txt
done 