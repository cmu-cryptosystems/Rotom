mkdir -p logs/rotom/logreg

# double-matmul-128-64
for run in {1..5}; do
python main.py --n 8192 --backend ckks --serialize --mock --rolls --benchmark logreg > logs/rotom/logreg/logreg_8192_$run.txt
done 

# double-matmul-256-128
for run in {1..5}; do
python main.py --n 32768 --backend ckks --serialize --mock --rolls --benchmark logreg > logs/rotom/logreg/logreg_32768_$run.txt
done