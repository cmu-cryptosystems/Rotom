mkdir -p logs/fhelipe/logreg/

for run in {1..5}; do
python main.py --fhelipe benchmarks/fhelipe_benchmarks/logreg --n 8192 --backend ckks > logs/fhelipe/logreg/logreg_8192_$run.txt
done 

for run in {1..5}; do
python main.py --fhelipe benchmarks/fhelipe_benchmarks/logreg_32768 --n 32768 --backend ckks > logs/fhelipe/logreg/logreg_32768_$run.txt
done 

