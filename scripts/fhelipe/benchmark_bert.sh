mkdir -p logs/fhelipe/bert/

for run in {1..5}; do
python main.py --fhelipe benchmarks/fhelipe_benchmarks/bert --n 8192 --backend ckks > logs/fhelipe/bert/bert_$run.txt
done 

for run in {1..5}; do
python main.py --fhelipe benchmarks/fhelipe_benchmarks/bert_32768 --n 32768 --backend ckks > logs/fhelipe/bert/bert_32768_$run.txt
done 

