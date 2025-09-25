mkdir -p logs/rotom/bert

# double-matmul-128-64
for run in {1..5}; do
python main.py --n 8192 --backend ckks --serialize --rolls --mock --benchmark bert_attention > logs/rotom/bert/bert_8192_$run.txt
done 

# double-matmul-256-128
for run in {1..5}; do
python main.py --n 32768 --backend ckks --serialize --rolls --mock --benchmark bert_attention_32768 > logs/rotom/bert/bert_32768_$run.txt
done 
