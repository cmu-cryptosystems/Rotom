mkdir -p logs/viaduct/logreg/

# logreg-8192-e1
for run in {1..5}; do
python main.py --viaduct benchmarks/viaduct_benchmarks/logreg_e1_o0.vhe --n 8192 --backend ckks > logs/viaduct/logreg/logreg_8192_e1_$run.txt
done 

# logreg-8192-e2
for run in {1..5}; do
python main.py --viaduct benchmarks/viaduct_benchmarks/logreg_e2_o1.vhe --n 8192 --backend ckks > logs/viaduct/logreg/logreg_8192_e2_$run.txt
done 

# logreg-32768-e1
for run in {1..5}; do
python main.py --viaduct benchmarks/viaduct_benchmarks/logreg_e1_o0_32768.vhe --n 32768 --backend ckks > logs/viaduct/logreg/logreg_32768_e1_$run.txt
done 

# logreg-32768-e2
for run in {1..5}; do
python main.py --viaduct benchmarks/viaduct_benchmarks/logreg_e2_o1_32768.vhe --n 32768 --backend ckks > logs/viaduct/logreg/logreg_32768_e2_$run.txt
done 

