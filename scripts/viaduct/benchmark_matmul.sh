mkdir -p logs/viaduct/matmul/

# matmul-8192-e1
for run in {1..5}; do
python main.py --viaduct benchmarks/viaduct_benchmarks/matmul_128_64_e1_o0.vhe --n 8192 --backend ckks > logs/viaduct/matmul/matmul_8192_e1_$run.txt
done 

# matmul-8192-e2
for run in {1..5}; do
python main.py --viaduct benchmarks/viaduct_benchmarks/matmul_128_64_e2_o1.vhe --n 8192 --backend ckks > logs/viaduct/matmul/matmul_8192_e2_$run.txt
done 

# matmul-32768-e1
for run in {1..5}; do
python main.py --viaduct benchmarks/viaduct_benchmarks/matmul_256_128_e1_o0.vhe --n 32768 --backend ckks > logs/viaduct/matmul/matmul_32768_e1_$run.txt
done 