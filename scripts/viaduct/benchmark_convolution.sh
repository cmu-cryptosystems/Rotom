mkdir -p logs/viaduct/convolution/

# convolution-8192-e1
for run in {1..5}; do
python main.py --viaduct benchmarks/viaduct_benchmarks/convolution_e1_o0.vhe --n 8192 --backend ckks > logs/viaduct/convolution/convolution_8192_e1_$run.txt
done 

# convolution-8192-e2
for run in {1..5}; do
python main.py --viaduct benchmarks/viaduct_benchmarks/convolution_e2_o1.vhe --n 8192 --backend ckks > logs/viaduct/convolution/convolution_8192_e2_$run.txt
done 

# convolution-32768-e1
for run in {1..5}; do
python main.py --viaduct benchmarks/viaduct_benchmarks/convolution_e1_o0_32768.vhe --n 32768 --backend ckks > logs/viaduct/convolution/convolution_32768_e1_$run.txt
done 

# convolution-32768-e2
for run in {1..5}; do
python main.py --viaduct benchmarks/viaduct_benchmarks/convolution_e2_o1_32768.vhe --n 32768 --backend ckks > logs/viaduct/convolution/convolution_32768_e2_$run.txt
done 

