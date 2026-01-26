mkdir -p logs/rotom/convolution

# double-matmul-128-64
for run in {1..5}; do
python main.py --n 8192 --backend ckks --rolls --benchmark convolution > logs/rotom/convolution/convolution_8192_$run.txt
done 

# double-matmul-256-128
for run in {1..5}; do
python main.py --n 32768 --backend ckks --rolls --benchmark convolution_32768 > logs/rotom/convolution/convolution_32768_$run.txt
done