mkdir -p logs/fhelipe/convolution/

for run in {1..5}; do
python main.py --fhelipe benchmarks/fhelipe_benchmarks/convolution --n 8192 --backend ckks > logs/fhelipe/convolution/convolution_8192_$run.txt
done 

for run in {1..5}; do
python main.py --fhelipe benchmarks/fhelipe_benchmarks/convolution_32768 --n 32768 --backend ckks > logs/fhelipe/convolution/convolution_32768_$run.txt
done 

