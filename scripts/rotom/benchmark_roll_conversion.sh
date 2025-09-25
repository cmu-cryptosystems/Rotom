mkdir -p logs/roll_conversions/

# conversion
for run in {1..5}; do
python main.py --n 4096 --size 64 --backend ckks --microbenchmark conversion > logs/roll_conversions/conversion_4096_$run.txt
done 

# # conversion
# for run in {1..5}; do
# python main.py --n 16384 --size 128 --backend ckks --microbenchmark conversion > logs/roll_conversions/conversion_16384_$run.txt
# done 

# roll
for run in {1..5}; do
python main.py --n 4096 --size 64 --backend ckks --microbenchmark roll > logs/roll_conversions/roll_4096_$run.txt
done 

# # roll
# for run in {1..5}; do
# python main.py --n 16384 --size 128 --backend ckks --microbenchmark roll > logs/roll_conversions/roll_16384_$run.txt
# done 

# roll
for run in {1..5}; do
python main.py --n 4096 --size 64 --backend ckks --microbenchmark rot_roll > logs/roll_conversions/rot_roll_4096_$run.txt
done 

# # roll
# for run in {1..5}; do
# python main.py --n 16384 --size 128 --backend ckks --microbenchmark rot_roll > logs/roll_conversions/rot_roll_16384_$run.txt
# done 
