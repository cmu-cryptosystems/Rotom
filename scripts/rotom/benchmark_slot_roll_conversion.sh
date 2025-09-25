mkdir -p logs/slot_roll_conversions/
mkdir -p logs/slot_roll_conversions/

# conversion
for run in {1..5}; do
python main.py --n 4096 --size 64 --backend ckks --microbenchmark slot_conversion > logs/slot_roll_conversions/slot_conversion_4096_$run.txt
done 

# # conversion
# for run in {1..5}; do
# python main.py --n 16384 --size 128 --backend ckks --microbenchmark slot_conversion > logs/slot_roll_conversions/slot_conversion_16384_$run.txt
# done 

# roll
for run in {1..5}; do
python main.py --n 4096 --size 64 --backend ckks --microbenchmark slot_roll > logs/slot_roll_conversions/slot_roll_4096_$run.txt
done 

# # roll
# for run in {1..5}; do
# python main.py --n 16384 --size 128 --backend ckks --microbenchmark slot_roll > logs/slot_roll_conversions/slot_roll_16384_$run.txt
# done 

# roll
for run in {1..5}; do
python main.py --n 4096 --size 64 --backend ckks --microbenchmark slot_bsgs_roll > logs/slot_roll_conversions/slot_bsgs_roll_4096_$run.txt
done 

# # roll
# for run in {1..5}; do
# python main.py --n 16384 --size 128 --backend ckks --microbenchmark slot_bsgs_roll > logs/slot_roll_conversions/slot_bsgs_roll_16384_$run.txt
# done 
