mkdir -p logs/rotom/strassens

for run in {1..5}; do
  python main.py --n 4096 --backend ckks --rolls --benchmark strassens --strassens --not-secure > logs/rotom/strassens/strassens_roll_$run.txt
done 

for run in {1..5}; do
  python main.py --n 4096 --backend ckks --benchmark strassens --strassens --not-secure > logs/rotom/strassens/strassens_no_roll_$run.txt
done 


for run in {1..5}; do
  python main.py --n 4096 --backend ckks --benchmark strassens --not-secure > logs/rotom/strassens/no_strassens_no_roll_$run.txt 
done

