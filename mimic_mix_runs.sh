#! /usr/bin/env bash
#$ -S /bin/bash  # run job as a Bash shell [IMPORTANT]
#$ -cwd          # run job in the current working directory
# mimic has 25k rows

xargs -L1 -P16 -I__ sh -c __ <<EOF
python run_mixture.py --data mimic --model lr --n1 1000 --n2 0 --n3 0
python run_mixture.py --data mimic --model lr --n1 800 --n2 100 --n3 100
python run_mixture.py --data mimic --model lr --n1 600 --n2 200 --n3 200
python run_mixture.py --data mimic --model lr --n1 400 --n2 300 --n3 300
python run_mixture.py --data mimic --model lr --n1 0 --n2 1000 --n3 0
python run_mixture.py --data mimic --model lr --n1 100 --n2 800 --n3 100
python run_mixture.py --data mimic --model lr --n1 200 --n2 600 --n3 200
python run_mixture.py --data mimic --model lr --n1 300 --n2 400 --n3 300
python run_mixture.py --data mimic --model lr --n1 0 --n2 0 --n3 1000
python run_mixture.py --data mimic --model lr --n1 100 --n2 100 --n3 800
python run_mixture.py --data mimic --model lr --n1 200 --n2 200 --n3 600
python run_mixture.py --data mimic --model lr --n1 300 --n2 300 --n3 400
python run_mixture.py --data mimic --model lr --n1 5000 --n2 0 --n3 0
python run_mixture.py --data mimic --model lr --n1 4000 --n2 500 --n3 500
python run_mixture.py --data mimic --model lr --n1 3000 --n2 1000 --n3 1000
python run_mixture.py --data mimic --model lr --n1 2000 --n2 1500 --n3 1500
python run_mixture.py --data mimic --model lr --n1 0 --n2 5000 --n3 0
python run_mixture.py --data mimic --model lr --n1 500 --n2 4000 --n3 500
python run_mixture.py --data mimic --model lr --n1 1000 --n2 3000 --n3 1000
python run_mixture.py --data mimic --model lr --n1 1500 --n2 2000 --n3 1500
python run_mixture.py --data mimic --model lr --n1 0 --n2 0 --n3 5000
python run_mixture.py --data mimic --model lr --n1 500 --n2 500 --n3 4000
python run_mixture.py --data mimic --model lr --n1 1000 --n2 1000 --n3 3000
python run_mixture.py --data mimic --model lr --n1 1500 --n2 1500 --n3 2000
python run_mixture.py --data mimic --model lr --n1 10000 --n2 0 --n3 0
python run_mixture.py --data mimic --model lr --n1 8000 --n2 1000 --n3 1000
python run_mixture.py --data mimic --model lr --n1 6000 --n2 2000 --n3 2000
python run_mixture.py --data mimic --model lr --n1 4000 --n2 3000 --n3 3000
python run_mixture.py --data mimic --model lr --n1 0 --n2 10000 --n3 0
python run_mixture.py --data mimic --model lr --n1 1000 --n2 8000 --n3 1000
python run_mixture.py --data mimic --model lr --n1 2000 --n2 6000 --n3 2000
python run_mixture.py --data mimic --model lr --n1 3000 --n2 4000 --n3 3000
python run_mixture.py --data mimic --model lr --n1 0 --n2 0 --n3 10000
python run_mixture.py --data mimic --model lr --n1 1000 --n2 1000 --n3 8000
python run_mixture.py --data mimic --model lr --n1 2000 --n2 2000 --n3 6000
python run_mixture.py --data mimic --model lr --n1 3000 --n2 3000 --n3 4000
python run_mixture.py --data mimic --model lr --n1 150000 --n2 0 --n3 0
python run_mixture.py --data mimic --model lr --n1 120000 --n2 15000 --n3 15000
python run_mixture.py --data mimic --model lr --n1 90000 --n2 30000 --n3 30000
python run_mixture.py --data mimic --model lr --n1 60000 --n2 45000 --n3 45000
python run_mixture.py --data mimic --model lr --n1 0 --n2 150000 --n3 0
python run_mixture.py --data mimic --model lr --n1 15000 --n2 120000 --n3 15000
python run_mixture.py --data mimic --model lr --n1 30000 --n2 90000 --n3 30000
python run_mixture.py --data mimic --model lr --n1 45000 --n2 60000 --n3 45000
python run_mixture.py --data mimic --model lr --n1 0 --n2 0 --n3 150000
python run_mixture.py --data mimic --model lr --n1 15000 --n2 15000 --n3 120000
python run_mixture.py --data mimic --model lr --n1 30000 --n2 30000 --n3 90000
python run_mixture.py --data mimic --model lr --n1 45000 --n2 45000 --n3 60000
python run_mixture.py --data mimic --model lr --n1 20000 --n2 0 --n3 0
python run_mixture.py --data mimic --model lr --n1 16000 --n2 2000 --n3 2000
python run_mixture.py --data mimic --model lr --n1 12000 --n2 4000 --n3 4000
python run_mixture.py --data mimic --model lr --n1 8000 --n2 6000 --n3 6000
python run_mixture.py --data mimic --model lr --n1 0 --n2 20000 --n3 0
python run_mixture.py --data mimic --model lr --n1 2000 --n2 16000 --n3 2000
python run_mixture.py --data mimic --model lr --n1 4000 --n2 12000 --n3 4000
python run_mixture.py --data mimic --model lr --n1 6000 --n2 8000 --n3 6000
python run_mixture.py --data mimic --model lr --n1 0 --n2 0 --n3 20000
python run_mixture.py --data mimic --model lr --n1 2000 --n2 2000 --n3 16000
python run_mixture.py --data mimic --model lr --n1 4000 --n2 4000 --n3 12000
python run_mixture.py --data mimic --model lr --n1 6000 --n2 6000 --n3 8000
EOF
