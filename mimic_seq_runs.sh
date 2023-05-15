#! /usr/bin/env bash
#$ -S /bin/bash  # run job as a Bash shell [IMPORTANT]
#$ -cwd          # run job in the current working directory
# mimic has 35k rows

xargs -L1 -P16 -I__ sh -c __ <<EOF
python run_sequential.py --data mimic --model lr -n 100
python run_sequential.py --data mimic --model lr -n 200
python run_sequential.py --data mimic --model lr -n 300
python run_sequential.py --data mimic --model lr -n 400
python run_sequential.py --data mimic --model lr -n 500
python run_sequential.py --data mimic --model lr -n 600
python run_sequential.py --data mimic --model lr -n 700
python run_sequential.py --data mimic --model lr -n 800
python run_sequential.py --data mimic --model lr -n 900
python run_sequential.py --data mimic --model lr -n 1000
EOF