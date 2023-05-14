#! /usr/bin/env bash
#$ -S /bin/bash  # run job as a Bash shell [IMPORTANT]
#$ -cwd          # run job in the current working directory

xargs -L1 -P16 -I__ sh -c __ <<EOF
# python run_mimic.py --n1 400 --n2 300 --n3 300 --model lr
# python run_mimic.py --n1 700 --n2 525 --n3 525 --model lr
# python run_mimic.py --n1 1000 --n2 750 --n3 750 --model lr
# python run_mimic.py --n1 1300 --n2 975 --n3 975 --model lr
# python run_mimic.py --n1 1600 --n2 1200 --n3 1200 --model lr
# python run_mimic.py --n1 1900 --n2 1425 --n3 1425 --model lr
# python run_mimic.py --n1 2200 --n2 1650 --n3 1650 --model lr
# python run_mimic.py --n1 2500 --n2 1875 --n3 1875 --model lr
# python run_mimic.py --n1 2800 --n2 2100 --n3 2100 --model lr
# python run_mimic.py --n1 3100 --n2 2325 --n3 2325 --model lr
# python run_mimic.py --n1 3400 --n2 2550 --n3 2550 --model lr
# python run_mimic.py --n1 3700 --n2 2775 --n3 2775 --model lr
# python run_mimic.py --n1 4000 --n2 3000 --n3 3000 --model lr
# python run_mimic.py --n1 4300 --n2 3225 --n3 3225 --model lr
# python run_mimic.py --n1 4600 --n2 3450 --n3 3450 --model lr
# python run_mimic.py --n1 4900 --n2 3675 --n3 3675 --model lr
# python run_mimic.py --n1 5200 --n2 3900 --n3 3900 --model lr
# python run_mimic.py --n1 5500 --n2 4125 --n3 4125 --model lr
# python run_mimic.py --n1 5800 --n2 4350 --n3 4350 --model lr
# python run_mimic.py --n1 6100 --n2 4575 --n3 4575 --model lr
# python run_mimic.py --n1 6400 --n2 4800 --n3 4800 --model lr
# python run_mimic.py --n1 6700 --n2 5025 --n3 5025 --model lr
# python run_mimic.py --n1 7000 --n2 5250 --n3 5250 --model lr
# python run_mimic.py --n1 7300 --n2 5475 --n3 5475 --model lr
# python run_mimic.py --n1 7600 --n2 5700 --n3 5700 --model lr
# python run_mimic.py --n1 7900 --n2 5925 --n3 5925 --model lr
# python run_mimic.py --n1 8200 --n2 6150 --n3 6150 --model lr

# python run_mimic.py --n1 400 --n2 300 --n3 300 --model svm
# python run_mimic.py --n1 700 --n2 525 --n3 525 --model svm
# python run_mimic.py --n1 1000 --n2 750 --n3 750 --model svm
# python run_mimic.py --n1 1300 --n2 975 --n3 975 --model svm
# python run_mimic.py --n1 1600 --n2 1200 --n3 1200 --model svm
# python run_mimic.py --n1 1900 --n2 1425 --n3 1425 --model svm
# python run_mimic.py --n1 2200 --n2 1650 --n3 1650 --model svm
# python run_mimic.py --n1 2500 --n2 1875 --n3 1875 --model svm
# python run_mimic.py --n1 2800 --n2 2100 --n3 2100 --model svm
# python run_mimic.py --n1 3100 --n2 2325 --n3 2325 --model svm
# python run_mimic.py --n1 3400 --n2 2550 --n3 2550 --model svm
# python run_mimic.py --n1 3700 --n2 2775 --n3 2775 --model svm
# python run_mimic.py --n1 4000 --n2 3000 --n3 3000 --model svm
# python run_mimic.py --n1 4300 --n2 3225 --n3 3225 --model svm
python run_mimic.py --n1 4600 --n2 3450 --n3 3450 --model svm
python run_mimic.py --n1 4900 --n2 3675 --n3 3675 --model svm
python run_mimic.py --n1 5200 --n2 3900 --n3 3900 --model svm
python run_mimic.py --n1 5500 --n2 4125 --n3 4125 --model svm
python run_mimic.py --n1 5800 --n2 4350 --n3 4350 --model svm
python run_mimic.py --n1 6100 --n2 4575 --n3 4575 --model svm
python run_mimic.py --n1 6400 --n2 4800 --n3 4800 --model svm
python run_mimic.py --n1 6700 --n2 5025 --n3 5025 --model svm
python run_mimic.py --n1 7000 --n2 5250 --n3 5250 --model svm
python run_mimic.py --n1 7300 --n2 5475 --n3 5475 --model svm
python run_mimic.py --n1 7600 --n2 5700 --n3 5700 --model svm
python run_mimic.py --n1 7900 --n2 5925 --n3 5925 --model svm
python run_mimic.py --n1 8200 --n2 6150 --n3 6150 --model svm

python run_mimic.py --n1 400 --n2 300 --n3 300 --model nn
python run_mimic.py --n1 700 --n2 525 --n3 525 --model nn
python run_mimic.py --n1 1000 --n2 750 --n3 750 --model nn
python run_mimic.py --n1 1300 --n2 975 --n3 975 --model nn
python run_mimic.py --n1 1600 --n2 1200 --n3 1200 --model nn
python run_mimic.py --n1 1900 --n2 1425 --n3 1425 --model nn
python run_mimic.py --n1 2200 --n2 1650 --n3 1650 --model nn
python run_mimic.py --n1 2500 --n2 1875 --n3 1875 --model nn
python run_mimic.py --n1 2800 --n2 2100 --n3 2100 --model nn
python run_mimic.py --n1 3100 --n2 2325 --n3 2325 --model nn
python run_mimic.py --n1 3400 --n2 2550 --n3 2550 --model nn
python run_mimic.py --n1 3700 --n2 2775 --n3 2775 --model nn
python run_mimic.py --n1 4000 --n2 3000 --n3 3000 --model nn
python run_mimic.py --n1 4300 --n2 3225 --n3 3225 --model nn
python run_mimic.py --n1 4600 --n2 3450 --n3 3450 --model nn
python run_mimic.py --n1 4900 --n2 3675 --n3 3675 --model nn
python run_mimic.py --n1 5200 --n2 3900 --n3 3900 --model nn
python run_mimic.py --n1 5500 --n2 4125 --n3 4125 --model nn
python run_mimic.py --n1 5800 --n2 4350 --n3 4350 --model nn
python run_mimic.py --n1 6100 --n2 4575 --n3 4575 --model nn
python run_mimic.py --n1 6400 --n2 4800 --n3 4800 --model nn
python run_mimic.py --n1 6700 --n2 5025 --n3 5025 --model nn
python run_mimic.py --n1 7000 --n2 5250 --n3 5250 --model nn
python run_mimic.py --n1 7300 --n2 5475 --n3 5475 --model nn
python run_mimic.py --n1 7600 --n2 5700 --n3 5700 --model nn
python run_mimic.py --n1 7900 --n2 5925 --n3 5925 --model nn
python run_mimic.py --n1 8200 --n2 6150 --n3 6150 --model nn
EOF
