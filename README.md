# The Data Addition Dilemma 

This is the code for reproducing results from the paper: [The Data Addition Dilemma](). 

## ICU Datasets
We run our main paper experiments using the [eICU Dataset](https://eicu-crd.mit.edu/) using the YAIB framework. The corresponding forked repositories can be founds here: 
- [YAIB](https://github.com/heyyjudes/YAIB)
- [YAIB-Cohorts](https://github.com/heyyjudes/YAIB-cohorts) 

## Other Datasets (Supplementary Materials)  
We also include code for mixture and sequential experiments that appear in our supplementary materials for three additional datasets.

### Folktables
This datset can be directly downloaded via the python package [Folktables](https://github.com/socialfoundations/folktables). 
Example commands to generate results for scaling and mixture experiments can be found at the top of each file in [folktables](folktables) directory

### Yelp Data
The Yelp dataset consists of reviews and ratings. You can download the Yelp data from [their data website](https://www.yelp.com/dataset/download). If downloading onto a server, the easiest way is to fill out the information and then right-click on "Download JSON" and copy data link address.

### MIMIC-IV Data
The MIMIC-IV data consists of the patient diagnoses and 15-day readmission. Cleaning code is inspired by the [WILD-TIME](https://github.com/huaxiuyao/Wild-Time/blob/main/wildtime/data/mimic.py) repo. 

Both the mixture and sequential experiments can be run with the same files in the [Yelp-MIMIC](Yelp-MIMIC) directory. Shell scripts are included to demonstrate usage.
For Yelp and MIMIC plots, see `run_dip.py --data {yelp, mimic}`

## Divergence Metrics
To generate divergence metrics: 
```
python compute_KL.py --mixture --n_runs 1 --n_samples 5000 --year 2014 --test_ratio 0.3 --dataset folktables
```

## Citation
Please use the following citation to reference our work. 
