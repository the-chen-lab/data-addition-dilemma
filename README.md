# The Data Addition Dilemma 

This is the code for reproducing results from the paper: [The Data Addition Dilemma](https://arxiv.org/pdf/2408.04154). 

## ICU Datasets
We run our main paper experiments using the [eICU Dataset](https://eicu-crd.mit.edu/) using the YAIB framework. The corresponding forked repositories can be founds here: 
- [YAIB](https://github.com/heyyjudes/YAIB)
- [YAIB-Cohorts](https://github.com/heyyjudes/YAIB-cohorts) 
### Divergence Metrics
To generate divergence metrics for eICU, you first need to extract the 
data from each hospital. This can be done with the `YAIB/data-addition-scripts/gen_data.sh` script. 
Once data from all hospitals are saved, we can then run this script. 
```
python kl_utils.py --n_samples 1000 --output_dir distances --input_dir distance_data --score
```

## Other Datasets (Supplementary Materials)  
We also include code for mixture and sequential experiments that appear in our supplementary materials for three additional datasets.

### Folktables
This datset can be directly downloaded via the python package [Folktables](https://github.com/socialfoundations/folktables). 
Example commands to generate results for scaling and mixture experiments can be found at the top of each file in [folktables](folktables_exp) directory

### Yelp Data
The Yelp dataset consists of reviews and ratings. You can download the Yelp data from [their data website](https://www.yelp.com/dataset/download). If downloading onto a server, the easiest way is to fill out the information and then right-click on "Download JSON" and copy data link address.

### MIMIC-IV Data
The MIMIC-IV data consists of the patient diagnoses and 15-day readmission. Cleaning code is inspired by the [WILD-TIME](https://github.com/huaxiuyao/Wild-Time/blob/main/wildtime/data/mimic.py) repo. 

Both the mixture and sequential experiments can be run with the same files in the [Yelp-MIMIC](Yelp-MIMIC) directory. Shell scripts are included to demonstrate usage.
For Yelp and MIMIC plots, see `run_dip.py --data {yelp, mimic}`

## Citation
Please use the following citation to reference our work: 
```
@InProceedings{pmlr-v252-shen24a,
  title = 	 {The Data Addition Dilemma},
  author =       {Shen, Judy Hanwen and Raji, Inioluwa Deborah and Chen, Irene Y.},
  booktitle = 	 {Proceedings of the 9th Machine Learning for Healthcare Conference},
  year = 	 {2024},
  editor = 	 {Deshpande, Kaivalya and Fiterau, Madalina and Joshi, Shalmali and Lipton, Zachary and Ranganath, Rajesh and Urteaga, Iñigo},
  volume = 	 {252},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {16--17 Aug},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v252/main/assets/shen24a/shen24a.pdf},
  url = 	 {https://proceedings.mlr.press/v252/shen24a.html},
}
```
