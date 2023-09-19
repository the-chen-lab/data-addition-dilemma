# More Data More Problems

## How to Recreate NeurIPS 2023 Plots

For Folktables plots, see `notebooks/folktables-income-sandbox.ipynb`

For Yelp and MIMIC plots, see `run_dip.py --data {yelp, mimic}`

Note that `run_full_experiments.py` from the submission NeurIPS supplement code is the same as `run_dip.py`

## Data
Data is loaded in a variety of ways.

### Yelp Data
You can download the Yelp data from [their data website](https://www.yelp.com/dataset/download). If downloading onto a server, the easiest way is to fill out the information and then right-click on "Download JSON" and copy data link address.

```
wget "[link address]"
```

### MIMIC Data
The MIMIC data is ... (TBD)

## TODO
 - Create environment yaml for easier setup
