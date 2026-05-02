# Data

Raw and processed data are not committed to this repo. The raw CSV is fetched
on demand from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset).

## Fetch the data

From the project root:

```bash
python src/data.py
```

This will populate `data/raw/online_shoppers_intention.csv` (~700 KB, 12,330
rows).

## Citation

Sakar, C. & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset*.
UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q
