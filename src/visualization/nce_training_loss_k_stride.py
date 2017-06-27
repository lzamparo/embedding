import re
import os
import pandas as pd
import altair as alt

# get list of files
txt_files = [f for f in os.listdir(os.path.expanduser("~/projects/embedding/results/debug_output")) if f.endswith('filtered.txt')]

# parse filename f'n
def parse_filename(f):
    parts = f.strip().split("_")
    return parts[1], parts[3]

# build df in tidy-data fashion
# build df with pd.DataFrame.from_items([('model',[name for i in range(len(model['losses_valid_auc']))]),('epoch',[i for i in range(len(model['losses_valid_auc']))]), ('measure', ['validation AUC' for i in range(len(model['losses_valid_auc']))]),('score', model['losses_valid_auc'])])
for fname in txt_files:
    K, stride = parse_filename(fname)
    my_f = open(fname, encoding='utf-8')
    for line in my_f:
        # scrape the data
    

# build the figure