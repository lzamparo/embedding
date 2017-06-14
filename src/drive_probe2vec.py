import sys
import os
import yaml
from probe2vec.w2v import word2vec
from probe2vec.dataset_reader import kmerize_fastq_parse, kmerize_fasta_parse


### Driver script for training a probe2vec model

# load the params from the yaml file given in sys.argv[1]
with open(sys.argv[1]) as f:
    params = yaml.load(f)

# parse params from yaml file
data_dir = os.path.abspath(params['data_dir'])
selex_save_dir = os.path.abspath(params['save_dir'])
selex_files = [os.path.join(data_dir,f) for f in os.listdir(data_dir) if f.endswith(params['file_suffixes'])]

if "fastq" in params['parser']:
    parser = kmerize_fastq_parse
else:
    parser = kmerize_fasta_parse

# build an embedder, save the embedder, dataset reader objects.
embedder, dictionary = word2vec(files=selex_files, parse=parser, save_dir=selex_save_dir, k=params['K'], stride=params['stride'])
