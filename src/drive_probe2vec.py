import sys
import os
import yaml
from probe2vec.w2v import word2vec
from probe2vec.dataset_reader import kmerize_fastq_parse, kmerize_fasta_parse


### Driver script for probe2vec

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

#selex_files = [os.path.join(data_dir, "MAFK_ESAI_TGCCTG30NTCC_4.txt.gz")]

# build an embedder
embedder, dictionary = word2vec(files=selex_files, parse=parser, save_dir=selex_save_dir, k=params['K'], stride=params['stride'])

# test the embedder


# ???


# Profit.