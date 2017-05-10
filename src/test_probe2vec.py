import os
from probe2vec.w2v import word2vec
from probe2vec.dataset_reader import kmerize_fastq_parse

#
# Testing driver for probe2vec
#

# data resides here
data_dir = os.path.abspath('../data/selex_sample/')
selex_save_dir = os.path.abspath('../results/selex_test_results')
#selex_files = [os.path.join(data_dir,f) for f in os.listdir(data_dir) if f.endswith("4.txt.gz")]
selex_files = [os.path.join(data_dir, "MAFK_ESAI_TGCCTG30NTCC_4.txt.gz")]

# build an embedder
embedder, dictionary = word2vec(files=selex_files, parse=kmerize_fastq_parse, save_dir=selex_save_dir, k=6, stride=4)

# test the embedder


# ???


# Profit.