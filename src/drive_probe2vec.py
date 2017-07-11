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
load_dir = params.get('load_dir',None)
num_processes = params.get('num_processes', 3)

if "fastq" in params['parser']:
    parser = kmerize_fastq_parse
else:
    parser = kmerize_fasta_parse
    
# build an embedder, save the embedder, dataset reader objects.
embedder, dictionary = word2vec(files=selex_files, 
                                parse=parser, 
                                save_dir=selex_save_dir, 
                                load_dictionary_dir=load_dir,
                                num_processes=num_processes,
                                read_data_async=params['read_data_async'], 
                                k=params['K'], stride=params['stride'], 
                                stdout_to_file=params['really_verbose'], 
                                timing=params['timing'], 
                                outfile=params['outfile'])
