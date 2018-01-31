import sys
import os
import yaml
from probe2vec.w2v import word2vec
from probe2vec.embedding_utils import SequenceParser


### Driver script for training a probe2vec model

# load the params from the yaml file given in sys.argv[1]
with open(sys.argv[1]) as f:
    params = yaml.load(f)

## parse params from yaml file
#data_dir = os.path.abspath(params['data_dir'])
#selex_save_dir = os.path.abspath(params['save_dir'])
#selex_files = [os.path.join(data_dir,f) for f in os.listdir(data_dir) if f.endswith(params['file_suffixes'])]
#load_dir = params.get('load_dir',None)
#num_processes = params.get('num_processes', 3)
#mb_size = params.get('macrobatch_size', 100000)
#num_embedding_dimensions  = params.get('num_embedding_dimensions', 100)
#num_epochs = params.get('num_epochs',20)
#outfile = params.get('outfile', None)
#kernel = params.get('kernel', [1,2,3,4,5,5,4,3,2,1])
#min_frequency = params.get('min_frequency',0)

# create sequence parser from yaml config file
parser = SequenceParser(**params)
params['parser'] = parser

# connect dataset reader to fasta files
data_dir = os.path.abspath(params['data_dir'])
fasta_files = [os.path.join(data_dir,f) for f in os.listdir(data_dir) if f.endswith(params['file_suffixes'])]
params['files'] = fasta_files
    
# build an embedder, save the embedder, dataset reader objects.
embedder, dictionary = word2vec(**params)

# save the embedder, dataset reader objects
embedder.save(params['save_dir'])
dictionary.save_dictionary(params['save_dir'])