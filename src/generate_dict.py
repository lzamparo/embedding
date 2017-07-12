import sys
import os
import yaml
from probe2vec.w2v import word2vec
from probe2vec.dataset_reader import kmerize_fastq_parse, kmerize_fasta_parse, generate_kmerized_fastq_parse, DatasetReader


### Driver script for building a dictionary and token chooser model

# load the model params from the yaml file given in sys.argv[1]
with open(sys.argv[1]) as f:
    params = yaml.load(f)

# parse params from yaml file
data_dir = os.path.abspath(params['data_dir'])
selex_save_dir = os.path.abspath(params['save_dir'])
selex_files = [os.path.join(data_dir,f) for f in os.listdir(data_dir) if f.endswith(params['file_suffixes'])]
load_dir = params.get('load_dir',None)
num_processes = params.get('num_processes', 3)
verbose = params.get('really_verbose', True)

if "fastq" in params['parser']:
    parser = kmerize_fastq_parse
else:
    parser = kmerize_fasta_parse
    
# Build the dataset reader object
reader = DatasetReader(
    files=selex_files,
    num_processes=num_processes,
    load_dictionary_dir=load_dir,
    parse=parser,
    verbose=verbose,
    k=params['K'],
    stride=params['stride']
)

# Prepare the dataset reader (this produces the counter_sampler stats)
if not reader.is_prepared():
    if verbose:
        print('preparing dictionaries...')
    reader_kwargs = {'verbose': verbose, 'save_dir': selex_save_dir, 'K': params['K'], 'stride': params['stride'], 'read_async': params['read_data_async']}    
    reader.prepare(**reader_kwargs)
    
# Save the dictionary
reader.save_dictionary(selex_save_dir)
