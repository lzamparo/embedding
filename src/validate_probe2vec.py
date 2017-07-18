import sys
import os
import yaml
from probe2vec.w2v import word2vec, Word2VecEmbedder
from probe2vec.dataset_reader import kmerize_fastq_parse, kmerize_fasta_parse, DatasetReader
from scipy.stats import bernoulli
from collections import Counter

from probe2vec.theano_minibatcher import (
    TheanoMinibatcher, NoiseContrastiveTheanoMinibatcher
)


def train_valid_split(filename, data_dir, split_percentage):
    ''' Split the given file into probes used for training and probes used for validation '''

    def parse_sentence(kmers, d, factor):
        for kmer in kmers:
            if kmer not in d.keys():
                d[kmer] = [factor]
            else:
                d[kmer].append(factor)

    # extract factor from filename
    factor = filename.split('_')[0]
    
    # parse probe in file by however they should be parsed (need extra arg)
    if filename.endswith('.gz'):
        f = gzip.open(os.path.join(data_dir, filename), mode='rt', encoding='utf-8')
    else:
        f = open(os.path.join(data_dir, filename), mode='r', encoding='utf-8')   
                 
    tokenized_sentences = kmerize_fasta_parse(f, **params) 
    f.close()
    partition = bernoulli.rvs(split_percentage,size=len(tokenized_sentences)).astype('bool').tolist()
    
    training_dict = {}
    validation_dict = {}
    
    #training_dict = {parse_sentence(sentence) for sentence, indicator in zip(tokenized_sentences, partition) if indicator}
    training_sentences = [s for s,i in zip(tokenized_sentences,partition) if i]
    validation_sentences = [s for s,i in zip(tokenized_sentences,partition) if not i]
    
    for sentence in training_sentences:        
        parse_sentence(sentence, training_dict, factor)
    for sentence in validation_sentences:
        parse_sentence(sentence, validation_dict, factor)
     
       
    return training_dict, validation_dict


def get_positive_selex_files(file_list):
    ''' filter the list of files containing validation data to include only those that
    we really want to use.  For SELEX data, that includes just the *_pos.fasta files'''
    return [f for f in file_list if f.endswith('pos.fasta')]

def get_negative_selex_files(file_list):
    ''' filter the list of fiels containing validation data to include only those
    containing so-called negative probes.  '''
    return [f for f in file_list if f.endswith('neg.fasta')]

def update_factor_dict(d, update_d):
    ''' Update the values of d with the values of update_d.  For k in update_d
    if k in d.keys(), append the value(s) to d[k], otherwise set d[k] = update_d[k] '''
    for k in update_d.keys():
        if k not in d.keys():
            d[k] = update_d[k]
        else:
            d[k].extend(update_d[k])
            
    
def create_valid_set(data_dir, split_percentage, file_filter=get_positive_selex_files):
    ''' Parse the data files in the given directory into a certain percentage for embedding a 
    labeled set of probes for given factors (training_probes), and the remainder for measuring the 
    accuracy of this embedding (validation_probes). 
    
    data_dir: string.  Directory containing the data to be embedded. 
    split_percentage: float.  Percentage to be used for labeleing embedded probes. '''
    
    assert(0.0 < split_percentage and split_percentage < 1.0) 
    data_files = [f for f in os.listdir(os.path.expanduser(data_dir))]
    selex_files = file_filter(data_files)
    
    training_probes = {}
    validation_probes = {}
    
    for f in selex_files:
        f_train, f_valid = train_valid_split(f, data_dir, split_percentage)
        update_factor_dict(training_probes,f_train)
        update_factor_dict(validation_probes,f_valid)
    
    # Transform values of training_probes, validation_probes to ordered list of tuples
    training_kmers_factor_counts = {k: Counter(v) for k,v in training_probes.items()}
    validation_kmers_factor_counts = {k: Counter(v) for k,v in validation_probes.items()}
    
    return training_kmers_factor_counts, validation_kmers_factor_counts


# load the params from the yaml file given in sys.argv[1]
with open(os.path.expanduser(sys.argv[1])) as f:
    params = yaml.load(f)

# parse params from yaml file
data_dir = os.path.abspath(params['data_dir'])
selex_save_dir = os.path.abspath(params['save_dir'])
selex_files = [os.path.join(data_dir,f) for f in os.listdir(data_dir) if f.endswith(params['file_suffixes'])]

if "fastq" in params['parser']:
    parser = kmerize_fastq_parse
else:
    parser = kmerize_fasta_parse
    
# load the DatasetReader object from the save dir
reader = DatasetReader(files=[], directories=[], skip=[], noise_ratio=15, 
                      t=1e-5, num_processes=3, 
                      unigram_dictionary=None, 
                      min_frequency=0, kernel=[1, 2, 3, 
                      4, 5, 5, 4, 3, 2, 1], 
                      load_dictionary_dir=params['load_dir'], 
                      max_queue_size=0, 
                      macrobatch_size=20000, 
                      parse=parser, 
                      verbose=True, k=params['K'], 
                      stride=params['stride'])
    
# load the embedder, DatasetReader objects
batch_size = 1000
noise_ratio=15
num_embedding_dimensions=200
full_batch_size = batch_size * (1 + noise_ratio)

minibatcher = NoiseContrastiveTheanoMinibatcher(
    batch_size=batch_size,
    noise_ratio=noise_ratio,
    dtype="int32",
    num_dims=2
)

embedder = Word2VecEmbedder(input_var=minibatcher.get_batch(),
                            batch_size=full_batch_size,
                            vocabulary_size=reader.get_vocab_size(),
                            num_embedding_dimensions=num_embedding_dimensions)
embedder.load(os.path.join(params['load_dir'],''))

# Make the training, validation embedding labels data sets
training_probes, validation_probes = create_valid_set(params['data_dir'], params['split'],get_positive_selex_files)

# Make the test embedding labels data set 


# Compute the top-k accuracy loss for the given embedder, probes, labels



    

