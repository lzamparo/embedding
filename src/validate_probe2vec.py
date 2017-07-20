import sys
import os
import yaml
import pickle

from probe2vec.w2v import word2vec, Word2VecEmbedder
from probe2vec.dataset_reader import kmerize_fastq_parse, kmerize_fasta_parse, DatasetReader
from probe2vec.embedding_utils import build_index, most_similar, merge_counters, reshape_to_vector
from probe2vec.theano_minibatcher import (
    TheanoMinibatcher, NoiseContrastiveTheanoMinibatcher
)

from scipy.stats import bernoulli
from collections import Counter


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
    # and split randomly into training and validation sentences
    tokenized_sentences = kmerize_fasta_parse(os.path.join(data_dir,filename), **params) 
    partition = bernoulli.rvs(split_percentage,size=len(tokenized_sentences)).astype('bool').tolist()
    
    training_dict = {}
    
    training_sentences = [s for s,i in zip(tokenized_sentences,partition) if i]
    validation_sentences = [(factor, s) for s,i in zip(tokenized_sentences,partition) if not i]
    
    for sentence in training_sentences:        
        parse_sentence(sentence, training_dict, factor)
        
    return training_dict, validation_sentences


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
            
    
def create_valid_set(data_dir, split_percentage, file_filter=get_positive_selex_files, k=6, stride=1):
    ''' Parse the data files in the given directory into a certain percentage for embedding a 
    labeled set of probes for given factors (training_probes), and the remainder for measuring the 
    accuracy of this embedding (validation_sentences). 
    
    data_dir: string.  Directory containing the data to be embedded. 
    split_percentage: float.  Percentage to be used for labeleing embedded probes. '''
    
    assert(0.0 < split_percentage and split_percentage < 1.0) 
    data_files = [f for f in os.listdir(os.path.expanduser(data_dir))]
    selex_files = file_filter(data_files)
    
    training_kmers = {}
    validation_sentences = []
    
    for f in selex_files:
        f_train, f_valid = train_valid_split(f, data_dir, split_percentage)
        update_factor_dict(training_kmers,f_train)
        validation_sentences.extend(f_valid)
    
    # Transform values of training_probes to ordered list of tuples
    training_kmers_factor_counts = {k: Counter(v) for k,v in training_kmers.items()}
    
    return training_kmers_factor_counts, validation_sentences


# load the params from the yaml file given in sys.argv[1]
with open(os.path.expanduser(sys.argv[1])) as f:
    params = yaml.load(f)

# parse params from yaml file
data_dir = os.path.abspath(params['data_dir'])
selex_save_dir = os.path.abspath(params['save_dir'])
selex_files = [os.path.join(data_dir,f) for f in os.listdir(data_dir) if f.endswith(params['file_suffixes'])]
split = params.get('split', 0.75)
top_n = 5

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
                      load_dictionary_dir=params['save_dir'], 
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
embedder.load(os.path.join(params['save_dir'],''))

# Make the training, validation embedding label data sets
print("Splitting into training, validation probe sentences")
training_kmers, validation_sentences = create_valid_set(params['data_dir'],split, get_positive_selex_files, params['K'],params['stride'])

# Embed training data 
print("Embedding training probes, building NN tree")
training_tokens_ids = [reader.unigram_dictionary.get_id(token) for token in training_kmers.keys()]
index_tree = build_index(embedder.num_embedding_dimensions, embedder, training_tokens_ids)

# Compute the most associated factor, and top-5 most associated factor accuracy loss for the given embedder,
# validation probes, training_kmers
per_factor_probe_accuracy = {}
print("Computing list of aggregated 5-nn factors")
for factor, sentence in validation_sentences:
    # get the NN tokens, distances for each kmer in the sentence
    sentence_token_ids = [reader.unigram_dictionary.get_id(token) for token in sentence]
    top_n_tokens_by_sentence = [most_similar(t, reader, index_tree, top_n) for t in sentence_token_ids]
    merged_counter_list = [merge_counters(l, training_kmers) for l in top_n_tokens_by_sentence]
    merged_counter = merge_counters(merged_counter_list, training_kmers)
    
    if factor not in per_factor_probe_accuracy:
        per_factor_probe_accuracy[factor] = [merged_counter]
    else:
        per_factor_probe_accuracy[factor].append(merged_counter)
    
print("Done, pickling to params['save_dir']")    
# save the factor dict to params['save_dir']    
pickle.dump(per_factor_probe_accuracy, open(os.path.join(params['save_dir'],'factor_validation_dict.pkl'),'wb'))


    

