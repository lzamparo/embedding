import os
from annoy import AnnoyIndex

# build annoy index tree
def build_index(n_hidden, embedder, tokens, n_trees=30):
    ''' build the AnnoyIndex for the (k-mer) word vectors in our vocabulary '''
    index = AnnoyIndex(n_hidden)

    for token in tokens:
        # get embedding vector for each token, add to index
        token_vec = embedder.embed(token)
        index.add_item(token, reshape_to_vector(token_vec, n_hidden))

    index.build(n_trees)
    return index

def most_similar(token_ID, reader, index, num_neighbors):
    ''' Given an ID, get the token IDs of the top-N most similar tokens according to the embedding '''
    # TODO: how can I use the distances?? (which seem to be cosine)
    
    ids, distances = index.get_nns_by_item(
        token_ID, num_neighbors, include_distances=True)
    return [(reader.unigram_dictionary.get_token(ids[i]), 1 - distances[i] / 2) for i in range(len(ids))]

def reshape_to_vector(array, dim):
    ''' Embedded tokens come back as arrays.  Reshape to vector for use in conjunction with annoy '''
    return array.reshape((dim,))
    
def merge_counters(input_list, token_counter_dict):
    ''' Merge the counters associated with the tokens in given a list of (token, distance) tuples '''
    if isinstance(input_list[0], tuple):
        counters_list = [token_counter_dict[t] for t,d in input_list]
    else:
        counters_list = input_list
    top_tokens = counters_list[0]
    for c in counters_list[1:]:
        top_tokens = top_tokens + c    
    return top_tokens

def get_top_k_factors(tokenized_list, index, embedder, unigram_dict, n=5):
    ''' Return the factors associated with the nearest k-mers for the tokenization of
    the given probe '''
    pass


def get_top_k_loss(activation):
    ''' For all probes in this batch of the validation set, return the top_k_loss '''
    pass

