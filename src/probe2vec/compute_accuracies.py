import os
from annoy import AnnoyIndex

# build annoy index tree
def build_index(n_hidden, embedder, tokens, n_trees=30):
    ''' build the AnnoyIndex for the (k-mer) word vectors in our vocabulary '''
    index = AnnoyIndex(n_hidden)

    for token in tokens:
        # get embedding vector for each token, add to index
        token_vec = embedder.embed(token)
        index.add_item(token, token_vec)

    index.build(n_trees)
    return index

def most_similar(vector, reader, index, num_neighbors):
    ''' Find and return the top-N most similar tokens '''
    # TODO: how can I use the distances (which seem to be cosine)
    ids, distances = index.get_nns_by_vector(
        vector, num_neighbors, include_distances=True)
    return [(reader.unigram_dictionary.get_token(ids[i]), 1 - distances[i] / 2) for i in range(len(ids))]
    

def get_top_k_factors(tokenized_list, index, embedder, unigram_dict, n=5):
    ''' Return the factors associated with the nearest k-mers for the tokenization of
    the given probe '''
    # tokenized list of k-mers -> ids -> vectors
    
    # find k tokens corresponding to nearest vectors to each probe
    closest_n_ids = [most_similar()]
    # closest_n_tokens = [get the token for each id]
    # update the counter
    
    
    # collect Counters associated count of factor(s) associated with each of those tokens to list
    
    # sort list by Counter, return top-k


#    use something like gensim.models.keyedvectors.most_similar() to 
#    find the k tokens corresponding to the k nearest vectors for this probe



def get_top_k_loss(activation):
    ''' For all probes in this batch of the validation set, return the top_k_loss '''
    pass

