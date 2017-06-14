import os
from annoy import AnnoyIndex

# build annoy index tree
def build_index(n_hidden, n_trees=30, embedder, tokens):
    ''' build the AnnoyIndex for the (k-mer) word vectors in our vocabulary '''
    index = AnnoyIndex(n_hidden)

    for i, token in enumerate(tokens):
        # get embedding vector for each token, add to index
        
        token_vec = embedder.embed(token)
        index.add_item(i, token_vec)

    index.build(n_trees)
    return index

def get_top_k_factors(tokenized_list, k=5, index):
    ''' Return the factors associated with the nearest k-mers for the tokenization of
    the given probe '''
    

# for each probe in the validation set:
#    initialize an empty list
#    tokenize the probe
#    use something like gensim.models.keyedvectors.most_similar() to 
#    find the k tokens corresponding to the k nearest vectors for this probe
#         add the factor(s) associated with those tokens to the list
#    sort the list by number of appearances per factor
#    report the top-k factors
#    
#  
#find the nearest top k tokens in the embedded space


def get_top_k_loss(activation):
    ''' For all probes in this batch of the validation set, return the top_k_loss '''
    pass

