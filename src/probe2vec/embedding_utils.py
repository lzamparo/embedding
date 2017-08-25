import os
from annoy import AnnoyIndex
import gzip



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

def get_top_k_factors(tokenized_list, index, embedder, unigram_dict, n=5):
    ''' Return the factors associated with the nearest k-mers for the tokenization of
    the given probe '''
    pass


def get_top_k_loss(activation):
    ''' For all probes in this batch of the validation set, return the top_k_loss '''
    pass

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

def ensure_str(s):
    '''
    Ensures that the string is encoded as a unicode str, not bytes
    '''
    try:
        return s.decode('utf8')
    except AttributeError:
        return s

class SequenceParserException(Exception):
    '''
    Used if the SequenceParser methods are called incorrectly, e.g using a 
    kmerized parser but not specifying K, or an improper specification of 
    'K' or 'stride'.
    '''
    pass
    
class SequenceParser(object):
    
    def __init__(self, **kwargs):
        '''
        Initialize the parser as a kmerizing parser of fastq files,
        fasta files, or space delimited word sentences.
        '''
        file_type = kwargs.get('parser', None)
        self.K = kwargs.get('K', -1)
        self.stride = kwargs.get('stride', -1)
        
        if file_type is not None and 'fastq' in file_type:
            self.parse = self.kmerize_fastq_parse
        elif file_type is not None and 'fasta' in file_type:
            self.parse = self.kmerize_fasta_parse
        else:
            self.parse = self.default_parse
        
        

    def rejoin_to_probe(self, kmer_list, k, stride):
        '''
        Re-joins the kmerized sentence into its original probe
        '''
        tojoin = [kmer_list[0]]
        for j in kmerized_test[1:]:
            tojoin.append(j[k-stride:])
        return ''.join(tojoin) 

    def kmerize(self, line, k, stride):
        '''
        Parses the sequences into kmers, using stride
        '''
        line = line.strip()
        kmerized = [line[i:i + k] for i in range(0, len(line) - k + 1, stride)]
        return kmerized

        
    def kmerize_fastq_parse(self, filename, **kwargs):
        '''
        Parses input from fastq a fastq encoded corpus files into a 
        file-format independent  in-memory representation.  The output
        of this function is passed into `build_examples` for any 
        processing that is needed, irrespective of file format, to 
        generate examples from the stored data.
    
        INPUTS
        * filename [str]: path to corpus file to be read
    
        RETURNS
        * [any]: representation of training data.
        '''
        
        if self.K < 0 or self.stride < 0:
            raise SequenceParserException("For kmerized parsing"
                                                     "k must be > 0 and "
                                                     "stride must be > 0")
        
        tokenized_sentences = []
        
        if filename.endswith('.gz'):
            f = gzip.open(filename, mode='rt', encoding='utf-8')
        else:
            f = open(filename, mode='r', encoding='utf-8')
            
        for fastq_record in self.generate_fastq(f):
            try:
                ID, seq, spacer, quality = fastq_record
            except ValueError:
                fastq_str = "\n".join(fastq_record)
                print("Got a malformed fastq record in ", filename, " : ", fastq_str)
                continue
            tokenized_sentences.append(self.kmerize(seq, self.K, self.stride))
            
        f.close()
        return tokenized_sentences



    def kmerize_fasta_parse(self, filename, **kwargs):
        '''
        Parses input from fastq a fasta encoded corpus files into a 
        file-format independent  in-memory representation.  The output
        of this function is passed into `build_examples` for any 
        processing that is needed, irrespective of file format, to 
        generate examples from the stored data.
    
        INPUTS
        * filename [str]: path to corpus file to be read
    
        RETURNS
        * [any]: representation of training data.
        '''
        if self.K < 0 or self.stride < 0:
            raise SequenceParserException("For kmerized parsing"
                                                     "k must be > 0 and "
                                                     "stride must be > 0")
        
        tokenized_sentences = []
        
        if filename.endswith('.gz'):
            f = gzip.open(filename, mode='rt', encoding='utf-8')
        else:
            f = open(filename, mode='r', encoding='utf-8')
            
        for fasta_record in self.generate_fasta(f):
            try:
                ID, seq = fasta_record
            except ValueError:
                fasta_str = "\n".join(fasta_record)
                print("Got a malformed fastq record in ", filename, " : ", fasta_str)
                continue
            tokenized_sentences.append(self.kmerize(seq, self.K, self.stride))
            
        f.close()
        return tokenized_sentences    


    def generate_fasta(self, file):
        ''' Parse and yield two line fasta records '''
        record = []
        for line in file:
            if line.startswith(">"):
                if record:
                    yield record
                record = [line.strip()]
            else:
                record.append(line.strip())
        yield record
        
        

    def generate_fastq(self, file):
        ''' Parse and yield four line fastq records '''
        record = []
        for line in file:
            if line.startswith("@HISEQ"):
                if record:
                    yield record
                record = [line.strip()]
            else:
                record.append(line.strip())
        yield record  
      

    def default_parse(self, filename, **kwargs):
        '''
        Parses input corpus files into a file-format-independent in-memory
        representation.  The output of this function is passed into
        `build_examples` for any processing that is needed, irrespective of
        file format, to generate examples form the stored data.
    
        INPUTS
        * filename [str]: path to a corpus file to be read
    
        RETURNS
        * [any]: file-format-independent representation of training data.
        '''
        tokenized_sentences = []
        
        if filename.endswith('.gz'):
            f = gzip.open(filename,encoding='utf-8')
        else:
            f = open(filename, encoding='utf-8')
            
        for line in f:
            tokenized_sentences.append(line.strip().split())
            
        f.close()
        return tokenized_sentences





    
    
        
        




