'''
This module keeps track of the vocabulary present in a corpus, provides
a two-way maping between tokens (strings) and token_ids (integers),
keeps track of token frequencies, and can yield samples from the
unigram distribution.
'''

import os
import numpy as np
from collections import OrderedDict
from .token_map import TokenMap, SILENT, WARN, ERROR, UNK
from .counter_sampler import UnigramCounterSampler
from .token_map import SeqTokenMap


class UnigramDictionary(object):
    '''
    Bundles together a TokenMap and CounterSampler.  Provides a method for
    pruning the vocabluary while keeping the TokenMap and CounterSampler
    in sync with one another.
    '''


    def __init__(self, on_unk=WARN, token_map=None, counter_sampler=None, seqmap=False):
        '''
        Create a new UnigramDictionary.  Typical usage provides no
        arguments, but a token_map and counter_sampler can be provided
        to build a UnigramDictionary that comprises them.
        '''
        self.on_unk = on_unk
        self.token_map = token_map
        if token_map is None:
            self.token_map = TokenMap(on_unk=on_unk) if not seqmap else SeqTokenMap(on_unk=on_unk)

        self.counter_sampler = counter_sampler
        self.prepared = False
        if counter_sampler is None:
            self.counter_sampler = UnigramCounterSampler()


    def sort(self):
        ### Not used in project
        unk_count = self.counter_sampler.counts[0]

        # Get the counts and tokens (skipping the first UNK entry)
        # They are parallel arrays (ith count corresponds to ith token)
        
        ## TODO: alter to return idx via cs_map
        counts = self.counter_sampler.counts[1:]
        tokens = self.token_map.tokens[1:]

        # Zip them together and sort by counts
        try:
            token_counts = zip(counts, tokens)
            token_counts.sort(reverse=True)
        except AttributeError:
            token_counts = list(zip(counts,tokens))
            token_counts.sort(reverse=True)

        # Separate them again
        new_counts = [unk_count]
        new_tokens = ['UNK']
        for count, token in token_counts:
            new_counts.append(count)
            new_tokens.append(token)

        # Rebuild the token_map and counter_sampler on the sorted arrays
        self.token_map = TokenMap(on_unk=self.on_unk, tokens=new_tokens)
        self.counter_sampler = CounterSampler(counts=new_counts)


    def remove(self, token):
        idx = self.get_id(token)
        self.token_map.remove(token)
        self.counter_sampler.remove(token)
        self.prepared = False
        


    def compact(self):
        self.token_map.compact()
        self.counter_sampler.compact()
        self.ensure_prepared()


    def prune(self, min_frequency=5, count=False):
        '''
        Remove all tokens that have been observed fewer than min_frequency
        times.  Counts for tokens that are removed are attributed to UNK.
        `count=True` enumerates the discarded tokens.
        '''
        counts = []
        tokens = []
        if count:
            dumped = []
        for token, idx in self.token_map.get_kviterator():

            # Copy over tokens that have at least min_frequency
            # observations. Also copy over UNK no matter what it's
            # frequency.
            if idx == 0:
                tokens = ['UNK'] + tokens
                counts.append(0)
                continue
            
            if (
                self.counter_sampler.get_frequency(token) >= min_frequency
            ):
                tokens.append(token)
                counts.append(self.get_frequency(token))

            # Skip tokens that have too little frequency.  Attribute their
            # observations to UNK
            else:
                counts[UNK] += self.get_frequency(token)
                if count:
                    dumped.append(token)
                    

        # Create a new TokenMap and CounterFrequency based on the
        # filtered tokens and their counts
        if isinstance(self.token_map, TokenMap):
            self.token_map = TokenMap(on_unk=self.on_unk, tokens=tokens)
        else:
            self.token_map = SeqTokenMap(on_unk=self.on_unk, tokens=tokens)
        self.counter_sampler = UnigramCounterSampler(counts = OrderedDict( ((t,c) for t,c in zip(tokens,counts)) ))
        if count:
            print("dropped ", len(dumped), " tokens in pruning the unigram dictionary")
        
        # Make sure the UD is prepared for fast sampling
        self.ensure_prepared()


    def add(self, token):
        '''
        Add a new token.  If this "token type" (which means this specific
        spelling of a word) has not been seen before, add it to the
        mapping.  Also increment the count for that token type.  Return
        its ID under the token mapping.
        '''

        # Get or create an id for this token
        token_id = self.token_map.add(token)

        # Increment the frequency count
        self.counter_sampler.add(token)

        return token_id


    def add_count(self, token, count):
        '''
        Add `count` to the counts for `token`, making a new entry if 
        necessary.
        '''
        # Get or create an id for this token
        token_id = self.token_map.add(token)
        # Increment the frequency count
        self.counter_sampler.add_count(token, count)


    def get_vocab_size(self):
        '''
        Return the number of unique tokens in the token_map.
        '''
        return len(self.token_map)


    def get_num_tokens(self):
        '''
        Return the total number of (non-distinct) tokens observed.
        '''
        return len(self.counter_sampler)


    def ensure_prepared(self):
        if self.prepared:
            return
        else:
            self.counter_sampler.ensure_prepared()
            self.ordered_tokens = list(self.counter_sampler.counts)
            self.prepared = True

    def __len__(self):
        '''
        Same as get_vocab_size().
        Return the number of unique tokens in the token_map.
        '''
        return len(self.token_map)


    def update(self, token_iterable):
        '''
        Like `add`, but accepts an iterable of tokens, incrementing the
        count for each of them.
        '''
        return [self.add(token) for token in token_iterable]


    def add_dictionary(self, other):
        '''
        Adds counts from another UnigramDictionary, `other`, to `self`'s
        counts, i.e. adding in place.
        '''
        self.update_counts(other.get_frequency_list())


    def update_counts(self, token_counts_iterable):
        '''
        Like `add_count` but accepts an iterable of (token,count) pairs,
        and increments the count for each token by the count given.
        Expected usage is to have a dictionary with tokens as keys
        and counts as values, and pass in your_dict.iteritems().
        '''
        return [
            self.add_count(token, count) 
            for token, count in token_counts_iterable
        ]


    def get_id(self, token):
        '''
        Get the id (int) for the corresponding token (string).
        '''
        # Delegate to the underlying token_map.
        return self.token_map.get_id(token)


    def get_ids(self, token_iterable):
        '''
        Get the ids (list of ints) for the corresponding tokens (strings)
        issued by token_iterable.
        '''
        # Delegate to the underlying token map.
        return self.token_map.get_ids(token_iterable)

    
    def get_token(self, idx):
        '''
        Return token (string) for the corresponding id (int)
        '''
        if self.prepared:
            return self.ordered_tokens[idx]
        else:
            self.ensure_prepared()
            return self.ordered_tokens[idx]
            

    ### Deprecated: not used except in testing
    def get_tokens(self, idx_iterable):
        '''
        Return the tokens (list of strings) for the corresponding ids
        (ints) issued by idx_iterable.
        '''
        # Delegate to the underlying counter sampler.
        ordered_tokens = list(self.counter_sampler.counts)
        return [self.get_token(idx) for idx in idx_iterable] 


    def save(self, savedir):
        '''
        Save the UnigramDictionary to the directory specified.  This saves
        the underlying TokenMap and CounterSampler in the directory
        given (savedir), using the default filenames "token-map.gz" and
        "counter-sampler.gz".
        '''
        
        # If the directory provided is a file, raise an error
        if os.path.exists(savedir):
            if os.path.isfile(savedir):
                raise IOError(
                    'Directory specified for saving UnigramDictionary is a '
                    'file.'
                )

        # If the directory provided doesn't exist, make it (this will not
        # make parent directories though).
        else:
            os.mkdir(savedir)


        # Save the TokenMap and CounterSampler by delegating to their
        # save functions.
        self.token_map.save(os.path.join(savedir, 'token-map.gz'))
        self.counter_sampler.save(os.path.join(
            savedir, 'counter-sampler.gz'
        ))


    def load(self, loaddir):
        '''
        Load a UnigramDictionary from the specified directory, by
        loading the TokenMap and CounterSampler stored there.  This assumes
        the filenames are 'token-map.gz' and 'counter-sampler.gz'.
        '''
        # Load the TokenMap by delegation to its load function
        self.token_map = TokenMap()
        self.token_map.load(os.path.join(loaddir, 'token-map.gz'))


        # Load the CounterSampler by delegation to its load function
        self.counter_sampler = UnigramCounterSampler()
        self.counter_sampler.load(
            os.path.join(loaddir, 'counter-sampler.gz'))


    def get_token_list(self):
        '''
        Gets an iterable of tokens currently in the dictionary.  Omits
        The 'UNK' token.
        '''
        
        return (
            token for token, _ in self.token_map.get_kviterator() if token is not 'UNK'
        )


    def get_frequency_list(self):
        '''
        Gets an iterable of (token, count) tuples.
        '''

        # Handle the case where there are no counts at all yet
        if len(self.counter_sampler.counts) == 0:
            return []

        # Otherwise get the counts normally
        return (
            (token, self.get_frequency(token))
            for token in self.get_token_list()
        )


    def sample(self, shape=None):
        '''
        Draw a sample according to the counter_sampler probability,
        Return the token_id of the token(s) sampled.
        '''
        # Delegate to the underlying CounterSampler
        cs_sample = self.counter_sampler.sample(shape)
        if shape is None:
            return np.int64(self.get_id(self.get_token(cs_sample)))
        token_ids = [self.get_id(self.get_token(idx)) for idx in cs_sample.flatten()]
        return np.asarray(token_ids, dtype=np.int64).reshape(shape)
        


    def get_probability(self, token):
        '''
        Return the probability associated to the given token.
        '''
        # Delegate to the underlying CounterSampler
        return self.counter_sampler.get_probability(token)


    def get_token_frequency(self, token):
        '''
        Return the frequency (count) associated to the token
        '''
        # If the token is unknown, return 0
        if token in self.counter_sampler.counts:
            return self.get_frequency(token)
        else:
            return 0


    def get_frequency(self, token):
        '''
        Return the frequency associated to token.
        '''
        # Delegate to the underlying CounterSampler
        return self.counter_sampler.get_frequency(token)

