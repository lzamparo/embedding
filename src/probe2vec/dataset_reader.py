import re
import gc
import time
from timeit import default_timer as timer
import random
import logging

from iterable_queue import IterableQueue
from multiprocessing import Process

from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter

from .counter_sampler import CounterSampler
from .token_map import UNK
from .unigram_dictionary import UnigramDictionary
from .embedding_utils import SequenceParser

import numpy as np
import os
import sys
import gzip
from . import embedding_utils


class TokenChooser(object):

    '''
    This choses which context token should be taken given a window
    of +/- K around a query token
    '''

    def __init__(self, K, kernel):
        if not len(kernel) == 2*K:
            raise ValueError(
                '`kernel` must have 2*K entries, one for '
                'each of the elements within the windows of +/- K tokens.'
            )

        self.K = K
        self.kernel = kernel
        self.samplers = {}
        self.indices = list(range(-K, 0)) + list(range(1, K+1))


    def choose_token(self, idx, length):
        '''
        Randomly choose a token according to the kernel supplied
        in the constructor.  Note that when sampling the context near
        the beginning of a sentence, the left part of the context window
        will be truncated.  Similarly, sampling context near the end of
        a sentence leads to truncation of the right part of the context
        window.  Short sentences lead to truncation on both sides.

        To ensure that samples are returned within the possibly truncated
        window, two values define the actual extent of the context to be
        sampled:

        `idx`: index of the query word within the context.  E.g. if the
            valid context is constrained to a sentence, and the query word
            is the 3rd token in the sentence, idx should be 2 (because
            of 0-based indexing)

        `length`: length of the the context, E.g. If context is
            constrained to a sentence, and sentence is 7 tokens long,
            length should be 7.
        '''

        # If the token is near the edges of the context, then the
        # sampling kernel will be truncated (we can't sample before the
        # first word in the sentence, or after the last word).
        # Determine the slice indices that define the truncated kernel.
        negative_idx = length - idx
        start = max(0, self.K - idx)
        stop = min(2*self.K, self.K + negative_idx - 1)

        # We make a separate multinomial sampler for each different
        # truncation of the kernel, because they each define a different
        # set of sampling probabilities.  If we don't have a sampler for
        # this particular kernel shape, make one.
        if not (start, stop) in self.samplers:

            trunc_probabilities = self.kernel[start:stop]
            self.samplers[start,stop] = (
                CounterSampler(trunc_probabilities)
            )

        # Sample from the multinomial sampler for the context of this shape
        outcome_idx = self.samplers[start,stop].sample()

        # Map this into the +/- indexing relative to the query word
        relative_idx = self.indices[outcome_idx + start]

        # And then map this into absolute indexing
        result_idx = relative_idx + idx

        return result_idx


MAX_NUMPY_SEED = 4294967295
def reseed():
    '''
    Makes a hop in the random chain.
    If called before spawning a child processes, it will ensure each child
    generates random numbers independently.  Unlike seeding child randomness
    from an os source of randomness, this is reproducible by starting the 
    parent with the same random seed.
    '''
    np.random.seed(np.random.randint(MAX_NUMPY_SEED))


class DataSetReaderIllegalStateException(Exception):
    '''
    Used if DatasetReader's methods are called in an incorrect order, e.g.
    calling generate_dataset() before calling prepare() on a DatasetReader
    that was not initialized with a UnigramDictionary.
    '''
    pass


class DatasetReader(object):

    def __init__(
        self,
        files=[],
        directories=[],
        skip=[],
        noise_ratio=15,
        t=1e-5,
        num_processes=3,
        unigram_dictionary=None,
        min_frequency=0,
        kernel=[1,2,3,4,5,5,4,3,2,1],
        load_dictionary_dir=None,
        max_queue_size=1000,
        macrobatch_size=16000,
        parser=SequenceParser(),
        verbose=True,
        k=None,
        stride=None
    ):

        # Register parameters to instance namespace
        self.files = files
        self.directories = directories
        self.skip = [re.compile(s) for s in skip]
        self.t = t
        self.noise_ratio = noise_ratio
        self.num_processes = num_processes
        self.kernel = kernel
        self.max_queue_size = max_queue_size
        self.macrobatch_size = macrobatch_size
        self._parse = parser.parse
        self.verbose = verbose
        self.min_frequency = min_frequency
        self.k = k
        self.stride = stride

        # If unigram dictionary not supplied, make one
        self.prepared = False
        self.unigram_dictionary = UnigramDictionary()

        # But load dictionary from file if load_dictionary_dir specified.
        if load_dictionary_dir is not None:
            if verbose:
                print('Loading dictionary from %s...' % load_dictionary_dir)
            self.load_dictionary(load_dictionary_dir)

        # Or, if an existing dictionary was passed in, use it
        if unigram_dictionary is not None:
            if verbose:
                print('A dictionary was supplied')
            self.unigram_dictionary = unigram_dictionary
            self.prune()
            self.prepared = True


    def is_prepared(self):
        '''
        Checks to see whether the dataset reader is ready to generate data.
        Given the simplicity, and that the logic of deciding when 
        self.prepared is True is found elsewhere, this method may seem
        unnecessary.  However, it provides a hook for more complex checking
        in subclasses.
        '''

        if self.prepared:
            return True
        return False


    def parse(self, filename, **kwargs):
        '''
        Delegate to the parse function given to the constructor.
        ''' 
        if kwargs.get('K') is None or kwargs.get('stride') is None:
            kwargs['K'] = self.k
            kwargs['stride'] = self.stride
        return self._parse(filename, **kwargs)


    def check_access(self, save_dir):
        '''
        Test out writing in save_dir.  The processes that generate the data
        to be saved can be really long-running, so we want to find out if 
        there is a simple IOError early!
        '''
        save_dir = os.path.abspath(save_dir)
        path, dirname = os.path.split(save_dir)

        # Make sure that the directory we want exists (make it if not)
        if not os.path.isdir(path):
            raise IOError('%s is not a directory or does not exist' % path)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        elif os.path.isfile(save_dir):
            raise IOError('%s is a file. %' % save_dir)

        # Make sure we can write to the file
        f = open(os.path.join(
            save_dir, '.__test-minibatch-generator-access'
        ), 'w')
        f.write('test')
        f.close
        os.remove(os.path.join(
            save_dir, '.__test-minibatch-generator-access' ))


    def generate_filenames(self):
        '''
        Generator that yields all filenames (absolute paths) making up the
        dataset.  (Files are specified to the Minibatcher constructor
        files and / or directories.  All listed files and all files directly
        contained in listed directories will be processed, unless they
        match regex patterns in the optional `skip` list.

        (no INPUTS)

        YIELDS
        * [str]: absolute path to a dataset file
        '''

        # Process the files listed in `files`, unles matches entry in skip
        if self.files is not None:
            # Randomize the ordering of the files!
            random.shuffle(self.files)   
            for filename in self.files:
                filename = os.path.abspath(filename)

                # Skip files if they match a regex in skip
                if any([s.search(filename) for s in self.skip]):
                    continue

                if self.verbose:
                    print('\tprocessing', filename)
                yield filename

        # Process all the files listed in each directory, unless they
        # match an entry in skip
        if self.directories is not None:
            for dirname in self.directories:
                dirname = os.path.abspath(dirname)

                # Skip directories if they match a regex in skip
                if any([s.search(dirname) for s in self.skip]):
                    continue

                # Randomize the ordering of the files!
                myfiles = os.listdir(dirname)
                random.shuffle(myfiles)
                for filename in myfiles:
                    filename = os.path.join(dirname, filename)

                    # Only process the *files* under the given directories
                    if not os.path.isfile(filename):
                        continue

                    # Skip files if they match a regex in skip
                    if any([s.search(filename) for s in self.skip]):
                        continue

                    if self.verbose:
                        print('\tprocessing', filename)

                    yield filename


    def numpyify(self, examples):
        '''
        Make an int32-type numpy array, ensuring that, even if the list of
        examples is empty, the array is two-dimensional, with the second
        dimension (i.e. number of columns) being 3.
        '''

        if len(examples) > 0:
            examples = np.array(examples, dtype='int32')
        else:
            examples = np.empty(shape=(0,3), dtype='int32')

        return examples


    def produce_macrobatches(self, filename_iterator):

        '''
        Assembles bunches of examples from the parsed data coming from
        files that were read.  Normally, this function might yield
        individual examples, however, in this case, we need to maintain
        a distinction between the noise- and signal-examples, and to
        keep them in consistent proportions.  So, here, we yield small
        bunches that consist of 1 signal example, and X noise examples,
        where X depends on `self.noise_ratio`.
        '''

        mcbatch_size = self.macrobatch_size
        noise_ratio = self.noise_ratio
        signal_examples = []
        noise_examples = []

        t0 = timer()
        examples = self.generate_examples(filename_iterator)
        t1 = timer()
        print("Time to generate this set of examples took ", (t1 - t0) * 1000, " microseconds")
        for signal_chunk, noise_chunk in examples:

            signal_examples.extend(signal_chunk)
            noise_examples.extend(noise_chunk)

            # Whenever we have enough examples, yield a macrobatch
            while len(signal_examples) > mcbatch_size:
                if self.verbose:
                    print('numpyifying')
                signal_macrobatch = self.numpyify(
                    signal_examples[:mcbatch_size])
                noise_macrobatch = self.numpyify(
                    noise_examples[:mcbatch_size * noise_ratio])

                if self.verbose:
                    print('no-padding:', len(signal_macrobatch))
                yield signal_macrobatch, noise_macrobatch

                signal_examples = signal_examples[mcbatch_size:]
                noise_examples = noise_examples[mcbatch_size*noise_ratio:]

        # After all files were processed, pad any remaining examples
        # to make up a final macrobatch
        if len(signal_examples) > 0:
            signal_remaining = mcbatch_size - len(signal_examples)
            noise_remaining = (
                mcbatch_size * noise_ratio - len(noise_examples))

            if self.verbose:
                print('padding and numpyifying')

            padding_row = self.get_padding_row()
            signal_macrobatch = self.numpyify(
                signal_examples + [padding_row] * signal_remaining)
            noise_macrobatch = self.numpyify(
                noise_examples + [padding_row] * noise_remaining)

            if self.verbose:
                print('padded to length:', len(signal_macrobatch))
            yield signal_macrobatch, noise_macrobatch


    def get_padding_row(self):
        return [UNK,UNK]


    def generate_dataset_serial(self):
        '''
        Generate the dataset from files handed to the constructor.  
        A single process is used to read the files. 
        '''

        # This cannot be called before calling prepare(), unless a prepared
        # UnigramDictionary was passed to the self's constructor
        if not self.is_prepared():
            raise DataSetReaderIllegalStateException(
                "DatasetReader: generate_examples() cannot be called "
                "before prepare() is called unless a prepared "
                "UnigramDictionary has was passed into the DatasetReader's "
                "constructor."
            )

        # Generate the data for each file
        file_iterator = self.generate_filenames()
        macrobatches = self.produce_macrobatches(file_iterator)
        for signal_examples, noise_examples in macrobatches:
            yield signal_examples, noise_examples


    def generate_dataset_worker(self, file_iterator, macrobatch_queue):
        macrobatches = self.produce_macrobatches(file_iterator)
        for signal_examples, noise_examples in macrobatches:
            if self.verbose:
                print('sending macrobatch to parent process')
            macrobatch_queue.put((signal_examples, noise_examples))
            #time.sleep(10.0)  ### trying to fix BrokenPipe error from not being able to put before the process dies and macrobatch_queue is wrapped up ###
        macrobatch_queue.close()
 


    def generate_dataset_parallel(self, save_dir=None):
        '''
        Parallel version of generate_dataset_serial.  Each worker is 
        responsible for saving its own part of the dataset to disk, called 
        a macrobatch.  the files are saved at 
        'save_dir/examples/<batch-num>.npz'.
        '''
        # This cannot be called before calling prepare(), unless a prepared
        # UnigramDictionary was passed to the self's constructor

        if not self.is_prepared():
            raise DataSetReaderIllegalStateException(
                "DatasetReader: generate_examples() cannot be called "
                "before prepare() is called unless a prepared "
                "UnigramDictionary has was passed into the "
                "DatasetReader's constructor."
            )

        # We save dataset in the "examples" subdir of the model_dir
        if save_dir is not None:
            examples_dir = os.path.join(save_dir, 'examples')
            # We are willing to create both the save_dir, and the
            # 'examples' subdir, but not their parents
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            if not os.path.exists(examples_dir):
                os.mkdir(examples_dir)
        else:
            examples_dir = None

        file_queue = IterableQueue()
        macrobatch_queue = IterableQueue(self.max_queue_size)

        # Put all the filenames on a producer queue
        file_producer = file_queue.get_producer()
        for filename in self.generate_filenames():
            file_producer.put(filename)
        file_producer.close()

        # Start a bunch of worker processes
        for process_num in range(self.num_processes):
            # Hop to a new location in the random-number-generator's state 
            # chain
            reseed()
            # Start child process that generates a portion of the dataset
            args = (
                file_queue.get_consumer(),
                macrobatch_queue.get_producer()
            )
            Process(target=self.generate_dataset_worker, args=args).start()

        # This will receive the macrobatches from all workers
        macrobatch_consumer = macrobatch_queue.get_consumer()

        # Close the iterable queues
        file_queue.close()
        macrobatch_queue.close()
     
        for signal_macrobatch, noise_macrobatch in macrobatch_consumer:

            if self.verbose:
                print('receiving macrobatch from child process')

            yield signal_macrobatch, noise_macrobatch

        # Explicitly close up macrobatch_consumer, which hopefully fixes the EOFError 
        macrobatch_consumer.close()

    def get_vocab_size(self):
        '''
        Get the size of the vocabulary.  Only makes sense to call this
        after Minibatcher.prepare() has been called, or if an
        existing (pre-filled) UnigramDictionary was loaded, since otherwise
        it would just return 0.

        (no INPUTS)

        OUTPUTS
        * [int]: size of vocabulary (including `UNK`).

        '''
        # Delegate to the underlying UnigramDictionary
        return len(self.unigram_dictionary)


    def load_dictionary(self, load_dir):
        '''
        Loads the unigram_dictionary from files stored in the supplied
        directory.

        INPUTS
        * directory [str]: Path to a directory in which unigram_dictionary
            files are stored.  Unigram dictionary will look for default
            filenames within that directory.

        OUTPUTS
        * [None]
        '''
        # Delegate to the underlying UnigramDictionary
        self.unigram_dictionary.load(os.path.join(
            load_dir, 'dictionary'
        ))
        self.prune()

        # It is now possible to call the data generators
        # `generate_dataset_serial()` and `generate_dataset_parallel()`
        self.prepared = True


    def save_dictionary(self, save_dir):
        '''
        Save the unigram_dictionary in the subfolder 'dictionary' beneath
        save_dir, in two files called 'counter-sampler.gz' and 
        'token-map.gz'.  `save_dir` will be created if it doesn't exist.
        '''
        # Make save_dir if it doesn't exist
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # Delegate to the underlying UnigramDictionary
        self.unigram_dictionary.save(os.path.join(
            save_dir, 'dictionary'
        ))


    def generate_token_worker(self, file_iterator, **kwargs):
        '''
        Enumerate all tokens in the file_iterator in a collections.Counter
        '''
        c = Counter()
        for tokens in self.parse(file_iterator, **kwargs):
            c.update(tokens)
        return c
    
    def preparation_parallel(self, **kwargs):
        '''
        Read through the corpus, building the UnigramDictionary in parallel,
        in the same manner as generate_dataset_parallel.
        '''
        
        # get all the files
        all_files = [filename for filename in self.generate_filenames()]
            
        # submit jobs to the worker processes    
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            futures = [executor.submit(self.generate_token_worker, filename, **kwargs) for filename in all_files]
            
            for future in as_completed(futures):
                countr = future.result()
                self.unigram_dictionary.update_counts(countr.items())
        
        self.prepared = True
    

    def preparation(self, **kwargs):

        # Read through the corpus, building the UnigramDictionary
        for filename in self.generate_filenames():
            for tokens in self.parse(filename, **kwargs):
                self.unigram_dictionary.update(tokens)

        self.prepared = True


    def prepare(self, *args, **kwargs):
        '''
        Used to perform any preparation steps that are needed before
        minibatching can be done.  E.g. assembling a dictionary that
        maps tokens to integers, and determining the total vocabulary size
        of the corpus.  It is assumed that files will need
        to be saved as part of this process, and that they should be
        saved under `save_dir`, with `self.save()` managing the details
        of writing files under `save_dir`.

        INPUTS

        * Note About Inputs *
        the call signature of this method is variable and is
        determined by the call signature of the core
        `self.preparation()` method.  Refer to that method's call
        signature.  Minimally, this method accepts `save_dir`

        * save_dir [str]: path to directory in which preparation files
            should be saved.

        RETURNS
        * [None]
        '''

        
        save_dir = kwargs.get('save_dir', None)
        read_async = kwargs.get('read_async', False)
        
        # Before doing anything, if we were requested to save the
        # dictionary, make sure we'll be able to do that (fail fast)        
        if save_dir is not None:
            self.check_access(save_dir)

        if not read_async:
            t0 = timer()
            self.preparation(**kwargs)
            t1 = timer()
            print("Serial unigram preparation took: ", (t1 - t0)*1000, " seconds" )
        else:
            t0 = timer()
            self.preparation_parallel(**kwargs)
            t1 = timer()
            print("Parallel unigram preparation took: ", (t1 - t0)*1000, " seconds" )            

        # Save the dictionary, if requested to do so.
        if save_dir is not None:
            self.save_dictionary(save_dir)

        # Prune the dictionary
        self.prune()



    def prune(self):
        '''
        Exposes the prune function for the underlying UnigramDictionary
        '''
        if self.verbose:
            print(
                'pruning dictionary to eliminate tokens occuring less than '
                '%d times.' % self.min_frequency
            )
        self.unigram_dictionary.prune(self.min_frequency, count=True)


    def generate_examples(self, filename_iterator):
        '''
        Using the data of a parsed file, generates examples.  Two kinds of
        examples are generated --- signal and noise.  They are yielded in a
        tuple, along with a flag indicating whether the particular example 
        is a signal, i.e.: (is_signal, example)
        '''

        num_examples = 0
        chooser = TokenChooser(K=len(self.kernel) // 2, kernel=self.kernel)
        
        # include parsing kwargs that were part of the declaration of this reader
        for filename in filename_iterator:

            # Parse the file, then generate a bunch of examples from it
            parsed = self.parse(filename)

            for tokens in parsed:

                # Isolated tokens (e.g. one-word sentences) have no context
                # and can't be used for training.
                if len(tokens) < 2:
                    continue

                for query_token_pos, query_token in enumerate(tokens):

                    # Possibly discard the token
                    if self.do_discard(query_token):
                        continue

                    # Sample a token from the context
                    context_token_pos = chooser.choose_token(
                        query_token_pos, len(tokens)
                    )
                    context_token_id = self.unigram_dictionary.get_id(tokens[context_token_pos])
                    signal_examples = [[self.unigram_dictionary.get_id(query_token), context_token_id]]
                    num_examples += 1

                    noise_examples = self.generate_noise_examples(
                        signal_examples)

                    num_examples += len(noise_examples)

                    yield (signal_examples, noise_examples)


    def generate_noise_examples(self, signal_examples):

        noise_examples = []
        for query_token_id, context_token_id in signal_examples:
            noise_examples.extend([
                [query_token_id, self.unigram_dictionary.sample()]
                for i in range(self.noise_ratio)
            ])

        return noise_examples


    def make_null_example(self):
        return [UNK, UNK]


    def do_discard(self, token):
        '''
        This function helps with downsampling of very common words.
        Returns true when the token should be discarded as a query word
        '''
        probability = self.unigram_dictionary.get_probability(token)
        discard_probability = 1 - np.sqrt(self.t/probability)
        do_discard = np.random.uniform() < discard_probability

        return do_discard
