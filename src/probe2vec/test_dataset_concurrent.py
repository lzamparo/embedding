import re
import gc
import time

from queue import Queue
from concurrent import futures

#from iterable_queue import IterableQueue
#from multiprocessing import Process
#from subprocess import check_output

from .counter_sampler import CounterSampler
from .token_map import UNK
from .unigram_dictionary import UnigramDictionary
import numpy as np
import os
import sys
import gzip

def generate_fastq(file):
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
    
def generate_fasta(file):
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
    
def sequence_parse(filename, seq_generator, **kwargs):
    '''
    Parses input corpus files into a file-format-independent in-memory
    representation.  The output of this function is passed into
    `build_examples` for any processing that is needed, irrespective of
    file format, to generate examples form the stored data.

    INPUTS
    * filename [str]: path to a corpus file to be read
    * seq_generator [generator]: generator that takes an open file handle and 
    generates successive sequences

    RETURNS
    * [any]: file-format-independent representation of training data.
    '''
    sentences = []
    
    if filename.endswith('.gz'):
        f = gzip.open(filename,encoding='utf-8')
    else:
        f = open(filename, encoding='utf-8')
        
    for line in generate_fastq(f):
        sentences.append(line.strip().split())
        
    f.close()
    return sentences    

class DatasetReader(object):

    def __init__(self,
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
        max_queue_size=0,
        macrobatch_size=20000,
        parse=sequence_parse,
        verbose=True,
        k=None,
        stride=None):
        
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
        self._parse = parse
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
        return self._parse(filename, **dict(kwargs, k=self.k, stride=self.stride))      
    
    
    def generate_macrobatches(self, filename_iterator):

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

        examples = self.generate_examples(filename_iterator)
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
        
    def generate_filenames(self):
        '''
        Takes a list of directories, generates the filenames to be processed for all this garbage
        '''
    
        # Process all the files listed in each directory, unless they
        # match an entry in skip
        if self.directories is not None:
            for dirname in self.directories:
                dirname = os.path.abspath(dirname)
    
                # Skip directories if they match a regex in skip
                if any([s.search(dirname) for s in self.skip]):
                    continue
    
                for filename in os.listdir(dirname):
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
    
    
    
    def generate_dataset_worker(self, file_iterator, macrobatch_queue):
        macrobatches = self.generate_macrobatches(file_iterator)
        for signal_examples, noise_examples in macrobatches:
            if self.verbose:
                print('sending macrobatch to parent process')
            macrobatch_queue.put((signal_examples, noise_examples))
    
        macrobatch_queue.close()
    
    
    
    def generate_dataset_concurrent(self, save_dir=None):
        '''
        Parallel version of generate_dataset_serial.  Each worker is 
        responsible for saving its own part of the dataset to disk, called 
        a macrobatch.  the files are saved at 
        'save_dir/examples/<batch-num>.npz'.
        '''
        
    
        ### Begin surgery here
        file_queue = Queue()
        macrobatch_queue = Queue(maxsize=0)
    
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
    
        ### end surgery here
    
        # Retrieve the macrobatches from the workers, write them to file
        signal_macrobatches = []
        noise_macrobatches = []
        for signal_macrobatch, noise_macrobatch in macrobatch_consumer:
    
            if self.verbose:
                print('receiving macrobatch from child process')
    
            yield signal_macrobatch, noise_macrobatch
        
        
if __name__ == "__main__":
    