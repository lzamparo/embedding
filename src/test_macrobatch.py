import sys
import os
import numpy as np
import yaml

from probe2vec.noise_contrast import get_noise_contrastive_loss
from probe2vec.embedding_utils import SequenceParser
from probe2vec.w2v import assemble_model_components
from probe2vec import embedding_utils


# Only import theano and lasagne if environment permits it

from theano import function
import lasagne
#from lasagne.layers import (
    #get_output, InputLayer, EmbeddingLayer, get_all_params,
    #get_all_param_values
#)
#from lasagne.init import Normal
from lasagne.updates import nesterov_momentum, adam


    # load the params from the yaml file given in sys.argv[1]

def test_macrobatch_for_nan(params_yaml):

        with open(params_yaml) as f:
                params = yaml.load(f)
            
        # parse params from yaml file
        data_dir = os.path.abspath(params['data_dir'])
        selex_save_dir = os.path.abspath(params['save_dir'])
        fasta_files = [os.path.join(data_dir,f) for f in os.listdir(data_dir) if f.endswith(params['file_suffixes'])]
        load_dir = params.get('load_dir',None)
        num_processes = params.get('num_processes', 3)
        mb_size = params.get('macrobatch_size', 100000)
        num_embedding_dimensions  = params.get('num_embedding_dimensions', 100)
        num_epochs = params.get('num_epochs',20)
        outfile = params.get('outfile', None)
        kernel = params.get('kernel', [1,2,3,4,5,5,4,3,2,1])
        min_frequency = params.get('min_frequency',0)
        verbose=True
        batch_size = params.get('batch_size', 1000)
        learning_rate = params.get('learning_rate', 0.01)
        momentum = params.get('momentum', 0.9)
        read_data_async = params.get('read_data_async', True)
        
        # create sequence parser from yaml config file
        parser = SequenceParser(**params)
        params['parser'] = parser
        
        # connect dataset reader to fasta files
        params['files'] = fasta_files
        
        ### set up what is needed for embedding test: embedder, reader, minibatcher, 
        reader, minibatcher, embedder = assemble_model_components(**params)
        
        # Architecture is ready.  Make the loss function, and use it to create 
        # the parameter updates responsible for learning
        loss = get_noise_contrastive_loss(embedder.get_output(), batch_size)
        updates = adam(loss, embedder.get_params())
        
        
        #nesterov_momentum(
            #loss, embedder.get_params(), learning_rate, momentum
        #)
        
        # Include minibatcher updates, which cause the symbolic batch to move
        # through the dataset like a sliding window
        updates.update(minibatcher.get_updates())
        
        # Use the loss function and the updates to compile a training function.
        # Note that it takes no inputs because the dataset is fully loaded using
        # theano shared variables
        train = function([], loss, updates=updates)
        
        macrobatch_limit = 1
        
        if read_data_async:
                macrobatches = reader.generate_dataset_parallel()
        else:
                macrobatches = reader.generate_dataset_serial()
        
        macrobatch_num = 0
        for signal_macrobatch, noise_macrobatch in macrobatches:
                macrobatch_num += 1
                if verbose:
                        print('running macrobatch %d' % (macrobatch_num - 1))
                
                minibatcher.load_dataset(signal_macrobatch, noise_macrobatch)
                losses = []
                for batch_num in range(minibatcher.get_num_batches()):
                        batch_loss = train()
                        if batch_num == 49:
                                values = embedder.get_param_values()                        
                        if not np.isnan(batch_loss):
                                losses.append(train())
                        else:
                                values = embedder.get_param_values()
                                print("Warning: NaN loss reported for batch", batch_num, " of epoch ", epoch)               
                print('\tmacrobatch average loss: %f' % np.mean(losses))
        
        if not np.isnan(np.mean(losses)):
                return True
        else:
                return False
        
params_yaml = sys.argv[1]
pass_fail = test_macrobatch_for_nan(params_yaml)
assert(pass_fail)
        