#!/usr/bin/env python2.6

import numpy
import pandas

from shogun.Features import CombinedFeatures, BinaryLabels
from shogun.Kernel import WeightedDegreePositionStringKernel, CombinedKernel, WeightedCommWordStringKernel
from shogun.Classifier import SVMLight
from modshogun import *
import cPickle
import bz2


class ShRNAtor(object):
    """
    class to talk to shogun
    """

    def __init__(self, svm=None, plot=False):
        """
        set stuff up
        """

        self.plot = plot
        self.svm = svm


    def train(self, train_examples, train_labels, param):
        """
        first simple script to test performance
        """

        lab = BinaryLabels(numpy.double(train_labels))
        train_feat = construct_features(train_examples)

        # set up kernel

        ########
        # set up WDK
        max_len = len(train_examples[0])
        num_shifts = param["shifts"]
        size = 400
        kernel_wdk = WeightedDegreePositionStringKernel(size, param["wdk_degree"])
        shifts_vector = numpy.ones(max_len, dtype=numpy.int32)*num_shifts
        kernel_wdk.set_shifts(shifts_vector)

        ########
        # set up spectrum
        use_sign = False
        kernel_spec_1 = WeightedCommWordStringKernel(size, use_sign)
        kernel_spec_2 = WeightedCommWordStringKernel(size, use_sign)

        ########
        # combined kernel
        kernel = CombinedKernel()
        kernel.append_kernel(kernel_wdk)
        kernel.append_kernel(kernel_spec_1)        
        kernel.append_kernel(kernel_spec_2)

        # init kernel
        kernel.init(train_feat, train_feat)

        max_len = len(train_examples[0])
        num_pos = sum(train_labels == 1.0)
        num_neg = sum(train_labels == -1.0)

        assert num_pos+num_neg == len(train_labels)
        assert num_pos > 0
        assert num_neg > 0

        # normalize by num of examples
        if param.has_key("cost_pos") and param.has_key("cost_neg"):
            cost_neg = param["cost_neg"] / numpy.sqrt(num_neg)
            cost_pos = param["cost_pos"] / numpy.sqrt(num_pos)
        else:
            cost_neg = param["cost"] / numpy.sqrt(num_neg)
            cost_pos = param["cost"] / numpy.sqrt(num_pos)

        self.svm = SVMLight(1.0, kernel, lab)

        # cost neg/pos
        self.svm.set_C(cost_neg, cost_pos)
        self.svm.parallel.set_num_threads(2)
        self.svm.train()

    def predict(self, test_examples):
        """
        first simple script to predict
        """
        # compute prediction performance
        test_feat = construct_features(test_examples)
        pred = self.svm.apply(test_feat)
        out = pred.get_values()
        return out


def load_data_nona(file_name, target_column):
    """
    load data from merged file, drop NAs
    """

    dat = pandas.io.parsers.read_table(file_name, index_col=0)

    # drop rows that have NAs in target_column
    dat_filtered = dat.dropna(axis=0, how='any', subset=[target_column])

    examples = dat_filtered["guide"]
    labels = dat_filtered[target_column]

    return examples, labels


def construct_features(features):
    """
    makes a list
    """
    feat_all = [inst for inst in features]
    feat_lhs = [inst[0:15] for inst in features]
    feat_rhs = [inst[15:] for inst in features]

    feat_wd = get_wd_features(feat_all)
    feat_spec_1 = get_spectrum_features(feat_lhs, order=3)
    feat_spec_2 = get_spectrum_features(feat_rhs, order=3)

    feat_comb = CombinedFeatures()
    feat_comb.append_feature_obj(feat_wd)
    feat_comb.append_feature_obj(feat_spec_1)
    feat_comb.append_feature_obj(feat_spec_2)

    return feat_comb


def save(filename, myobj, compression_format="bz2"):
    f = bz2.BZ2File(filename, 'wb')
    cPickle.dump(myobj, f, protocol=0)
    f.close()


def get_spectrum_features(data, order=3, gap=0, reverse=True):
    """
    create feature object used by spectrum kernel
    """
    charfeat = StringCharFeatures(data, DNA)
    feat = StringWordFeatures(charfeat.get_alphabet())
    feat.obtain_from_char(charfeat, order-1, order, gap, reverse)
    preproc = SortWordString()                                            
    preproc.init(feat)
    feat.add_preprocessor(preproc)
    feat.apply_preprocessor()

    return feat


def get_wd_features(data, feat_type="dna"):
    """
    create feature object for wdk
    """
    feat = StringCharFeatures(DNA)
    feat.set_features(data)
    return feat



def main():
    """
    main
    """
    
    # For parameter selection, please see code on github.
    c=15
    t=1.1
    a=0.6
    
    mir30_file = "mir30_data.txt"
    mirE_file = "mirE_data.txt"
    
    # build mir30 model
    examples_mir30, labels_mir30 = load_data_nona(mir30_file, "combined_thres")
    svm_mir30 = ShRNAtor(plot=True)
    svm_mir30.train(numpy.array(examples_mir30), numpy.array(labels_mir30), {"shifts": 1, "wdk_degree": 10, "cost": c})
    save("mir30_model.bz2", svm_mir30)
    
    # build mirE model
    examples_mirE, labels_mirE = load_data_nona(mirE_file, "class")
    svm_mirE = ShRNAtor(plot=True)
    svm_mirE.train(numpy.array(examples_mirE), numpy.array(labels_mirE), {"shifts": 1, "wdk_degree": 10, "cost": c})
    save("mirE_model.bz2", svm_mirE)
    
    
    # predicting on miR-E training data
    scores_svm_mir30 = svm_mir30.predict(examples_mirE)
    scores_svm_mirE = svm_mirE.predict(examples_mirE)
    scores_splash = [""] * len(scores_svm_mir30)

    # combine scores
    for i in range(len(scores_svm_mir30)):
        if scores_svm_mir30[i] >= t:
            scores_splash[i] = a*scores_svm_mir30[i] + (1-a)*scores_svm_mirE[i]
        else:
            scores_splash[i] = scores_svm_mir30[i]

    for i in range(len(examples_mirE)):
        print examples_mirE[i], scores_splash[i]
        
if __name__ == '__main__':
    main()
