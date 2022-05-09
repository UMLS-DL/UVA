import time
import pandas as pd
import numpy as np
import os
import math
import pickle
import sys
import gc
import csv
import glob
import pdb
import inspect
from pathlib import Path
from tqdm import tqdm
import random
import logging
import queue
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import multiprocessing
from itertools import islice
#import bert

from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, precision_score, recall_score, f1_score, matthews_corrcoef, \
    confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_recall_curve, plot_precision_recall_curve
import matplotlib.pyplot as plt

import tensorflow as tf
#import tensorflow_addons as tfa
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Activation, Layer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("f", "", "kernel")

flags.DEFINE_string("application_name", "run_umls_classifier", "")

flags.DEFINE_string("task_name", "umls", "The name of the task to train.")

flags.DEFINE_string("pre_trained_word2vec", "biowordvec.txt", "The vocabulary file that the BioWordVec model was trained on.")
flags.DEFINE_string("vocab_file", None, "The vocabulary file with a list of tokens for BERT")

flags.DEFINE_string("gen_vocab_file", "gen_vocab", "The vocabulary file with a list of tokens for BERT will be generated")

# COMMON FLAGS
flags.DEFINE_string("workspace_dp", "..", "The output directory where the model checkpoints will be written.")
flags.DEFINE_string("umls_dp", "../UMLS_VERSIONS/2020AA-ACTIVE", "The output directory where the model checkpoints will be written.")
flags.DEFINE_string("umls_version_dn", "2020AA-ACTIVE", "")
flags.DEFINE_string("umls_version_dp", None, "")
flags.DEFINE_string("dataset_version_dn", None, "")
flags.DEFINE_string("dataset_version_dp", None,"")
flags.DEFINE_string("important_info_fn","IMPORTANT_INFO.RRF","")
flags.DEFINE_string("umls_dl_dp", "../UMLS_VERSIONS/2020AA-ACTIVE/META_DL", "")
flags.DEFINE_string("umls_dl_dn", "META_DL", "")
flags.DEFINE_string("extra_dn", "extra", "")

flags.DEFINE_string("train_fn", "TRAIN_DS.RRF", "The output directory where the model checkpoints will be written.")
flags.DEFINE_string("val_fn", "DEV_DS.RRF", "The output directory where the model checkpoints will be written.")
flags.DEFINE_string("test_fn", "TEST_DS.RRF", "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("mrconso_master_fn", "MRCONSO_MASTER.RRF", "The output directory where the model checkpoints will be written.")
flags.DEFINE_list("mrconso_master_fields", ["ID", "CUI", "LUI", "SUI", "AUI", "AUI_NUM", "SCUI", "NS_ID", "NS_LEN", "NS", "NORM_STR", "STR", "SG"], "")
flags.DEFINE_list("ds_fields", ["jacc", "AUI1", "AUI2", "Label"],"")

flags.DEFINE_string("delimiter", "|", "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("test_dataset_fp", None, "")
flags.DEFINE_string("test_dataset_fp_rba_predictions", None, "")
flags.DEFINE_string("test_dataset_dp", None, "")
flags.DEFINE_string("train_dataset_dp", None, "")
flags.DEFINE_string("train_dataset_dn", None, "The input dataset dir inside. Should contain the .RRF files (or other data files) "
    "for the task.")
flags.DEFINE_string("test_dataset_dn", None, "The input dataset dir inside. Should contain the .RRF files (or other data files) "
    "for the task.")

flags.DEFINE_string("datasets_dp", None, "The input data dir. Should contain the .RRF files (or other data files) "
    "for the task.")
flags.DEFINE_string("datasets_dn", "UMLS_VERSIONS", "The input dataset dir inside. Should contain the .RRF files (or other data files) "
    "for the task.")


flags.DEFINE_string("run_id", "run_1", "")

flags.DEFINE_string("training_dn", "TRAINING", "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("output_dn", "test_8192b_2ep_exp3", "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("logs_dn", "logs", "The output directory where the model checkpoints will be written.")
flags.DEFINE_string("logs_fp", None, "File path to console logs")

flags.DEFINE_string("checkpoint_dn", "CHECKPOINT", "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string("train_result_dn", "TRAIN_RESULT", "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("predict_result_dn", "PREDICT_RESULT", "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("test_dp", None, "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("val_test_results_log_fn", "val_test_results_log", "The output directory where the model checkpoints will be written.")

flags.DEFINE_integer("start_col_idx", 1 , "The output directory where the model checkpoints will be written.")

flags.DEFINE_bool("skip_header", False , "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("tokenizer_pickle_fn", "TOKENIZER.PICKLE", "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("aui2vec_pickle_fn", "AUI2VEC.PICKLE", "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("embedding_fp", None, "Embedding dictionary pickle file")

flags.DEFINE_string("aui2id_pickle_fn", "AUI2ID.PICKLE", "The output directory where the model checkpoints will be written.")
flags.DEFINE_string("id2aui_pickle_fn", "ID2AUI.PICKLE", "The output directory where the model checkpoints will be written.")
flags.DEFINE_string("mrc_atoms_pickle_fn", "MRC_ATOMS.PICKLE", "The output directory where the model checkpoints will be written.")

flags.DEFINE_integer("exp_flavor", 1, 
                     "1: Base model"
                     "2: Base model + context vector"
                    )
                     
flags.DEFINE_integer("max_seq_length", 30, "The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_string("lstm_attention", "lstm", "Attention, LSTM, or both")

flags.DEFINE_string("word_embedding", "BioWordVec", "BioWordVec or BERT for Word embedding layer")

flags.DEFINE_bool("is_trainable", True, "Whether to train the embeddings.")

flags.DEFINE_integer("embedding_dim", 200, "Dimension for the embedding layer")

flags.DEFINE_integer("context_vector_dim", 150, "Dimension for the context vector")

flags.DEFINE_string("ConOptimizer", "SGD", "SGD or Adam")

flags.DEFINE_integer("ConEpochs", 100, "Num of epochs trained for context")

flags.DEFINE_string("padding_sequence", "pre", "Dimension for the context vector")

flags.DEFINE_string("truncating_sequence", "post", "Dimension for the context vector")

flags.DEFINE_integer("first_units_dense", 128, "Dimension for the context vector")

flags.DEFINE_integer("second_units_dense", 50, "Dimension for the context vector")

flags.DEFINE_bool("use_shared_dropout_layer", False, "Whether to run training.")

flags.DEFINE_float("shared_dropout_layer_first_rate", 0.5, "Whether to run training.")

flags.DEFINE_float("shared_dropout_layer_second_rate", 0.2, "Whether to run training.")

flags.DEFINE_string("dense_activation", "relu", "Dimension for the context vector")

flags.DEFINE_string("kernel_initializer", "random_normal", "Dimension for the context vector")

flags.DEFINE_string("bias_initializer", "random_normal", "Dimension for the context vector")

flags.DEFINE_string("loss_function", "mean_squared_error", "Dimension for the context vector")

flags.DEFINE_string("acc_metric", "acc", "Dimension for the context vector")

flags.DEFINE_float("learning_rate", 0.001, "The initial learning rate for Adam.")

flags.DEFINE_integer("batch_size", 8192, "Total batch size for training.")
flags.DEFINE_integer("predict_batch_size", 8192, "Total batch size for training.")

flags.DEFINE_float("predict_threshold", 0.5, "Total batch size for training.")

flags.DEFINE_integer("n_hidden", 50, "Number of hidden layers.")

flags.DEFINE_integer("n_epoch", 5, "Total number of training epochs to perform.")

flags.DEFINE_integer("start_epoch_predict", None, "Total number of training epochs to perform.")

flags.DEFINE_integer("end_epoch_predict", None, "Total number of training epochs to perform.")

flags.DEFINE_integer("epoch_predict_all", None, "")

flags.DEFINE_integer("start_aui1", None, "")

flags.DEFINE_integer("end_aui1", None, "")

flags.DEFINE_integer("train_verbose", 1, "Total number of training epochs to perform.")

flags.DEFINE_integer("predict_verbose", 1, "Total number of training epochs to perform.")

flags.DEFINE_integer("generator_workers", 6, "")

flags.DEFINE_bool("do_train", False, "Whether to run training.")
flags.DEFINE_bool("continue_training", False, "Whether to continueing training.")
flags.DEFINE_bool("do_predict", False, "Whether to run the model in inference mode on the test set.")
flags.DEFINE_bool("do_predict_all", False, "")

flags.DEFINE_integer("checkpoint_epoch", 0, "Epoch # to resume training from.")


flags.DEFINE_bool("do_prep", False, "")
flags.DEFINE_bool("do_report", False, "")
flags.DEFINE_bool("do_analyze", False, "")
flags.DEFINE_bool("wait_for_train", False, "")


# flags.DEFINE_bool("do_prep", True, "")
# flags.DEFINE_bool("do_train", False, "Whether to run training.")
# flags.DEFINE_bool("do_predict", False, "Whether to run the model in inference mode on the test set.")
flags.DEFINE_bool("predict_test_dir_after_every_epoch", False, "Whether to run training.")
flags.DEFINE_bool("load_IDs", True, "")
flags.DEFINE_list("metric_names", ["epoch", "accuracy", "precision", "recall", "f1", "auc", "best_threshold", "best_threshold_accuracy", "best_threshold_precision", "best_threshold_recall", "best_threshold_f1", "auc"], "")
flags.DEFINE_list("metric_labels", ["Epoch", "Accuracy", "Precision", "Recall", "F1", "AUC", "Best Threshold", "Best Threshold Accuracy", "Best Threshold Precision", "Best Threshold Recall", "Best Threshold F1", "Best Threshold AUC"], "")

flags.DEFINE_string("tmp_metrics_fn", "tmp_metrics.txt", "")
flags.DEFINE_string("tmp_metrics_graph_fn", "tmp_metrics_graph.png", "")
flags.DEFINE_string("final_metrics_fn", "final_metrics.txt", "")
flags.DEFINE_string("final_metrics_graph_fn", "final_metrics_graph.png", "")
flags.DEFINE_string("tmp_history_fn", "tmp_history.txt", "")
flags.DEFINE_string("tmp_history_graph_fn", "tmp_history_graph.png", "")
flags.DEFINE_string("final_history_fn", "final_history.txt", "")
flags.DEFINE_string("final_history_graph_fn", "final_history_graph.png", "")
flags.DEFINE_string("predictions_fn", "predictions.txt", "")
    
flags.DEFINE_string("ds_to_pickle_dp", None, "")

flags.DEFINE_string("KGE_Home", "..", "Path to the home directory of KGE")
flags.DEFINE_string("Embeddings", "6-Outputs/embeddings", "Path to the embedding folder")
flags.DEFINE_string("Model", "TransE_SGD", "TransE, HolE, ")
flags.DEFINE_string("TrainingData", "2-TrainingData", "Path to the training data")
flags.DEFINE_string("Raw", "1-Raw", "Path to intermediate dir")
flags.DEFINE_string("AUI2ID", "AUI2ID.PICKLE", "AUI2ID pickle file")
flags.DEFINE_string("ConVariant", "All_Triples", "All_Triples")
flags.DEFINE_string("AUI2CONVEC","aui2convec.pickle","The pickle file for the dictionary AUI:Context vector")
flags.DEFINE_string("WordEmbVariant", "BioWordVec", "BioBERT, BlueBERT, BERT, BioBERT_UMLS, UMLS")

flags.DEFINE_string("PredictionLayer", "Threshold", "Threshold, Softmax")
flags.DEFINE_string("DistanceScore", "Manhattan", "Manhattan, Cosine")

class DataGenerator(tf.keras.utils.Sequence):
    "Generates data for Keras"
    def __init__(self, fp_partition, aui2vec, aui2convec, max_seq_length, embedding_dim,word_embedding, context_vector_dim, exp_flavor,  batch_size = 32, 
                 n_classes = 2, shuffle = True, is_test = True):
        "Initialization"
        self.fp_partition = fp_partition
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.word_embedding = word_embedding
        self.context_vector_dim = context_vector_dim
        self.exp_flavor = exp_flavor
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.aui2vec = aui2vec
        self.aui2convec = aui2convec
        self.is_test = is_test
        
        
        self.IDs = [ID for ID in self.fp_partition.keys()]
        self.indexes = np.arange(len(self.IDs))
        self.on_epoch_end()
        
        
    def __len__(self):
        "Denotes the number of batches per epoch"
        return math.ceil(len(self.IDs) / self.batch_size)

    def __getitem__(self, index):
        "Generate one batch of data"

        # Generate data
        
        # Generate indexes of the batch
        if self.batch_size * (1+index) > len(self.IDs):
            indexes = self.indexes[index*self.batch_size:len(self.IDs)]
        else:
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
            
        list_IDs_temp = [self.IDs[k] for k in indexes]
        
        if self.is_test:
            if self.exp_flavor == 1:
                left, right = self.__read_batch_from_indexes(list_IDs_temp)
                return [left, right]
            elif self.exp_flavor == 2:
                left, context_left, right, context_right = self.__read_batch_from_indexes(list_IDs_temp)
                return [left, context_left, right, context_right]
        else:
            if self.exp_flavor == 1:
                left, right, labels = self.__read_batch_from_indexes(list_IDs_temp)
                return [left, right], labels
            elif self.exp_flavor == 2:
                left, context_left, right, context_right, labels = self.__read_batch_from_indexes(list_IDs_temp)
                return [left, context_left, right, context_right], labels
        
    def on_epoch_end(self):
        "Updates indexes after each epoch"
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __read_batch_from_indexes(self, list_IDs_temp):
        "Generates data containing batch_size samples" # X : (n_samples, *dim, embedding_dim)
        # Initialization
        batch_size = len(list_IDs_temp)
        
        lefts = np.empty((batch_size, self.max_seq_length), dtype = int)
        rights = np.empty((batch_size, self.max_seq_length), dtype = int)
        labels = None
        if not self.is_test:
            labels = np.empty((batch_size), dtype = int)
 
        if self.exp_flavor == 2:
            # Adding context vector
            context_lefts = np.empty((batch_size, FLAGS.context_vector_dim), dtype = float)
            context_rights = np.empty((batch_size, FLAGS.context_vector_dim), dtype = float)

        if self.exp_flavor == 1:
            # Generate data
            for i, ID in enumerate(list_IDs_temp):
                # Store sample
                if (ID[0] in self.aui2vec) and (ID[1] in self.aui2vec):
                    lefts[i,] = self.aui2vec[ID[0]]
                    rights[i,] = self.aui2vec[ID[1]]
                    if not self.is_test:
                        #labels[i] = self.fp_partition[ID][1]
                        labels[i] = self.fp_partition[ID]                 
    
            del list_IDs_temp
            if not self.is_test:
                return lefts, rights, labels
            else:
                return lefts, rights
        elif self.exp_flavor == 2:
            # Generate data
            for i, ID in enumerate(list_IDs_temp):
                # Store sample
                if (ID[0] in self.aui2vec) and (ID[1] in self.aui2vec):
                    lefts[i,] = self.aui2vec[ID[0]]
                    rights[i,] = self.aui2vec[ID[1]]
                    context_lefts[i,] = self.aui2convec[ID[0]]
                    context_rights[i,] = self.aui2convec[ID[1]]

                    if not self.is_test:
                        labels[i] = self.fp_partition[ID]
                        #labels[i] = self.fp_partition[ID][1]

            del list_IDs_temp
            if not self.is_test:
                return lefts, context_lefts, rights, context_rights, labels
            else:
                return lefts, context_lefts, rights, context_rights

class PairGenerator(tf.keras.utils.Sequence):
    "Generates data for Keras"
    def __init__(self, start_aui1, end_aui1, aui2vec, aui2convec, aui2cui, max_seq_length, embedding_dim, word_embedding, context_vector_dim, exp_flavor, batch_size = 32, 
                 n_classes = 2, shuffle = True, is_test = True):
        "Initialization"
        self.start_aui1 = start_aui1
        self.end_aui1 = end_aui1
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.word_embedding = word_embedding
        self.context_vector_dim = context_vector_dim
        self.exp_flavor = exp_flavor
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.aui2vec = aui2vec
        self.aui2convec = aui2convec
        self.aui2cui = aui2cui
        self.is_test = is_test
        
        self.num_pairs = (self.end_aui1 - self.start_aui1 + 1) * len(aui2vec)
        self.all_auis = list(self.aui2vec.keys())
        self.aui_len = len(aui2vec)
        self.num_batches = math.ceil(self.num_pairs/self.batch_size)

        self.batch_idx = dict()

    def compute_batch_idx(self):
        cur_aui1 = self.start_aui1
        cur_aui2 = 0
        batch_max_len = self.batch_size
        logger.info("Computing batch indexes")
        with tqdm(total = self.num_batches) as pbar:
            for idx in range(self.num_batches):
                pbar.update(1)
                # Check for negative pairs
                batch_len = 0 # Batch only contains negative pairs
                aui1 = cur_aui1
                aui2 = cur_aui2
                for i in range(batch_max_len):
                    if (self.aui2cui[aui1] != self.aui2cui[aui2]):
                        batch_len += 1
                    if aui2 == self.aui_len - 1:
                        aui2 = 0
                        aui1 += 1
                    else:
                        aui2 += 1

                #logger.info("Batch {}: ({}, {}, {}, {})".format(idx, cur_aui1, cur_aui2, batch_max_len, batch_len))         
                self.batch_idx[idx] = (cur_aui1, cur_aui2, batch_max_len, batch_len)
                # Check for the next batch"s indexes
                if (cur_aui2 + self.batch_size > self.aui_len):
                    if (cur_aui1 < self.end_aui1):
                        cur_aui2 = cur_aui2 + self.batch_size - self.aui_len
                        cur_aui1 += 1
                    else:
                        batch_max_len = self.aui_len + self.batch_size - cur_aui2
                else:
                    cur_aui2 += self.batch_size

        return self.batch_idx

    def get_batch_idx(self):
        return self.batch_idx
   
    def __len__(self):
        "Denotes the number of batches per epoch"
        return self.num_batches

    def __getitem__(self, index):
        "Generate one batch of data"
        logger.info("Batch {}".format(index))
        # Get indexes of the batch
        if index <= self.num_batches:
            if self.exp_flavor == 1:
                left, right = self.__read_batch_from_indexes(index)
                return [left, right]
            elif self.exp_flavor == 2:
                left, context_left, right, context_right = self.__read_batch_from_indexes(index)  
                return [left, context_left, right, context_right]
      
    def __read_batch_from_indexes(self, index):
        "Generates data containing batch_size samples" # X : (n_samples, *dim, embedding_dim)
        # Initialization
        aui1, start_aui2, batch_max_len, batch_len = self.batch_idx[index]
        
        lefts = np.empty((batch_len, self.max_seq_length), dtype = int)
        rights = np.empty((batch_len, self.max_seq_length), dtype = int)
    
        if self.exp_flavor == 2:
            # Adding context vector
            context_lefts = np.empty((batch_len, FLAGS.context_vector_dim), dtype = float)
            context_rights = np.empty((batch_len, FLAGS.context_vector_dim), dtype = float)

        if self.exp_flavor == 1:
            # Generate data
            aui2 = start_aui2
            i = 0
            for idx in range(batch_max_len):
                if (self.aui2cui[aui1] != self.aui2cui[aui2]):
                #if (aui1 in self.aui2cui) and (aui2 in self.aui2cui) and (self.aui2cui[aui1] != self.aui2cui[aui2]):
                    lefts[i,] = self.aui2vec[aui1]
                    rights[i,] = self.aui2vec[aui2]
                    i += 1
                if aui2 == self.aui_len - 1:
                    aui2 = 0
                    aui1 += 1
                else:
                    aui2 += 1
                    
            return lefts, rights
        elif self.exp_flavor == 2:
            # Generate data
            aui2 = start_aui2
            i = 0
            for idx in range(batch_max_len):
                if (self.aui2cui[aui1] != self.aui2cui[aui2]):
                #if (aui1 in self.aui2cui) and (aui2 in self.aui2cui) and (self.aui2cui[aui1] != self.aui2cui[aui2]):
                    lefts[i,] = self.aui2vec[aui1]
                    rights[i,] = self.aui2vec[aui2]
                    context_lefts[i,] = self.aui2convec[aui1]
                    context_rights[i,] = self.aui2convec[aui2]
                    i += 1
                if aui2 == self.aui_len - 1:
                    aui2 = 0
                    aui1 += 1
                else:
                    aui2 += 1

            return lefts, context_lefts, rights, context_rights
        
class FileGenerator(tf.keras.utils.Sequence):
    "Generates data for Keras"
    def __init__(self, fp, aui2vec, aui2convec, aui2id, max_seq_length, embedding_dim, word_embedding, context_vector_dim, exp_flavor, batch_size = 32, 
                 shuffle = False, is_test = True, n_classes = 2):
        "Initialization"
        self.fp = fp
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.word_embedding = word_embedding
        self.context_vector_dim = context_vector_dim
        self.exp_flavor = exp_flavor
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.aui2vec = aui2vec
        self.aui2convec = aui2convec
        self.aui2id = aui2id
        self.is_test = is_test
        
        self.file_size = get_size(self.fp, self.aui2vec, self.aui2id)
        self.pointer_queues = dict()
        self.files = list()
        
#         self.csv_reader = csv.DictReader(self.f, fieldnames = ["jacc", "AUI1", "AUI2", "Label"], delimiter = FLAGS.delimiter)
        
        self.batches = math.ceil(self.file_size / self.batch_size) 
        self.init_pointer_queues()
    def init_pointer_queues(self):
        self.pointer_queues = dict()
        self.files = list()
#         self.f = open(self.fp, "r")
        # Create a dict of queues for each index, a pointer will rotate from one queue to the next queue and return to the first
        for i in range(0, self.batches):
            # Create one index for each batch, list(islice(f, index*self.batch_size, (index+1)*self.batch_size)
            self.pointer_queues[i] = queue.Queue()
        for j in range(0, FLAGS.generator_workers):
            csv_reader = self.create_new_reader()
            self.pointer_queues[0].put(csv_reader)
            
    def close_pointer_queues(self):
        for f in self.files:
            f.close()
        del self.pointer_queues
        del self.files
        
    def create_new_reader(self):
        f = open(self.fp, "r")
        self.files.append(f)
        csv_reader = csv.DictReader(f, fieldnames = ["jacc", "AUI1", "AUI2", "Label"], delimiter = FLAGS.delimiter)
        return csv_reader
    def __len__(self):
        "Denotes the number of batches per epoch"
        return self.batches

    def __getitem__(self, index):
        "Generate one batch of data"
        # Get reader of the index
#         if (index + 1)*self.batch_size < self.file_size:
#             lines = islice(self.csv_reader, index*self.batch_size, (index+1)*self.batch_size)
#         else:
#             lines = islice(self.csv_reader, index*self.batch_size, self.file_size)
            
        # Slice the lines
        if index >= self.batches:
            return None
        elif (index < self.batches and index >= 0):
            if self.pointer_queues[index].empty():
                csv_reader = self.create_new_reader()
                self.pointer_queues[index].put(csv_reader)
            else:
                csv_reader = self.pointer_queues[index].get()
            if index < self.batches-1:
                lines = list(islice(csv_reader, index*self.batch_size, (index+1)*self.batch_size))
                self.pointer_queues[index+1].put(csv_reader)
            else:
                lines = list(islice(csv_reader, index*self.batch_size, self.file_size))
                csv_reader = self.create_new_reader()
                self.pointer_queues[0].put(csv_reader)
                         
        if self.is_test:
            logging.debug("Generating data for test")
            if self.exp_flavor == 1:
                left, right = self.__read_batch_from_file(lines)
                test_list(lines, "test %s lines with batch_size=%d, batches=%d at index %d with len %d"%(self.fp, self.batch_size, self.batches, index,len(lines)))
                test_list(left, "test %s left with batch_size=%d, batches=%d at index %d with len %d"%(self.fp, self.batch_size, self.batches, index,len(left)))
                return [left, right]
            elif self.exp_flavor == 2:
                left, context_left, right, context_right = self.__read_batch_from_file(lines)
                return [left, context_left, right, context_right]
        else:
            logging.debug("Generating data for non-test")
            if self.exp_flavor == 1:
                left, right, labels = self.__read_batch_from_file(lines)
                return [left, right], labels
            elif self.exp_flavor == 2:
                left, context_left, right, context_right, labels = self.__read_batch_from_file(lines)
                return [left, context_left, right, context_right], labels

    def on_epoch_end(self):
        "Close the file and open a new one"
        self.close_pointer_queues()
        self.init_pointer_queues()
        
#         if self.shuffle:
#             self.close_pointer_queues()
#             shuffle_file(self.fp, self.fp)
                
    def __read_batch_from_file(self, lines):
        # BioWordVec embedding matrix        
        batch_size = len(lines)
        lefts = np.empty((batch_size, self.max_seq_length), dtype=int)
        rights = np.empty((batch_size, self.max_seq_length), dtype=int)
        labels = None
        if not self.is_test:
            labels = np.empty((batch_size), dtype=int)
        
        if self.exp_flavor == 2:
            # Adding context vector
            context_lefts = np.empty((batch_size, FLAGS.context_vector_dim), dtype = float)
            context_rights = np.empty((batch_size, FLAGS.context_vector_dim), dtype = float)        
       
        if self.exp_flavor == 1:
            for i, line in enumerate(lines):
                id1 = self.aui2id[line["AUI1"]]
                id2 = self.aui2id[line["AUI2"]]
                if (id1 in self.aui2vec) and (id2 in self.aui2vec):
                    lefts[i,] = self.aui2vec[id1]
                    rights[i,] = self.aui2vec[id2]
                    if not self.is_test:
                        labels[i] = int(line["Label"])
            if not self.is_test:
                return lefts, rights, labels
            else:
                return lefts, rights

        elif self.exp_flavor == 2:
            for i, line in enumerate(lines):
                id1 = self.aui2id[line["AUI1"]]
                id2 = self.aui2id[line["AUI2"]]
                if (id1 in self.aui2vec) and (id2 in self.aui2vec):
                    lefts[i,] = self.aui2vec[id1]
                    rights[i,] = self.aui2vec[id2]
                    context_lefts[i,] = self.aui2convec[id1]
                    context_rights[i,] = self.aui2convec[id2]
                    if not self.is_test:
                        labels[i] = int(line["Label"])
            if not self.is_test:
                return lefts, context_lefts, rights, context_rights, labels
            else:
                return lefts, context_lefts, rights, context_rights 


class MetricsPerEpochCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_generator, test_partition, mrc_atoms, dirpaths, load_IDs):
        self.test_generator = test_generator
        self.test_partition = test_partition
        self.dirpaths = dirpaths
        self.mrc_atoms = mrc_atoms
        self.load_IDs = load_IDs

        return
        
    def on_train_begin(self, logs = {}):
        self.training_logs = dict()
        self.training_logs["training"] = dict()
        self.training_logs["training"]["epoch"] = list()
        self.metrics_per_epoch_test = dict()

        if FLAGS.predict_test_dir_after_every_epoch:
            for fp in self.test_partition.keys():
                
                self.metrics_per_epoch_test[fp] = dict()
                for metric in FLAGS.metric_names:
                    self.metrics_per_epoch_test[fp][metric] = list()
        return
    
    #def on_train_end(self, logs = {}):
        #del self.test_generator
        #del self.test_partition
        #gc.collect()
        
    def on_epoch_end(self, epoch, logs = {}):
        
        if FLAGS.predict_test_dir_after_every_epoch:
            # compute scores for test files
            for fp in self.test_partition.keys():
                start_time = time.time()
                if self.load_IDs:
                    scores, predictions = predict_generator(self.test_generator[fp], self.test_partition[fp].values(),  
                                                            len(self.test_partition[fp]), self.model, FLAGS.batch_size, epoch = epoch,
                                                            log_scores = self.metrics_per_epoch_test[fp]) 
                    logger.info("Finished predicting %s from memory in %s sec."%(fp, time.time() - start_time))    
                else:
                    scores, predictions = predict_generator(self.test_generator[fp], self.test_partition[fp], 
                                                            len(self.test_partition[fp]), self.model, FLAGS.batch_size, epoch = epoch,
                                                            log_scores = self.metrics_per_epoch_test[fp])
                    
                    logger.info("Finished predicting %s from loading file in %s sec."%(fp, time.time() - start_time))    
                print(fp)
                print(scores)
                logger.info(fp)
                logger.info(scores)

                save_metrics(self.dirpaths["train_result_dp"], FLAGS.tmp_metrics_fn, self.metrics_per_epoch_test)
                save_metrics_graph(self.dirpaths["train_result_dp"], FLAGS.tmp_metrics_graph_fn,  self.metrics_per_epoch_test)
                
                #if epoch == FLAGS.n_epoch:

                    #predictions_fp = os.path.join(dirpaths["predict_result_dp"], "_".join([get_base(fp), str(epoch), FLAGS.predictions_fn]))
                    #compare_predictions(labels, predictions, self.mrc_atoms, fp, predictions_fp, epoch)
                del predictions
        # Saving training history until this epoch
        
        self.training_logs["training"]["epoch"].append(epoch)
#         print(logs)
        for metric in logs.keys():
            if metric not in self.training_logs["training"]:
                self.training_logs["training"][metric] = list()
            self.training_logs["training"][metric].append(round(logs[metric],4))
        save_metrics(self.dirpaths["train_result_dp"], FLAGS.tmp_history_fn, self.training_logs)
        save_metrics_graph(self.dirpaths["train_result_dp"], FLAGS.tmp_history_graph_fn, self.training_logs)
        
        return
    
    def get_metrics(self):
        return self.metrics_per_epoch_test
    
class ManDist(Layer):
    """
        Keras Custom Layer that calculates Manhattan Distance.
    """

    # initialize the layer, No need to include inputs parameter!
    def __init__(self, **kwargs):
        self.result = None
        super(ManDist, self).__init__(**kwargs)

    # input_shape will automatic collect input shapes to build layer
    def build(self, input_shape):
        super(ManDist, self).build(input_shape)

    # This is where the layer"s logic lives.
    def call(self, x, **kwargs):
        self.result = K.exp(-K.sum(K.abs(x[0] - x[1]), axis = 1, keepdims = True))
        return self.result
    # return output shape
    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)

class CosineSim(Layer):
    """
        Keras Custom Layer that calculates Consine Distance.
    """

    # initialize the layer, No need to include inputs parameter!
    def __init__(self, **kwargs):
        self.result = None
        super(CosineSim, self).__init__(**kwargs)

    # input_shape will automatic collect input shapes to build layer
    def build(self, input_shape):
        super(CosineSim, self).build(input_shape)

    # This is where the layer"s logic lives.
    def call(self, x, **kwargs):
        self.result = tf.keras.layers.Dot(axes=(1,1), normalize=True)([x[0], x[1]])
        return self.result
    # return output shape
    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)

class attention(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")        
        super(attention, self).build(input_shape)

    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(attention,self).get_config()
    
class SharedDropout(tf.keras.layers.Layer):
    # Ref: https://github.com/keras-team/keras/issues/8802
    # learnt this from this link: https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/ops/nn_ops.py
    def __init__(self, keep_prob_rate = 0.5, **kwargs):
        self.keep_prob_rate = keep_prob_rate
        super(SharedDropout, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SharedDropout, self).build(input_shape)

    def call(self, inputs, **kwargs):
        input_left = inputs[0]
        input_right = inputs[1]
        random_tensor = self.keep_prob_rate
        random_tensor +=  tf.compat.v1.random_uniform(tf.shape(input_left), dtype = input_left.dtype)
        binary_tensor = tf.floor(random_tensor)

        def DropoutLeft():
            ret_left = tf.divide(input_left, self.keep_prob_rate) * binary_tensor
            return ret_left

        def DropoutRight():
            ret_right = tf.divide(input_right, self.keep_prob_rate) * binary_tensor
            return ret_right

        return [K.in_train_phase(DropoutLeft(), input_left, training = None), K.in_train_phase(DropoutRight(), input_right, training = None)]

    def compute_output_shape(self, input_shapes):
        return [(input_shapes[0][0], input_shapes[0][1]), (input_shapes[1][0], input_shapes[1][1])]
    
def read_file_to_datagenerator(fp, aui2vec, aui2id):
    pickle_fp = fp + "_ID_LABEL.PICKLE"
    if (os.path.isfile(pickle_fp)):
        logger.info("Loading file %s ..."%pickle_fp)   
        partition = load_pickle(pickle_fp)
    else:
        logger.info("Loading file %s ..."%fp)      

        partition = dict()
        with open(fp, "r") as fi:
            reader = csv.DictReader(fi, fieldnames = ["jacc", "AUI1", "AUI2", "Label"], delimiter = FLAGS.delimiter)
            with tqdm(total = count_lines(fp)) as pbar:
                for line in reader:
                    pbar.update(1)

                    if ((len(line) == 4) and (aui2id[line["AUI1"]] in aui2vec) and (aui2id[line["AUI2"]] in aui2vec)):
                        ID = (aui2id[line["AUI1"]], aui2id[line["AUI2"]])
                        partition[ID] = int(line["Label"])
        dump_pickle(partition, pickle_fp)

    test_dict(partition)
    return partition

def read_file_to_filegenerator(fp, aui2vec, aui2id):
    
    logger.info("Loading file %s ..."%fp)      
    
    partition = list()
    with open(fp, "r") as fi:
        reader = csv.DictReader(fi, fieldnames = ["jacc", "AUI1", "AUI2", "Label"], delimiter = FLAGS.delimiter)
        with tqdm(total = count_lines(fp)) as pbar:
            for line in reader:
                pbar.update(1)
                
                if ((len(line) == 4) and (aui2id[line["AUI1"]] in aui2vec) and (aui2id[line["AUI2"]] in aui2vec)):
                    partition.append(int(line["Label"]))
                    
    test_list(partition)
    return partition

def read_file_to_datagenerator_jacc_label(fp, aui2vec, aui2id):
    pickle_fp = fp + ".PICKLE"
    if (os.path.isfile(pickle_fp)):
        logger.info("Loading file %s ..."%pickle_fp)
        partition = load_pickle(pickle_fp)
    else:
        logger.info("Loading file %s ..."%fp)

        partition = dict()
        with open(fp, "r") as fi:
            reader = csv.DictReader(fi, fieldnames = ["jacc", "AUI1", "AUI2", "Label"], delimiter = FLAGS.delimiter)
            with tqdm(total = count_lines(fp)) as pbar:
                for line in reader:
                    pbar.update(1)

                    if ((len(line) == 4) and (aui2id[line["AUI1"]] in aui2vec) and (aui2id[line["AUI2"]] in aui2vec)):
                        ID = (aui2id[line["AUI1"]], aui2id[line["AUI2"]])
                        partition[ID] = (float(line["jacc"]),int(line["Label"]))

        dump_pickle(partition, pickle_fp)

    test_dict(partition)
    return partition

def get_size(fp, aui2vec, aui2id):
    cnt_size = 0
    logger.info("Getting size from file %s ..."%fp)      
    with open(fp, "r") as fi:
        reader = csv.DictReader(fi, fieldnames = ["jacc", "AUI1", "AUI2", "Label"], delimiter = FLAGS.delimiter)
        with tqdm(total = count_lines(fp)) as pbar:
            for line in reader:
                pbar.update(1)
                if ((aui2id[line["AUI1"]] in aui2vec) and (aui2id[line["AUI2"]] in aui2vec)):
                    cnt_size += 1
    return cnt_size

def gen_aui2id(mrconso_master_fp, aui2id_fp, id2aui_fp):
    id2aui = dict()
    aui2id = dict()
    logger.info("Loading file %s ..."%mrconso_master_fp)      
    with open(mrconso_master_fp, "r") as fi:
        reader = csv.DictReader(fi, fieldnames = FLAGS.mrconso_master_fields, delimiter = FLAGS.delimiter)
        with tqdm(total = count_lines(mrconso_master_fp)) as pbar:
            for line in reader:
                pbar.update(1)
                aui2id[line["AUI"]] = int(line["ID"])
                id2aui[int(line["ID"])] = line["AUI"]
    dump_pickle(aui2id, aui2id_fp)
    dump_pickle(id2aui, id2aui_fp)
    return 

def predict_all_negatives(generator, model, test_size, batch_size):
    start_time = time.time()
    predict = model.predict(generator,
                            steps = math.ceil(test_size/batch_size),
                            use_multiprocessing = True,
                            workers = FLAGS.generator_workers,
                            verbose = FLAGS.predict_verbose)
    logger.info("Predicting time: {}".format(time.time()-start_time))
    FP = (predict >= 0.5).sum()
    #predictions = (x[0] for x in predict)
    #y_pred = np.empty((test_size), dtype=int)
    #FP = 0
    #for x in predict:
    #    x_0 = (k for k in [x[0]])
    #    for k in x_0:
            #y_pred[i] = 1 if k >=  FLAGS.predict_threshold else 0
    #        FP += 1 if k >=  FLAGS.predict_threshold else 0
    #FP = sum(y_pred)
    TN = test_size - FP
    return TN, FP

def predict_generator(generator, labels, labels_size, model, batch_size, epoch = None, log_scores = None):
    #logger.debug("labels: {}".format(labels))
    classes = {}
    test_size = 0
    y_true = np.empty((labels_size), dtype=int)
    for i, label in enumerate(labels):
        y_true[i] = label
        classes[label] = True
        test_size += 1
    del labels
   
    logger.info("Classes: %d of %s"%(len(classes), classes))
    start_time = time.time()
    predict = model.predict(generator, 
#                             batch_size = FLAGS.batch_size, 
                            steps = math.ceil(test_size/batch_size),
                            use_multiprocessing = True,
                            workers = FLAGS.generator_workers,
                            verbose = FLAGS.predict_verbose)
    
    logger.info("Predicting time: {}".format(time.time()-start_time))
    if FLAGS.PredictionLayer == "Threshold":
        predictions = [x[0] for x in predict]
        test_list(predictions, "test predictions")
        y_pred = np.empty((len(predictions)), dtype=int)
        y_pred_best_threshold = np.empty((len(predictions)), dtype=int)
        # y_pred_best_threshold_roc = np.empty((len(predictions)), dtype=int)

        precision, recall, thresholds = precision_recall_curve(y_true, predictions)
        f1_score = (2 * precision * recall) / (precision + recall)
        index_of_best_f1_score = np.argmax(f1_score)
        logger.info("Best Precision Recall Threshold=%f, F1-Score=%f, Precision=%f, Recall=%f" % (thresholds[index_of_best_f1_score], f1_score[index_of_best_f1_score], precision[index_of_best_f1_score], recall[index_of_best_f1_score]))
        
        
        for i, x in enumerate(predictions):
            y_pred[i] = 1 if x >=  FLAGS.predict_threshold else 0
            y_pred_best_threshold[i] = 1 if x >= thresholds[index_of_best_f1_score] else 0
    else:
        predictions = [np.max(x) for x in predict]
        y_pred = [np.argmax(x) for x in predict]
        for x in predict:
            logger.info("x:{}, np.max: {}, np.argmax: {}".format(x, np.max(x), np.argmax(x)))


            
    
    #logger.debug("y_true: {}".format(y_true))
    #logger.debug("y_pred: {}".format(y_pred))
    

    accuracy, precision, recall, f1 = compute_scores(y_true, y_pred)
    best_threshold_accuracy, best_threshold_precision, best_threshold_recall, best_threshold_f1 = compute_scores(y_true, y_pred_best_threshold)
    auc = round(roc_auc_score(y_true, predictions), 4)
    del predictions
    scores = {"epoch": epoch, "accuracy":accuracy, "precision":precision, "recall":recall, "f1":f1, "auc": auc}
    best_threshold_scores = { "best_threshold" : thresholds[index_of_best_f1_score], "epoch": epoch, "best_threshold_accuracy": best_threshold_accuracy, "best_threshold_precision": best_threshold_precision, "best_threshold_recall": best_threshold_recall, "best_threshold_f1": best_threshold_f1, "auc": auc}
    scores.update(best_threshold_scores)
    if (log_scores is not None) and (epoch is not None):
        for metric in FLAGS.metric_names:
            log_scores[metric].append(scores[metric])

    return scores, y_pred

def compute_scores(y_true, y_pred):
    TP, TN, FP, FN = 0, 0, 0, 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            TP += 1
        elif t == 1 and p == 0:
            FN += 1
        elif t == 0 and p == 1:
            FP += 1
        else:
            TN += 1
    accuracy = (TP + TN)/(TP + TN + FP + FN)
    precision, recall, f1 = 0, 0, 0
    
    if TP + FP > 0:
        precision = TP/(TP+FP)
        
    if TP + FN > 0:
        recall = TP/(TP+FN)
        
    if recall + precision > 0:
        f1 = 2*(recall * precision) / (recall + precision)
    
    return round(accuracy,4), round(precision,4), round(recall,4), round(f1,4)
        
def do_analyze(mrc_atoms, test_fp, aui2vec, aui2id, id2aui, predictions_fp, epoch):
    results = dict()
    fp = predictions_fp + "_predictions_" + str(epoch)
    logger.info("Loading file .. %s"%fp)
    if os.path.isfile(fp + ".PICKLE"):
        results = load_pickle(fp + ".PICKLE")
    else:
        with open(fp, "r") as fi:
            reader = csv.DictReader(fi, fieldnames = ["label","DL","RBA"], delimiter=FLAGS.delimiter,doublequote=False,quoting=csv.QUOTE_NONE)
            for idx, line in enumerate(reader):
                results[idx] = dict()
                results[idx].update(line)

    logger.info("Results size: %s"%(len(results)))

    dl_correct_file = open(predictions_fp + "_DL_correct" + str(epoch), "w")
    rba_correct_file = open(predictions_fp + "_RBA_correct_" + str(epoch), "w")
    stat_jacc_file = open(predictions_fp + "_stat_jacc_" + str(epoch), "w")
    inputs = read_file_to_datagenerator_jacc_label(test_fp, aui2vec, aui2id)
    logger.info("Inputs size: %s"%(len(inputs)))
    
    stat_jacc = dict()
    stat_jacc["DL"] = dict()
    stat_jacc["DL"][1] = dict()
    stat_jacc["DL"][1]["correct"] = dict()
    stat_jacc["DL"][1]["incorrect"] = dict()
    stat_jacc["DL"][1]["correct"]["zero"] = 0
    stat_jacc["DL"][1]["incorrect"]["zero"] = 0
    stat_jacc["DL"][1]["correct"]["notzero"] = 0
    stat_jacc["DL"][1]["incorrect"]["notzero"] = 0
    stat_jacc["DL"][0] = dict()
    stat_jacc["DL"][0]["correct"] = dict()
    stat_jacc["DL"][0]["incorrect"] = dict()
    stat_jacc["DL"][0]["correct"]["zero"] = 0
    stat_jacc["DL"][0]["incorrect"]["zero"] = 0
    stat_jacc["DL"][0]["correct"]["notzero"] = 0
    stat_jacc["DL"][0]["incorrect"]["notzero"] = 0

    stat_jacc = dict()
    stat_jacc[1] = dict()
    stat_jacc[1]["correct"] = dict()
    stat_jacc[1]["incorrect"] = dict()
    stat_jacc[1]["correct"]["zero"] = 0
    stat_jacc[1]["incorrect"]["zero"] = 0
    stat_jacc[1]["correct"]["notzero"] = 0
    stat_jacc[1]["incorrect"]["notzero"] = 0
    ddstat_jacc[0] = dict()
    stat_jacc[0]["correct"] = dict()
    stat_jacc[0]["incorrect"] = dict()
    stat_jacc[0]["correct"]["zero"] = 0
    stat_jacc[0]["incorrect"]["zero"] = 0
    stat_jacc[0]["correct"]["notzero"] = 0
    stat_jacc[0]["incorrect"]["notzero"] = 0

    with tqdm(total = len(inputs)) as pbar:
        for idx, ID in enumerate(inputs.keys()):
            atom1 = mrc_atoms[id2aui[ID[0]]]
            atom2 = mrc_atoms[id2aui[ID[1]]]
            if (results[idx]["label"] == results[idx]["DL"]) and (results[idx]["label"] != results[idx]["RBA"]):
                print_pair_info(dl_correct_file, id2aui, ID, inputs, atom1, atom2, results[idx]["DL"], results[idx]["RBA"])
                stat_jacc = update_stat_jacc(stat_jacc, ID, inputs, inputs[ID][1])

            elif (results[idx]["label"] != results[idx]["DL"]) and (results[idx]["label"] == results[idx]["RBA"]):
                print_pair_info(rba_correct_file, id2aui, ID, inputs, atom1, atom2, results[idx]["DL"], results[idx]["RBA"])
                stat_jacc = update_stat_jacc(stat_jacc, ID, inputs, inputs[ID][1])
            pbar.update(1)
    logger.info(stat_jacc)
    print(stat_jacc, file=stat_jacc_file)
    dl_correct_file.close()
    rba_correct_file.close()
    stat_jacc_file.close()

def update_stat_jacc(stat_jacc, ID, inputs, pos_or_neg):
    if DL_or_RBA == "RBA":
        if inputs[ID][0] > 0:
            stat_jacc[pos_or_neg]["incorrect"]["notzero"] += 1 
        else:
            stat_jacc[pos_or_neg]["incorrect"]["zero"] += 1
    else:
        if inputs[ID][0] > 0:
            stat_jacc["DL"][pos_or_neg]["correct"]["notzero"] += 1
        else:
            stat_jacc["DL"][pos_or_neg]["correct"]["zero"] += 1

    return stat_jacc

def save_predicted_results(y_pred, predictions_fp, epoch):
    dump_pickle(y_pred, predictions_fp + "_predictions_" + str(epoch))

def compare_predictions(y_true, y_pred, mrc_atoms, test_fp, aui2vec, aui2id, id2aui, predictions_fp, epoch, rba_pred_pickle_fp):
    rba_pred = load_pickle(rba_pred_pickle_fp)
    labels = list(y_true.values())
    with open(predictions_fp + "_predictions_" + str(epoch), "w") as fo:
        for label, pred, rba_p in zip(labels, y_pred, rba_pred):
            fo.write(str(label) + "|" + str(pred) + "|" + str(rba_p) + "\n")

    tp = open(predictions_fp + "_TP_" + str(epoch), "w")
    tn = open(predictions_fp + "_TN_" + str(epoch), "w")
    fp = open(predictions_fp + "_FP_" + str(epoch), "w")
    fn = open(predictions_fp + "_FN_" + str(epoch), "w")
    lines = read_file_to_datagenerator_jacc_label(fp, aui2vec, aui2id)
    y_r = 0
    for line, t, p, r in zip(lines, labels, y_pred, rba_pred):
        atom1 = mrc_atoms[line["AUI1"]]
        atom2 = mrc_atoms[line["AUI2"]]
        if t == 1 and p == 1:
            print_pair_info(tp, id2aui, line, atom1, atom2, p, r)
        elif t == 1 and p == 0:
            print_pair_info(fn, id2aui, line, atom1, atom2, p, r)
        elif t == 0 and p == 1: 
            print_pair_info(fp, id2aui, line, atom1, atom2, p, r)
        else:
            print_pair_info(tn, id2aui, line, atom1, atom2, p, r)
                        
def gen_mrc_atoms(mrconso_master_fp, mrc_atoms_pickle_fp):
    mrc_atoms = {}
    logger.info("Loading mrc_atoms from %s ..."%mrconso_master_fp)
    with tqdm(total = count_lines(mrconso_master_fp)) as pbar:
        with open(mrconso_master_fp, "r") as fi:
            reader = csv.DictReader(fi, fieldnames = FLAGS.mrconso_master_fields, delimiter=FLAGS.delimiter,doublequote=False,quoting=csv.QUOTE_NONE)
            idx = 0
            for i, line in enumerate(reader):
                pbar.update(1)
                mrc_atoms[line["AUI"]] = {"CUI":line["CUI"],"LUI":line["LUI"],\
                                              "SCUI": line["SCUI"], "SG": line["SG"],\
                                                 "STR": line["STR"], "ID": line["ID"]}
    dump_pickle(mrc_atoms, mrc_atoms_pickle_fp) 
    return

def print_pair_info(fo, id2aui, ID, inputs, atom1, atom2, d, r):
    
    fo.write("%s|%s|%s|label=%s|DL=%s|RBA=%s\n"%(inputs[ID][0], id2aui[ID[0]], id2aui[ID[1]], inputs[ID][1], d, r))
    fo.write("%s|%s|%s|%s|%s|%s\n"%(id2aui[ID[0]], atom1["CUI"], atom1["LUI"], atom1["SCUI"], atom1["SG"], atom1["STR"]))
    fo.write("%s|%s|%s|%s|%s|%s\n"%(id2aui[ID[1]], atom2["CUI"], atom2["LUI"], atom2["SCUI"], atom2["SG"], atom2["STR"]))
    return
    
def create_folder(folder):
    """
        Create the folder.
    """
    Path(folder).mkdir(parents = True, exist_ok = True)

    return folder
    
def make_word_embeddings(embedding_filepath, vocab_length, tokenizer):
    """
        Generate embedding matrix from word2vec pre_trained model.
        Parameter:
            vocab_length: The length of vocalbulary. 
            tokenizer: The token dictionary.
    """

    embedding_dict = {}
    with open(embedding_filepath, "r") as f:
        header = f.readline()
        line = f.readline()

        while line !=  "":
            values = line.split()
            word = values[0]
            vector_w2v = np.asarray(values[1:], dtype = "float32")
            embedding_dict[word] = vector_w2v

            line = f.readline()

    embedding_matrix = np.zeros((vocab_length, FLAGS.embedding_dim))

    for word, i in tokenizer.word_index.items():
        embedding_vector = embedding_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

def make_bert_embedding_matrix(embedding_fp, vocab_file):
    embedding_dict = load_pickle(embedding_fp)
#    vocab_file = embedding_fp + "_vocab"
#    with open(vocab_file, "w") as f:
#        for token in embedding_dict.keys():
#            f.write(token + "\n")
#
    embedding_matrix = np.zeros((len(embedding_dict), FLAGS.embedding_dim))    
    tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file = vocab_file)
    for token, emb in embedding_dict.items():
        token_id = tokenizer.convert_tokens_to_ids([token])[0]
        embedding_matrix[token_id] = emb

    return embedding_matrix


def save_history(dp, history_fn, trained_history):
    """
        Save history of epochs.
        Parameter:
            malstm_trained: The trained model.
    """
    metrics = [metric for metric in trained_history.history.keys()]
    
    fp = os.path.join(dp, history_fn)
    with open(fp, "a") as f:
        s = FLAGS.delimiter
        f.write("For test dataset %s \n"%fp)
        f.write(FLAGS.delimiter.join(metrics))
        f.write("\n")
        for i, epoch in enumerate(trained_history.history[metrics[0]]):
            scores = [trained_history.history[metric][i] for metric in metrics]
            f.write(FLAGS.delimiter.join(["%.4f"%(score) for score in scores]))
            f.write("\n")
    
    return

def save_history_graph(dp, history_graph_fn, trained_history):
    """
        Generate the graph. It includes:
            Accuracy.
            Loss.
        Parameter:
            malstm_trained: The trained model.
    """
    metrics = [metric for metric in trained_history.history.keys() if "val" not in metric]
    epochs = [i for i in enumerate(trained_history.history[metrics[0]])]
    for metric in metrics:
        plt.clf()
        plt.subplot(211)
        plt.plot(trained_history.history[metric])
        plt.plot(trained_history.history["val_" + metric])
        plt.title("Learning Curves for %s"%(metric))
        plt.ylabel("Score")
        plt.xlabel("Epoch")
        plt.legend(["Training " + metric, "Validation " + metric], loc = "upper left")
        plt.tight_layout(h_pad = 1.0)
        
        fp = os.path.join(dp, "%s_%s"%(metric, history_graph_fn))
        plt.savefig(fp)
    
def get_base(fn):
    base = os.path.basename(fn)
    name = os.path.splitext(base)[0]
    return name

def save_metrics(dp, metrics_fn, test_metrics):
    """
        Save history of epochs.
        Parameter:
            malstm_trained: The trained model.
    """
                
    if test_metrics is not None:
        # write test metrics
        for test_fn in test_metrics.keys():
            # metrics = [metric for metric in test_metrics[test_fn].keys()]
            metrics = []
            for metric in test_metrics[test_fn].keys():
                if "best" not in metric:
                    metrics.append(metric)
            logger.info("{}".format(metrics))
            name = get_base(test_fn)
            with open(os.path.join(dp,"%s_%s_%s"%(name, metrics_fn, FLAGS.DistanceScore)), "w") as f:
                s = FLAGS.delimiter
                f.write(name)
                f.write("\n")
                f.write(FLAGS.delimiter.join(metrics))
                f.write("\n")
                for i, epoch in enumerate(test_metrics[test_fn]["epoch"]):
                    scores = []
                    scores = [test_metrics[test_fn][metric][i] for metric in metrics]
                    # print(scores)
                    # scores.append(test_metrics[test_fn][metric][i])
                    f.write(FLAGS.delimiter.join([str(score) for score in scores]))
                    f.write("\n")
        
        for test_fn in test_metrics.keys():
            #metrics = [metric for metric in test_metrics[test_fn].keys()]
            metrics = ["epoch",]
            for metric in test_metrics[test_fn].keys():
                if "best" in metric or "auc" in metric:
                    metrics.append(metric)
            #metrics.append("auc")
            logger.info("{}".format(metrics))
            name = get_base(test_fn)
            with open(os.path.join(dp,"best_threshold_%s_%s_%s"%(name, metrics_fn, FLAGS.DistanceScore)), "w") as f:
                s = FLAGS.delimiter
                f.write(name)
                f.write("\n")
                f.write(FLAGS.delimiter.join(metrics))
                f.write("\n")
                for i, epoch in enumerate(test_metrics[test_fn]["epoch"]):
                    scores = [test_metrics[test_fn][metric][i] for metric in metrics]
                    print(scores)
                    f.write(FLAGS.delimiter.join([str(score) for score in scores]))
                    f.write("\n")
            
    return
    
def save_metrics_graph(dp, metrics_graph_fn, test_metrics):
    if test_metrics is not None:
        # write test metrics
        for test_fn in test_metrics.keys():
            
            metrics = [metric for metric in test_metrics[test_fn].keys() if ((metric != "epoch") and ("val" not in metric))]
            for metric in metrics:
                plt.clf()
                plt.subplot(211)
                legends = [metric]
                name = get_base(test_fn)
                plt.plot(test_metrics[test_fn]["epoch"], test_metrics[test_fn][metric])
                if ("val_"+metric in test_metrics[test_fn].keys()):
                    plt.plot(test_metrics[test_fn]["epoch"], test_metrics[test_fn]["val_" + metric])
                    legends.append("val_" + metric)
                plt.title("%s %s"%(name, metric))
                plt.ylabel("Score")
                plt.xlabel("Epoch")
                plt.legend(legends, loc = "upper left")
                plt.tight_layout(h_pad = 1.0)
            
                
                fp = os.path.join(dp,"%s_%s_%s"%(name, metric, metrics_graph_fn))
                logger.info("Updating metrics graph in %s"%fp)
                plt.savefig(fp)
                plt.show()
    
def save_metrics_graph_old(filepath, test_metrics):

    """
        Generate the graph. It includes:
            Accuracy.
            Loss.
        Parameter:
            malstm_trained: The trained model.
    """
    epochs = [i for i in range(1, FLAGS.n_epoch+1)]
    scores = {}
            
    plt.clf()
    figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2)
    
    h1 = ax1.plot(epochs, scores["val_precision"], color = "green")[0]
    h2 = ax1.plot(epochs, scores["test_precision"], color = "orange")[0]
    ax1.set_ylabel("Precision")

    h3 = ax2.plot(epochs, scores["val_recall"], color = "green")[0]
    h4 = ax2.plot(epochs, scores["test_recall"], color = "orange")[0]
    ax2.set_ylabel("Recall")

    h5 = ax3.plot(epochs, scores["val_f1_score"], color = "green")[0]
    h6 = ax3.plot(epochs, scores["test_f1_score"], color = "orange")[0]
    ax3.set_ylabel("F1 Score")
    ax3.set_xlabel("Epoch")
    
    h7 = ax4.plot(epochs, scores["val_accuracy"], label = "val", color = "green")[0]
    h8 = ax4.plot(epochs, scores["test_accuracy"], label = "test", color = "orange")[0]
    ax4.set_ylabel("Accuracy")
    ax4.set_xlabel("Epoch")
    ax4.legend(bbox_to_anchor = (1.05, 1), loc = "upper left", borderaxespad = 0.)
    
    plt.suptitle("Valiation and Testing Metrics per Epoch", y = 0.95)
    plt.tight_layout()
    
    plt.savefig(filepath)
    #plt.show()

def count_lines(filein):
    return sum(1 for line in open(filein))

def load_pickle(pickle_fp):
    with open(pickle_fp, "rb") as f:
        obj = pickle.load(f)
    if type(obj) == list:
        test_list(obj)
    elif type(obj) == dict:
        test_dict(obj)
    return obj

def dump_pickle(obj, pickle_fp):
    with open(pickle_fp, "wb") as f:
        pickle.dump(obj, f, protocol = 4)
    return obj

def gen_tokenizer(mrconso_master_fp, tokenizer_pickle_fp):
    logger.info("Loading aui str list from %s ..."%mrconso_master_fp)
    auis_str = get_auis_str(mrconso_master_fp)
    
    tokenizer = Tokenizer()
    logger.info("Fitting aui str list from %s ..."%mrconso_master_fp)
    tokenizer.fit_on_texts(auis_str.values())
    
    logger.info("Writing tokenizer to %s ..."%tokenizer_pickle_fp)
    dump_pickle(tokenizer, tokenizer_pickle_fp)
    
    write_important_info("Tokenizer size: %d"%(len(tokenizer.word_index) + 1))
    test_dict(tokenizer.word_index)
    
    return tokenizer
    
def gen_aui2vec(tokenizer, aui2vec_pickle_fp, mrconso_master_fp):
    aui2vec = dict()
    logger.info("Loading aui list from %s ..."%mrconso_master_fp)
    auis_str = get_auis_str(mrconso_master_fp)
    logger.info("Convert aui str to sequences with padding...")
    with tqdm(total = len(auis_str)) as pbar:
        for aui_id, aui_str in auis_str.items():
            if aui_str !=  "":
                #Convert word to sequence and padding sequence.
                encoded_aui = tokenizer.texts_to_sequences([aui_str])
                if aui_id < 2:
                    logger.debug("tokenizing id = {}, aui = {} to {}".format(aui_id, aui_str, encoded_aui))
                padded_aui = pad_sequences(encoded_aui, maxlen = FLAGS.max_seq_length, 
                                            padding = FLAGS.padding_sequence, 
                                            truncating = FLAGS.truncating_sequence)
                aui2vec[aui_id] = padded_aui
            else:
                aui2vec[aui_id] = None
            pbar.update(1)
            
    test_dict(aui2vec)
    
    logger.info("Writing aui2vec to %s ..."%aui2vec_pickle_fp)
    dump_pickle(aui2vec, aui2vec_pickle_fp)
    
    return aui2vec

def gen_bert_aui2vec(vocab_file, aui2vec_pickle_fp, mrconso_master_fp):
    aui2vec = dict()
    logger.info("Loading aui list from %s ..."%mrconso_master_fp)
    auis_str = get_auis_str(mrconso_master_fp)
    logger.info("Convert aui str to sequences with padding...")
    tokenizer = bert.bert_tokenization.FullTokenizer(#max_length = FLAGS.max_seq_length, truncation = True, 
                                                     #padding = True, 
                                                     vocab_file = vocab_file)
    encoded_aui = None
    tokens = None
    with tqdm(total = len(auis_str)) as pbar:
        for aui_id, aui_str in auis_str.items():
            if aui_str !=  "":
                #Convert word to sequence and padding sequence.
                encoded_aui = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(aui_str)[:FLAGS.max_seq_length])
                if aui_id < 2:
                    logger.debug("tokenizing id = {}, aui = {} to {}".format(aui_id, aui_str, encoded_aui))
                padded_aui = pad_sequences([encoded_aui], maxlen = FLAGS.max_seq_length,
                                            padding = FLAGS.padding_sequence,
                                            truncating = FLAGS.truncating_sequence)
                aui2vec[aui_id] = padded_aui
            else:
                aui2vec[aui_id] = None
            pbar.update(1)

    test_dict(aui2vec)

    logger.info("Writing aui2vec to %s ..."%aui2vec_pickle_fp)
    dump_pickle(aui2vec, aui2vec_pickle_fp)

    return aui2vec

def test_dict(d, dn=None):
    for i, (k,v) in enumerate(d.items()):
        if i < 2:
            logger.debug("{}({}): {} -> {}".format(inspect.stack()[1][3], dn, k, v))
    return

def test_list(l, dn=None):
    for i, v in enumerate(l):
        if i < 2:
            logger.debug("{}({}): [{}] = {}".format(inspect.stack()[1][3], dn, i, v))
    return

def test_type(t, dn=None):
    logger.debug("{}({}): type({}) = {}".format(inspect.stack()[1][3],dn, t, type(t)))
    
def test_member(e, d, dn=None):
    if e not in d:
        logger.debug("{}({}): {} not in {}".format(inspect.stack()[1][3], dn, e, d))
    else:
        logger.debug("{}({}): t[{}] = {}".format(inspect.stack()[1][3], dn, e, d[e]))

def get_auis_str(mrconso_master_fp):
    auis_str = dict()
    with open(mrconso_master_fp, "r") as fi:
        reader = csv.DictReader(fi, fieldnames = FLAGS.mrconso_master_fields, delimiter = FLAGS.delimiter, doublequote = False, quoting = csv.QUOTE_NONE)
        with tqdm(total = count_lines(mrconso_master_fp)) as pbar:
            for line in reader:
                pbar.update(1)
                auis_str[int(line["ID"])] = line["STR"]
    test_dict(auis_str)
    return auis_str

def shuffle_file(file_in, file_out):
    lines = open(file_in).readlines()
    random.shuffle(lines)
    open(file_out, "w").writelines(lines)
    return

def get_logger(log_level, name, filepath):
    # get TF logger
    log = logging.getLogger(name)
    log.setLevel(log_level)

    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    ch = logging.StreamHandler()
    ch.setLevel(level = logging.DEBUG)
    ch.setFormatter(formatter)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(filepath)
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    log.addHandler(ch)
    log.addHandler(fh) 
    return log

def generate_dirpaths():
    dirpaths = {}
    dirpaths["umls_version_dp"] = FLAGS.umls_version_dp if FLAGS.umls_version_dp is not None else create_folder(os.path.join(FLAGS.umls_dp, FLAGS.umls_version_dn))

    dirpaths["dataset_version_dp"] = FLAGS.dataset_version_dp if FLAGS.dataset_version_dp is not None else create_folder(os.path.join(dirpaths["umls_version_dp"], FLAGS.dataset_version_dn))

    dirpaths["training_dp"] = create_folder(os.path.join(dirpaths["dataset_version_dp"], FLAGS.training_dn))
    dirpaths["logs_dp"] = create_folder(os.path.join(FLAGS.workspace_dp, FLAGS.logs_dn))
    dirpaths["extra_dp"] = create_folder(os.path.join(FLAGS.workspace_dp, FLAGS.extra_dn))
    
    dirpaths["train_dataset_dn"] = os.path.basename(FLAGS.train_dataset_dp)
    use_shared_dropout = "withdropout" if FLAGS.use_shared_dropout_layer is True else "nodropout"
    dirpaths["output_dn"] = "%s_%s_%db_%dep_%s_exp%d_model%s_variant%s_wordemb%s"%(dirpaths["train_dataset_dn"], FLAGS.run_id, FLAGS.batch_size, FLAGS.n_epoch, use_shared_dropout, FLAGS.exp_flavor, FLAGS.Model, FLAGS.ConVariant, FLAGS.WordEmbVariant)
    dirpaths["output_dp"] = create_folder(os.path.join(dirpaths["training_dp"], dirpaths["output_dn"]))
    
    dirpaths["train_result_dp"] = create_folder(os.path.join(dirpaths["output_dp"], FLAGS.train_result_dn))
    dirpaths["checkpoint_dp"] = create_folder(os.path.join(dirpaths["output_dp"], FLAGS.checkpoint_dn))
    dirpaths["predict_result_dp"] = create_folder(os.path.join(dirpaths["output_dp"], FLAGS.predict_result_dn))
    
    return dirpaths

def create_model(embedding_matrix):

    #Define Input for model
    left_base_input = Input(shape = (FLAGS.max_seq_length, ), dtype = "int32")
    right_base_input = Input(shape = (FLAGS.max_seq_length, ), dtype = "int32")
    left_enrich_input = Input(shape = (FLAGS.context_vector_dim), dtype = "float32")
    right_enrich_input = Input(shape = (FLAGS.context_vector_dim), dtype = "float32")

    #Build model for experiment
    if FLAGS.exp_flavor == 1:
        logger.info("Building base model")

        left_base_input = Input(shape = (FLAGS.max_seq_length, ), dtype = "int32") 
        right_base_input = Input(shape = (FLAGS.max_seq_length, ), dtype = "int32")
        embedding_layer = Embedding(len(embedding_matrix), FLAGS.embedding_dim,  weights = [embedding_matrix], 
                        input_shape = (FLAGS.max_seq_length, ), trainable = FLAGS.is_trainable)
        x0_left, x0_right = embedding_layer(left_base_input), embedding_layer(right_base_input)

        if FLAGS.lstm_attention == "lstm_attention":
            lstm_layer = LSTM(FLAGS.n_hidden, return_sequences=True)
            att_in_left, att_in_right = lstm_layer(x0_left), lstm_layer(x0_right)

            attention_layer = attention()
            x1_left, x1_right = attention_layer(att_in_left), attention_layer(att_in_right)
            #x1_left, x1_right = attention()(att_in_left), attention()(att_in_right) # This attention layer is not shared between the two inputs
        elif FLAGS.lstm_attention == "lstm":
            lstm_layer = LSTM(FLAGS.n_hidden)
            x1_left, x1_right = lstm_layer(x0_left), lstm_layer(x0_right)
        elif FLAGS.lstm_attention == "attention":
            attention_layer = attention()
            x1_left, x1_right = attention_layer(x0_left), attention_layer(x0_right)
            #x1_left, x1_right = attention()(x0_left), attention()(x0_right) # This attention layer is not shared between the two inputs

        shared_dense_1 = Dense(FLAGS.first_units_dense, FLAGS.dense_activation)
        x2_left, x2_right = shared_dense_1(x1_left), shared_dense_1(x1_right)

        shared_dense_2 = Dense(FLAGS.second_units_dense, FLAGS.dense_activation)
        
        if FLAGS.use_shared_dropout_layer:
            shared_dropout_1 = SharedDropout(FLAGS.shared_dropout_layer_first_rate)
            x3_left, x3_right = shared_dropout_1([x2_left, x2_right])
            
            x4_left, x4_right = shared_dense_2(x3_left), shared_dense_2(x3_right)
            
            shared_dropout_2 = SharedDropout(FLAGS.shared_dropout_layer_second_rate)
            x5_left, x5_right = shared_dropout_2([x4_left, x4_right])

            if FLAGS.DistanceScore == "Manhattan":
                last_layer_score = ManDist()([x5_left, x5_right])
            else:
                last_layer_score = CosineSim()([x5_left, x5_right])
                
            if FLAGS.PredictionLayer == "Softmax":
                softmax_layer = tf.keras.layers.Softmax()
                last_layer_score = softmax_layer(last_layer_score)

        else:
            x3_left, x3_right = shared_dense_2(x2_left), shared_dense_2(x2_right)

            if FLAGS.DistanceScore == "Manhattan":
                last_layer_score = ManDist()([x3_left, x3_right])
            else:
                last_layer_score = CosineSim()([x3_left, x3_right])

            if FLAGS.PredictionLayer == "Softmax":
                softmax_layer = tf.keras.layers.Softmax()
                last_layer_score =  softmax_layer(last_layer_score)

        
        model = Model(inputs = [left_base_input, right_base_input], outputs = [last_layer_score])

        metrics = [
            "acc",
            tf.keras.metrics.Precision(), 
            tf.keras.metrics.Recall(),
            #tfa.metrics.F1Score
        ]
        model.compile(loss = FLAGS.loss_function, 
                      optimizer = Adam(lr = FLAGS.learning_rate), 
                      metrics = metrics) # 

        #Check model
        model.summary()
        logger.info(model.summary())
    
    elif FLAGS.exp_flavor == 2:
        # With a context vector
        left_base_input = Input(shape = (FLAGS.max_seq_length, ), dtype = "int32")
        right_base_input = Input(shape = (FLAGS.max_seq_length, ), dtype = "int32")
        embedding_layer = Embedding(len(embedding_matrix), FLAGS.embedding_dim,  weights = [embedding_matrix], 
                                    input_shape = (FLAGS.max_seq_length, ), trainable = FLAGS.is_trainable)
        x0_left, x0_right = embedding_layer(left_base_input), embedding_layer(right_base_input)

        if FLAGS.lstm_attention == "lstm_attention":
            lstm_layer = LSTM(FLAGS.n_hidden, return_sequences=True)
            att_in_left, att_in_right = lstm_layer(x0_left), lstm_layer(x0_right)

            attention_layer = attention()
            x1_left, x1_right = attention_layer(att_in_left), attention_layer(att_in_right)
            #x1_left, x1_right = attention()(att_in_left), attention()(att_in_right) # This attention layer is not shared between the two inputs

        elif FLAGS.lstm_attention == "lstm":
            lstm_layer = LSTM(FLAGS.n_hidden)
            x1_left, x1_right = lstm_layer(x0_left), lstm_layer(x0_right)
        elif FLAGS.attention == "attention":
            attention_layer = attention()
            x1_left, x1_right = attention_layer(x0_left), attention_layer(x0_right)
            #x1_left, x1_right = attention()(x0_left), attention()(x0_right) # This attention layer is not shared between the two inputs

        left_context_input = Input(shape = (FLAGS.context_vector_dim), dtype = "float32")
        right_context_input = Input(shape = (FLAGS.context_vector_dim), dtype = "float32")
 
        # W/o con_dense [LSTM concat CONTEXT]
        #x1_left_con, x1_right_con = left_context_input, right_context_input
        
        con_dense = Dense(FLAGS.n_hidden, FLAGS.dense_activation)
        x1_left_con, x1_right_con = con_dense(left_context_input), con_dense(right_context_input)

        x2_left = Concatenate()([x1_left, x1_left_con])
        x2_right = Concatenate()([x1_right, x1_right_con])

        shared_dense_1 = Dense(FLAGS.first_units_dense, FLAGS.dense_activation)
        x3_left, x3_right = shared_dense_1(x2_left), shared_dense_1(x2_right)

        shared_dense_2 = Dense(FLAGS.second_units_dense, FLAGS.dense_activation)
        x4_left, x4_right = shared_dense_2(x3_left), shared_dense_2(x3_right)
        #malstm_distance = ManDist()([x4_left, x4_right])

        if FLAGS.DistanceScore == "Manhattan":
            last_layer_score = ManDist()([x4_left, x4_right])
        else:
            last_layer_score = CosineSim()([x4_left, x4_right])

        model = Model(inputs = [left_base_input, left_context_input, right_base_input, right_context_input], outputs = [last_layer_score])

        metrics = [
            "acc",
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            #tfa.metrics.F1Score
        ]
        model.compile(loss = FLAGS.loss_function,
                      optimizer = Adam(lr = FLAGS.learning_rate),
                      metrics = metrics) #

        #Check model
        model.summary()
        logger.info(model.summary())

    else:
        logger.info("Building enriched model")
        model_exp = Model_umls(embedding_matrix)
        shared_model_base = model_exp.build_base_model()
        x3_left = Concatenate()([shared_model_base(left_base_input), left_enrich_input])
        x3_right = Concatenate()([shared_model_base(right_base_input), right_enrich_input])
        x4 = model_exp.build_dense_layer()
        #malstm_distance = ManDist()([x4(x3_left), x4(x3_right)])

        if FLAGS.PredictiDistanceScoreonLayer == "Manhattan":
            last_layer_score = ManDist()([x4(x3_left), x4(x3_right)])
        else:
            last_layer_score = CosineSim()([x4(x3_left), x4(x3_right)])
        
        if FLAGS.PredictionLayer == "Softmax":
            softmax_layer = tf.keras.layers.Softmax()
            last_layer_score = softmax_layer(last_layer_score)

        model = model_exp.make_model(last_layer_score, left_base_input, right_base_input, left_enrich_input, right_enrich_input)

        #Check model
        model.summary()
        shared_model_base.summary()
        x4.summary()


    return model

def get_file(fp):
    lst = glob.glob(fp)
    if len(lst) > 0:
        return lst[0]
    else:
        return None
    
def generate_id_dataset(mrconso_master_fp, aui2id_pickle_fp, ds_dp, level):
    logger.info("Generating ID datasets ...")
    g = "/*DS*/*DS.RRF" if level == 2 else "/*DS.RRF"
    files = [f for f in glob.iglob(ds_dp + g, recursive=True)]
    if len(files) == 0:
        logger.info("No files to pickle")
        return
    aui2id = load_pickle(aui2id_pickle_fp)
    for filename in files: 
        pickle_fp = filename + ".PICKLE"
        if not os.path.isfile(pickle_fp):
            logger.info("Processing file %s"%filename)
            pairs = read_file_to_pairs(filename)
            partition = dict() 
            for pair in pairs:
                ID = (aui2id[pair["AUI1"]], aui2id[pair["AUI2"]])
                partition[ID] = (pair["jacc"],pair["Label"])
            logger.info("Writing %s to %s"%(filename, pickle_fp))
            dump_pickle(partition, pickle_fp)
    
    return
def read_file_to_pairs(pairs_fp):
    pairs = []
    logger.info("Loading file %s ..."%pairs_fp)                
    with open(pairs_fp, "r") as fi:
        reader = csv.DictReader(fi,fieldnames=["jacc", "AUI1", "AUI2", "Label"], delimiter=FLAGS.delimiter)
        with tqdm(total = count_lines(pairs_fp)) as pbar:
            for line in reader:
                pbar.update(1)
                pairs.append(line)
    
    return pairs

def write_important_info(msg):
    with open(os.path.join(FLAGS.workspace_dp, FLAGS.important_info_fn), "w") as fo:
        fo.write("%s: %s\n"%(FLAGS.application_name, msg))
    
def main(_):
    import sys 
    dirpaths = generate_dirpaths()
    # Get logger for file
    global logger
    log_filepath = os.path.join(dirpaths["logs_dp"], "%s.log"%dirpaths["output_dn"]) if FLAGS.logs_fp is None else FLAGS.logs_fp
    logger = get_logger(logging.DEBUG, FLAGS.application_name, log_filepath)
    sys.stdout = open(log_filepath + "_stdout", "w")
   
    config_proto = tf.compat.v1.ConfigProto(log_device_placement = False, device_count = {"GPU": 1})
    off = rewriter_config_pb2.RewriterConfig.OFF
    config_proto.graph_options.rewrite_options.arithmetic_optimization = off
    with tf.compat.v1.Session(config = config_proto) as sess2:
        tf.compat.v1.keras.backend.set_session(sess2)

        logger.info("Loading data from string dictionary and context")

        mrconso_master_fp = os.path.join(FLAGS.umls_dl_dp, FLAGS.mrconso_master_fn)
        aui2vec_pickle_fp = os.path.join(FLAGS.umls_dl_dp, FLAGS.WordEmbVariant + "_" + FLAGS.aui2vec_pickle_fn)
        aui2convec_pickle_fp = os.path.join(FLAGS.KGE_Home, FLAGS.Embeddings, FLAGS.Model+ "_" +  FLAGS.ConOptimizer + "_" + str(FLAGS.ConEpochs), FLAGS.ConVariant, FLAGS.ConVariant + "_" + FLAGS.AUI2CONVEC)
        tokenizer_pickle_fp = os.path.join(FLAGS.umls_dl_dp, FLAGS.tokenizer_pickle_fn)
        aui2id_pickle_fp = os.path.join(FLAGS.umls_dl_dp, FLAGS.aui2id_pickle_fn)
        id2aui_pickle_fp = os.path.join(FLAGS.umls_dl_dp, FLAGS.id2aui_pickle_fn)
        mrc_atoms_pickle_fp = os.path.join(FLAGS.umls_dl_dp, FLAGS.mrc_atoms_pickle_fn)
        embedding_fp = FLAGS.embedding_fp if FLAGS.embedding_fp is not None else os.path.join(FLAGS.workspace_dp, FLAGS.extra_dn, FLAGS.pre_trained_word2vec)
        tokenizer = None
        if FLAGS.do_prep:
            
            # Generate the AUI2VEC file
            if FLAGS.word_embedding == "BioWordVec":
                tokenizer = gen_tokenizer(mrconso_master_fp, tokenizer_pickle_fp)
                gen_aui2vec(tokenizer, aui2vec_pickle_fp, mrconso_master_fp)
            elif FLAGS.word_embedding == "BERT":
                embedding_dict = load_pickle(embedding_fp)
                vocab_file = FLAGS.vocab_file
                if FLAGS.gen_vocab_file == "gen_vocab":
                    vocab_file = embedding_fp + "_vocab"
                    with open(vocab_file, "w") as f:
                        for token in embedding_dict.keys():
                            f.write(token + "\n") 
                gen_bert_aui2vec(vocab_file, aui2vec_pickle_fp, mrconso_master_fp)
            #gen_aui2id(mrconso_master_fp, aui2id_pickle_fp, id2aui_pickle_fp)
            #gen_mrc_atoms(mrconso_master_fp, mrc_atoms_pickle_fp)
            #if FLAGS.ds_to_pickle_dp is not None:
            #    ds_to_pickle_dp = FLAGS.ds_to_pickle_dp  
            #   level = 1
            #else: 
            #    ds_to_pickle_dp = FLAGS.dataset_version_dp
            #    level = 2
            #generate_id_dataset(mrconso_master_fp, aui2id_pickle_fp, ds_to_pickle_dp, level)
            
        if FLAGS.word_embedding == "BioWordVec":       
            #Load token dictionary.
            logger.info("Loading tokenizer pickle ... ")
            tokenizer = load_pickle(tokenizer_pickle_fp)
            vocab_length = len(tokenizer.word_index) + 1
            #Generate embedding matrix.
            embedding_matrix = make_word_embeddings(embedding_fp, vocab_length, tokenizer)

        elif FLAGS.word_embedding == "BERT":
            vocab_file = FLAGS.vocab_file
            if FLAGS.gen_vocab_file == "gen_vocab":
                    vocab_file = embedding_fp + "_vocab"
            embedding_matrix = make_bert_embedding_matrix(embedding_fp, vocab_file)           
        logger.info("Loading aui2vec pickle ...")
        aui2vec = load_pickle(aui2vec_pickle_fp)
        logger.info("Loading aui2convec pickle ...")
        if FLAGS.exp_flavor == 2:
            aui2convec = load_pickle(aui2convec_pickle_fp)
        else:
            aui2convec = None
        logger.info("Loading aui2id pickle ...")
        aui2id = load_pickle(aui2id_pickle_fp)
        logger.info("Loading id2aui pickle ...")
        id2aui = load_pickle(id2aui_pickle_fp)
        logger.info("Loading mrc atoms...")
        mrc_atoms = load_pickle(mrc_atoms_pickle_fp)
        
        model = create_model(embedding_matrix)
        
        if FLAGS.do_train:

            val_log_fp = os.path.join(dirpaths["train_result_dp"], 
                                            "%s_%s"%(FLAGS.val_test_results_log_fn, dirpaths["output_dn"]))

            train_fp = glob.glob(os.path.join(FLAGS.train_dataset_dp, "*%s"%FLAGS.train_fn))[0]
            logger.info("Loading training data from %s ..."%train_fp)
            train_partition = read_file_to_datagenerator(train_fp, aui2vec, aui2id)
            train_generator = DataGenerator(train_partition, aui2vec, aui2convec, FLAGS.max_seq_length, 
                                               FLAGS.embedding_dim, FLAGS.word_embedding, FLAGS.context_vector_dim, FLAGS.exp_flavor,
                                               batch_size = FLAGS.batch_size, 
                                               shuffle = True, is_test = False)
            
            val_fp = glob.glob(os.path.join(FLAGS.train_dataset_dp, "*%s"%FLAGS.val_fn))[0]
            logger.info("Loading validation data from %s ..."%val_fp)
            validation_partition = read_file_to_datagenerator(val_fp, aui2vec, aui2id)
            validation_generator = DataGenerator(validation_partition, aui2vec, aui2convec, FLAGS.max_seq_length, 
                                                    FLAGS.embedding_dim, FLAGS.word_embedding, FLAGS.context_vector_dim, FLAGS.exp_flavor, 
                                                    batch_size = FLAGS.batch_size, 
                                                    shuffle = False, is_test = False)

            test_fp = glob.glob(os.path.join(FLAGS.train_dataset_dp, "*%s"%FLAGS.test_fn))
            test_partition = dict()
            test_generator = dict()
            if (FLAGS.predict_test_dir_after_every_epoch is True) and ((FLAGS.test_dataset_dp is not None) or (FLAGS.test_dataset_fp is not None)):
                if FLAGS.test_dataset_dp is not None:
                    test_fp +=  glob.glob(os.path.join(FLAGS.test_dataset_dp, "*%s"%(FLAGS.test_fn)))
                if FLAGS.test_dataset_fp is not None:
                    test_fp += FLAGS.test_dataset_fp
                                                         
                for fp in test_fp:
                    logger.info("Loading testing data from %s ..."%fp)
                    
                    if FLAGS.load_IDs:
                        test_partition[fp] = read_file_to_datagenerator(fp, aui2vec, aui2id)
                        test_generator[fp] = DataGenerator(test_partition[fp], aui2vec, aui2convec, FLAGS.max_seq_length, 
                                                              FLAGS.embedding_dim, FLAGS.word_embedding, FLAGS.context_vector_dim, FLAGS.exp_flavor,
                                                              batch_size = FLAGS.batch_size, 
                                                              shuffle = False, is_test = True)
                    else:
                        test_partition[fp] = read_file_to_filegenerator(fp, aui2vec, aui2id)
                        test_generator[fp] = FileGenerator(fp, aui2vec, aui2convec, aui2id, FLAGS.max_seq_length, 
                                                              FLAGS.embedding_dim, FLAGS.word_embedding, FLAGS.context_vector_dim, FLAGS.exp_flavor,
                                                              batch_size = FLAGS.batch_size, 
                                                              shuffle = False, is_test = True)
                        
            metrics_per_epoch = MetricsPerEpochCallback(test_generator, test_partition, mrc_atoms, dirpaths, FLAGS.load_IDs)
            early_stop = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=10)
            csv_logger = CSVLogger(val_log_fp, append = True, separator = FLAGS.delimiter)

            if FLAGS.continue_training:
                checkpoint_fp = os.path.join(dirpaths["checkpoint_dp"], "weights.%d.hdf5"% FLAGS.checkpoint_epoch)
                logger.info("Predicting file at epoch %d using checkpoint %s"%(FLAGS.checkpoint_epoch, checkpoint_fp))
                if os.path.isfile(checkpoint_fp):
                    model.load_weights(checkpoint_fp)
                else:
                    logger.info("Not a valid fp!")
                    import sys
                    sys.exit(1)

            # save weights
            checkpoint_fp = os.path.join(dirpaths["checkpoint_dp"], "weights.{epoch:d}.hdf5")
            save_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(checkpoint_fp, verbose = 1, save_weights_only = True)
            
            train_size_dataset = len(train_partition)
            val_size_dataset = len(validation_partition)

            training_start_time = time.time()
            logger.info("Start training ...")

            trained_history = model.fit(
                x = train_generator, 
                validation_data = validation_generator, 
                epochs = FLAGS.n_epoch, 
#                 batch_size = FLAGS.batch_size, shuffle = True, 
                steps_per_epoch = math.ceil(train_size_dataset/FLAGS.batch_size), 
                validation_steps = math.ceil(val_size_dataset/FLAGS.batch_size), 
                callbacks = [metrics_per_epoch, 
                             early_stop,
                             csv_logger, 
                             save_checkpoint_cb], 
                verbose = FLAGS.train_verbose, 
                use_multiprocessing = True, 
                workers = FLAGS.generator_workers, 
                initial_epoch = FLAGS.checkpoint_epoch
            )  

            test_metrics = metrics_per_epoch.get_metrics()
            del metrics_per_epoch
            
            training_end_time = time.time()
            logger.info("Training time finished.\n%d epochs in %12.2f" % (FLAGS.n_epoch, training_end_time - training_start_time))
            #Get result.

            model_path = os.path.join(dirpaths["checkpoint_dp"], "training_model_{}.h5")
            logger.info("Saving trained model weights to %s"%model_path)
            model.save_weights(model_path)

            save_history(dirpaths["train_result_dp"], FLAGS.final_history_fn, trained_history)
#             save_history_graph(dirpaths["train_result_dp"], FLAGS.final_history_graph_fn, trained_history)

            save_metrics(dirpaths["train_result_dp"], FLAGS.final_metrics_fn, test_metrics)
#             save_metrics_graph(dirpaths["train_result_dp"], FLAGS.final_metrics_graph_fn, test_metrics)
            
            del train_partition
            del validation_partition
            del test_partition
            del train_generator
            del validation_generator
            del test_generator

        if FLAGS.do_predict_all:

            tf.compat.v1.keras.backend.set_learning_phase(0)
            epoch_predict_all = FLAGS.epoch_predict_all
            
            test_fp = "PREDICT_ALL_USING_ALL"
            
            "Initialize all dicts"
            test_generator = None

            all_scores = dict()
            all_scores[test_fp] = dict()
            for metric in FLAGS.metric_names:
                all_scores[test_fp][metric] = list()

            logger.info("Generating aui2cui")
            aui2cui = dict()
            for aui, aui_info in mrc_atoms.items():
                aui2cui[int(aui_info["ID"])] = int(aui_info["CUI"][1:])
            test_dict(aui2cui)
               
            #Load model.
            checkpoint_fp = os.path.join(dirpaths["checkpoint_dp"], "weights.%d.hdf5"%epoch_predict_all)
            logger.info("Predicting file at epoch %d using checkpoint %s"%(epoch_predict_all, checkpoint_fp))
            if os.path.isfile(checkpoint_fp):
                model.load_weights(checkpoint_fp)
                start_time = time.time()
                test_generator = PairGenerator(FLAGS.start_aui1, FLAGS.end_aui1, aui2vec, aui2cui, FLAGS.max_seq_length, 
                                               FLAGS.embedding_dim, batch_size = FLAGS.predict_batch_size, 
                                               shuffle = False, is_test = True)
                # Get labels for the generator
                batch_idx = test_generator.compute_batch_idx() 
                test_dict(batch_idx)
                num_neg_pairs = 0
                for aui1, aui2, batch_max, batch_len in batch_idx.values():
                    num_neg_pairs += batch_len
                logger.info("Num neg pairs: {}".format(num_neg_pairs))

                TN, FP = predict_all_negatives(test_generator, model, num_neg_pairs, FLAGS.predict_batch_size)

                logger.info("Results for AUI1 range({}, {}): TN={} or {}, FP={} or {} in total {} sec".format(FLAGS.start_aui1, FLAGS.end_aui1, TN, round(TN/(TN+FP),4), FP, round(FP/(TN+FP)), time.time()-start_time))
                  
                # Save the scores to file
#                logger.info("Saving metrics per epoch to %s"%dirpaths["predict_result_dp"])
#                save_metrics(dirpaths["predict_result_dp"], "testing_allneg_epoch_{}_{}_{}".format(epoch_predict, FLAGS.final_metrics_fn), all_scores)
#                 save_metrics_graph(dirpaths["predict_result_dp"], "testing_epoch_{}_{}_{}".format(start_epoch_predict, FLAGS.end_epoch_predict, FLAGS.final_metrics_graph_fn), all_scores)
                del test_generator     

        if FLAGS.do_predict:

            tf.compat.v1.keras.backend.set_learning_phase(0)
            start_epoch_predict = FLAGS.n_epoch-2 if FLAGS.start_epoch_predict is None else FLAGS.start_epoch_predict
            end_epoch_predict = FLAGS.n_epoch if FLAGS.start_epoch_predict is None else FLAGS.end_epoch_predict
            
            if (FLAGS.test_dataset_dp is not None) or (FLAGS.test_dataset_fp is not None):
                test_filepath = []
                if (FLAGS.test_dataset_dp is not None):
                    test_filepath += glob.glob(os.path.join(FLAGS.test_dataset_dp, "*%s"%(FLAGS.test_fn)))
                if (FLAGS.test_dataset_fp is not None):
                    test_filepath += glob.glob(FLAGS.test_dataset_fp)
            
                "Initialize all dicts"
                all_scores = dict()
                test_partition = None
                test_generator = None

                for fp in test_filepath:
                    all_scores[fp] = dict()
                    for metric in FLAGS.metric_names:
                        all_scores[fp][metric] = list()

                for cur_epoch in range(start_epoch_predict, end_epoch_predict + 1):
                    #Load model.
                    checkpoint_fp = os.path.join(dirpaths["checkpoint_dp"], "weights.%d.hdf5"%cur_epoch)
                    # if os.path.isfile(checkpoint_fp):
                    #     time.sleep(600)                
                    #     model.load_weights(checkpoint_fp)
                    # else:
                    logger.info("Looking for checkpoint at %s"%checkpoint_fp)
                    while (not os.path.isfile(checkpoint_fp)):
                        logger.info("Waiting for checkpoint at %s"%checkpoint_fp)
                        time.sleep(10)
                    time.sleep(10)
                    model.load_weights(checkpoint_fp)

                    #Predict label
                    for fp in test_filepath:
                        logger.info("Loading testing data from %s ..."%fp)

                        if FLAGS.load_IDs:
                            test_partition = read_file_to_datagenerator(fp, aui2vec, aui2id)
                            test_generator = DataGenerator(test_partition, aui2vec, aui2convec, FLAGS.max_seq_length, 
                                                                  FLAGS.embedding_dim, FLAGS.word_embedding, FLAGS.context_vector_dim, FLAGS.exp_flavor,
                                                                  batch_size = FLAGS.batch_size, 
                                                                  shuffle = False, is_test = True)
                            logger.info("Predicting file at epoch %d using checkpoint %s"%(cur_epoch, checkpoint_fp))
                            scores, predictions = predict_generator(test_generator, test_partition.values(), len(test_partition), model, 
                                                                   FLAGS.batch_size, epoch = cur_epoch, log_scores = all_scores[fp])

                        else:
                            test_partition = read_file_to_filegenerator(fp, aui2vec, aui2id)
                            test_generator = FileGenerator(fp, aui2vec, aui2convec, aui2id, FLAGS.max_seq_length, 
                                                                  FLAGS.embedding_dim, FLAGS.word_embedding, FLAGS.context_vector_dim, FLAGS.exp_flavor,
                                                                  batch_size = FLAGS.batch_size, 
                                                                  shuffle = False, is_test = True)
                            logger.info("Predicting file at epoch %d using checkpoint %s"%(cur_epoch, checkpoint_fp))
                            scores, predictions = predict_generator(test_generator, test_partition, len(test_partition), model, 
                                                                   FLAGS.batch_size, epoch = cur_epoch, log_scores = all_scores[fp])


                        logger.info(scores)

                        if cur_epoch == FLAGS.n_epoch:
                            # Should be the epoch with the best f1

                            predictions_fp = os.path.join(dirpaths["predict_result_dp"], "_".join([get_base(fp), str(cur_epoch), FLAGS.predictions_fn]))
                            save_predicted_results(predictions, predictions_fp, FLAGS.n_epoch)
                            #compare_predictions(test_partition, predictions, mrc_atoms, fp, aui2vec, aui2id, id2aui, predictions_fp, cur_epoch, FLAGS.test_dataset_fp_rba_predictions)

                        del predictions
                        del test_partition
                        del test_generator
                    save_metrics(dirpaths["predict_result_dp"], FLAGS.tmp_metrics_fn, all_scores)
#                     save_metrics_graph(dirpaths["predict_result_dp"], FLAGS.tmp_metrics_graph_fn,  all_scores)
                # Save the scores to file
                logger.info("Saving metrics per epoch to %s"%dirpaths["predict_result_dp"])
                save_metrics(dirpaths["predict_result_dp"], "testing_epoch_{}_{}_{}".format(start_epoch_predict, FLAGS.end_epoch_predict, FLAGS.final_metrics_fn), all_scores)
#                 save_metrics_graph(dirpaths["predict_result_dp"], "testing_epoch_{}_{}_{}".format(start_epoch_predict, FLAGS.end_epoch_predict, FLAGS.final_metrics_graph_fn), all_scores)
        if FLAGS.do_analyze:
            if (FLAGS.test_dataset_dp is not None) or (FLAGS.test_dataset_fp is not None):
                test_filepath = []
                if (FLAGS.test_dataset_dp is not None):
                    test_filepath += glob.glob(os.path.join(FLAGS.test_dataset_dp, "*%s"%(FLAGS.test_fn)))
                if (FLAGS.test_dataset_fp is not None):
                    test_filepath += glob.glob(FLAGS.test_dataset_fp)
                for test_fp in test_filepath:
                    predictions_fp = os.path.join(dirpaths["predict_result_dp"], "_".join([get_base(test_fp), str(FLAGS.n_epoch), FLAGS.predictions_fn]))
                    do_analyze(mrc_atoms, test_fp, aui2vec, aui2id, id2aui, predictions_fp, FLAGS.n_epoch)     
        if FLAGS.do_report:
            # Collecting the results from the runs and generating report tables and figures
            # Dataset size table for learning and generalization test
            print("Todo")
            # Performance for samedist and gentest in all flavors per metric (acc, pre, recall, f1)
        del model
        del aui2vec
        del aui2id
        del id2aui
        del mrc_atoms
        del tokenizer
        gc.collect()
    tf.compat.v1.keras.backend.clear_session()
    # remember to close the handlers
    for handler in logger.handlers:
        handler.close()
        logger.removeFilter(handler)
    
    import sys
    logger.info("Done.")
    
    sys.stdout.close()
if __name__ == "__main__":
    
    app.run(main)

