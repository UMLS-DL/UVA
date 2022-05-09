import csv
import random
import subprocess
import math
import shutil
import glob
import pickle
import os
import bisect
from pathlib import Path
import logging
from tqdm import tqdm
import time
import inspect
import itertools
import queue
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from multiprocessing import Process, Queue, Manager
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, roc_curve, auc
import absl
from absl import app
from absl import flags

from common import SlurmJob, NodeParallel, Utils

FLAGS = flags.FLAGS
flags.DEFINE_string('f', '', 'kernel')
flags.DEFINE_string('server','FM','')
flags.DEFINE_string('application_name', 'run_rba_evaluator', '')
flags.DEFINE_string('application_py_fn', 'run_rba_evaluator.py', '')
flags.DEFINE_string('important_info_fn', 'IMPORTANT_INFO', '')

flags.DEFINE_string('workspace_dp','/data/nguyenvt2/aaai2020','')
flags.DEFINE_string('umls_dp','/data/nguyenvt2/aaai2020/UMLS_VERSIONS','')
flags.DEFINE_string('umls_version_dn', '2020AA-ACTIVE', '')
flags.DEFINE_string('umls_version_dp', None, '')
flags.DEFINE_string('dataset_version_dn', None, '')
flags.DEFINE_string('dataset_version_dp', None,'')
flags.DEFINE_string('umls_dl_dp','/data/nguyenvt2/aaai2020/UMLS_VERSIONS/2020AA-ACTIVE/META_DL','')
flags.DEFINE_string('umls_dl_dn','META_DL', '')
flags.DEFINE_string('umls_meta_dp', None, '')
flags.DEFINE_string('umls_meta_dn','META','')
flags.DEFINE_string('datasets_dp', None, '')
flags.DEFINE_string('datasets_dn', "datasets", '')
#flags.DEFINE_string("test_dataset_dp", '/data/nguyenvt2/aaai2020/UMLS_VERSIONS/2020AA-ACTIVE/GENTEST_DS', "The input dataset dir inside. Should contain the .RRF files (or other data files) "
#    "for the task.")
flags.DEFINE_string("test_dataset_dn", None, "The input dataset dir inside. Should contain the .RRF files (or other data files) "
    "for the task.")
flags.DEFINE_string('test_dataset_dp', None, '')
flags.DEFINE_string('test_dataset_fp', None, '')
flags.DEFINE_string('flavor', 'ALL', '')
flags.DEFINE_string('rba_dp', None, '')
flags.DEFINE_string('rba_bin_dp', None, '')
flags.DEFINE_string('rba_log_dp', None, '')
flags.DEFINE_string('rba_dn','RBA','')

flags.DEFINE_string('log_dn','logs','')
flags.DEFINE_string('bin_dn','bin','')
flags.DEFINE_string('extra_dn','extra','')

flags.DEFINE_string('mrconso_master_fn','MRCONSO_MASTER.RRF','')
flags.DEFINE_string('mrconso_master_rba_pickle_fn','MRCONSO_MASTER_RBA.PICKLE','')
flags.DEFINE_list('mrconso_master_fields', ["ID", "CUI", "LUI", "SUI", "AUI", "AUI_NUM", "SCUI", "NS_ID", "NS_LEN", "NS", "NORM_STR","STR", "SG"], '')
flags.DEFINE_string("aui2id_pickle_fn", 'AUI2ID.PICKLE', "The output directory where the model checkpoints will be written.")
flags.DEFINE_string("id2aui_pickle_fn", 'ID2AUI.PICKLE', "The output directory where the model checkpoints will be written.")
flags.DEFINE_string('mrconso_fn','MRCONSO.RRF','')
flags.DEFINE_string("test_fn", 'TEST_DS.RRF', "")

flags.DEFINE_string("trans_closure_fn", 'TRANS_CLOSURE_DERIVED_PAIRS.RRF', "")
flags.DEFINE_string("trans_closure_pickle_fn", 'TRANS_CLOSURE.PICKLE', "")
flags.DEFINE_string("trans_closure_info_pickle_fn", 'TRANS_CLOSURE_INFO.PICKLE', "")
flags.DEFINE_string("trans_closure_path_pickle_fn", 'TRANS_CLOSURE_PATH.PICKLE', "")
flags.DEFINE_string("merged_clusters_batch_dn", 'MERGED_CLUSTERS_BATCH', "")
flags.DEFINE_string("merged_clusters_pickle_fn", 'MERGED_CLUSTERS.PICKLE', "")
flags.DEFINE_string("all_clusters_dict_pickle_fn", 'ALL_CLUSTERS_DICT.PICKLE', "")
flags.DEFINE_string("excluded_clusters_pickle_fn", 'EXCLUDED_CLUSTERS.PICKLE', "")
flags.DEFINE_string("completed_inputs_fn", 'COMPLETED_INPUTS.RRF', "")

flags.DEFINE_string("all_clusters_dict_pickle_fp", None, '')
flags.DEFINE_string("merged_clusters_pickle_fp", None, '')
flags.DEFINE_string("merged_clusters_batch_pickle_fp", None, '')
flags.DEFINE_string("excluded_clusters_batch_pickle_fp", None, '')
flags.DEFINE_string("completed_inputs_fp", None, '')

flags.DEFINE_string('delimiter', '|', '')
flags.DEFINE_bool('do_eval', True, '')
flags.DEFINE_bool('do_prep', True, '')
flags.DEFINE_bool('do_compute_rule_closure_in_batch', False, '')
flags.DEFINE_bool('regenerate_files', False, '')
flags.DEFINE_string('closure_approach', 'cluster', '')

flags.DEFINE_integer('interval_time_check_for_job', 120, '')

# For multiprocessing
flags.DEFINE_bool('debug', True, '')
flags.DEFINE_bool('run_slurm_job', False, '')
flags.DEFINE_integer('closure_n_processes', 20, '')
flags.DEFINE_integer('ntasks', 499, '')
flags.DEFINE_integer('start_idx', 0, '')
flags.DEFINE_integer('end_idx', 100, '')
flags.DEFINE_string('conda_env', 'tf22', '')
flags.DEFINE_string('job_name', 'MERGE_CLTR', '')
flags.DEFINE_integer('start_loop_cnt', 0, '')
flags.DEFINE_integer('ram', '6', '')

def compute_rba_baseline(flavor, paths, test_fp, out_fp):
    id_2_cluster_id = utils.load_pickle(paths['trans_closure_pickle_fp'])
    id_2_cluster_info = utils.load_pickle(paths['trans_closure_info_pickle_fp'])
    id_2_cluster_path = utils.load_pickle(paths['trans_closure_path_pickle_fp'])
    mrc_atoms = utils.load_pickle(paths['mrconso_master_rba_pickle_fp'])
    aui2id = utils.load_pickle(paths['aui2id_pickle_fp'])
    id2aui_dict = utils.load_pickle(paths['id2aui_pickle_fp'])
    all_clusters_dict = utils.load_pickle(paths['all_clusters_dict_pickle_fp'])
    
    inputs = utils.read_file_to_id_ds(test_fp, aui2id)

    FP_file = open(out_fp['FP'] + '_' + flavor, 'w')
    FN_file = open(out_fp['FN'] + '_' + flavor, 'w') 
    TP_file = open(out_fp['TP'] + '_' + flavor, 'w')
    TN_file = open(out_fp['TN'] + '_' + flavor, 'w')
    pred_file = out_fp['pred'] + '_' + flavor
    pos, neg, TP, TN, FP, FN = 0, 0, 0, 0, 0, 0
    #f = open(test_fp, 'r') 
    #inputs = f.readlines()
    #f.close()
    val_pred = np.empty((len(inputs)), dtype=int)
    val_label = np.empty((len(inputs)), dtype=int)
    with tqdm(total = len(inputs)) as pbar:
        #for idx, line in enumerate(inputs):
        for idx, (ID, info) in enumerate(inputs.items()):
            #fields = line.split('|')
            #ID = (aui2id[fields[1]], aui2id[fields[2]])
            #info = (fields[0], fields[3])           
            pbar.update(1)
            
            pos = pos + 1 if info[1] == 1 else pos
            neg = neg + 1 if info[1] == 0 else neg
            val_label[idx] = info[1]

            atom1 = mrc_atoms[ID[0]]
            atom2 = mrc_atoms[ID[1]]
            common_sg = atom1['SG'].intersection(atom2['SG'])

            if flavor == "LS_SG":

                if (atom1['LUI'] == atom2['LUI']) and (len(common_sg) > 0):
                    val_pred[idx] = 1
                else:
                    val_pred[idx] = 0

            elif flavor == "SCUI":

                if (atom1['SCUI'] != '') and (atom1['SCUI'] == atom2['SCUI']):
                    val_pred[idx] = 1
                else:
                    val_pred[idx] = 0
      
            elif flavor == "SCUI_LS_SG":
 
                if (atom1['SCUI'] != '') and (atom1['SCUI'] == atom2['SCUI']):
                    val_pred[idx] = 1

                elif (atom1['LUI'] == atom2['LUI']) and (len(common_sg) > 0):
                    val_pred[idx] = 1
                else:
                    val_pred[idx] = 0

            elif flavor == "SCUI_LS_SG_TRANS":
           
                if (ID[0] in id_2_cluster_id) and (ID[1] in id_2_cluster_id) and (id_2_cluster_id[ID[0]] == id_2_cluster_id[ID[1]]):
                    val_pred[idx] = 1
                else:
                    val_pred[idx] = 0
        
            if (val_pred[idx] == 1) and (val_label[idx] == 0):
                # False positive  
                FP += 1
                print_verification(FP_file, info, atom1, atom2, id_2_cluster_id, id_2_cluster_info, id_2_cluster_path, all_clusters_dict, id2aui_dict)

            elif (val_pred[idx] == 0) and (val_label[idx] == 1):
                FN += 1
                # False negative
                print_verification(FN_file, info, atom1, atom2, id_2_cluster_id, id_2_cluster_info, id_2_cluster_path, all_clusters_dict, id2aui_dict)

            elif (val_pred[idx] == 1) and (val_label[idx] == 1):
                TP += 1
                #print_verification(TP_file, info, atom1, atom2, id_2_cluster_id, id_2_cluster_info, id_2_cluster_path, all_clusters_dict, id2aui_dict)
            else:
                TN += 1
                #print_verification(TN_file, info, atom1, atom2, id_2_cluster_id, id_2_cluster_info, id_2_cluster_path, all_clusters_dict, id2aui_dict)

    FP_file.close()
    FN_file.close()   
    TP_file.close()
    TN_file.close()
    utils.dump_pickle(val_pred, pred_file)
    accuracy, precision, recall, f1 = utils.cal_scores(TP, TN, FP, FN)

    scores = {'epoch': 0, 'accuracy':accuracy, 'precision':precision, 'recall':recall, 'f1':f1}
    scores_stat = {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN, 'POS': pos, 'NEG': neg}
   
    with open(out_fp['stat'] + '_' + flavor, 'a') as fo:
        fo.write("\n=== Data Distribution & Model Statistics by compute_rba_baseline ===")
        fo.write("\nFile: %s"%test_fp)
        fo.write("\nNon Synonym [0]:\t{}\n".format(neg))
        fo.write("Is Synonym  [1]:\t{}\n".format(pos))

        fo.write("Accuracy:\t\t{:.4f}\n".format(accuracy))
        fo.write("Precision:\t\t{:.4f}\n".format(precision))
        fo.write("Recall:  \t\t{:.4f}\n".format(recall))
        fo.write("F1-Score:\t\t{:.4f}\n".format(f1))
        fo.write("TP:\t\t{:.4f}\n".format(TP))
        fo.write("FP:\t\t{:.4f}\n".format(FP))
        fo.write("TN:\t\t{:.4f}\n".format(TN))
        fo.write("FN:\t\t{:.4f}\n".format(FN))

    return scores

def compute_lexical_baseline(id_2_cluster_id, id_2_cluster_info, aui2id, mrc_atoms,filein, fileouts, flavor):
    # Input: AUI, SCUI, LUI, SG
    # Output: preds = array of predictions
    val_pred = []
    #         labels = array of labels 
    val_true = []
    nonsym_count = 0
    sym_count = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    # Step 1: Read pairs from ds_file
    # Step 2: For every aui pair, retrieve the info from mrc_atoms
    max_size = utils.count_lines(filein)
    with tqdm(total = utils.count_lines(filein)) as pbar:
        with open(filein) as f:
            with open(fileouts['false_positives'] + '_' + flavor, 'w') as false_positive_file:
                with open(fileouts['false_negatives'] + '_' + flavor, 'w') as false_negative_file:
                    pbar.update(1)
                    i = 0
                    none = 0
                    reader = csv.DictReader(f,fieldnames=["Type","AUI1","AUI2","Label"],delimiter=FLAGS.delimiter)
                    for line in reader:
                        atom1 = mrc_atoms.get(aui2id[line["AUI1"]])
                        atom2 = mrc_atoms.get(aui2id[line["AUI2"]])

                        if (atom1 and atom2):
                            common_sg = atom1['SG'].intersection(atom2['SG'])
                            # CUI","LUI",SCUI","SG"
                            if flavor == "LS_SG": # LS_SG
                                if ((atom1['LUI'] == atom2['LUI']) and (len(common_sg) > 0)):
                                    # Same lui, same sg
                                    pred = 1
                                else:
                                    pred = 0
                            elif flavor == "SCUI": # SCUI
                                if (atom1['SCUI'] == atom2['SCUI']) and (atom1['SCUI'] != ''):
                                    # Same scui
                                        pred = 1
                                else:
                                    pred = 0
                            elif flavor == "SCUI_LS_SG": #SCUI + LS_SG
                                if (atom1['SCUI'] == atom2['SCUI']) and (atom1['SCUI'] != ''):
                                    # Same scui
                                        pred = 1
                                elif ((atom1['LUI'] == atom2['LUI']) and (len(common_sg) > 0)):
                                    # Same lui, same sg
                                    pred = 1
                                else:
                                    pred = 0
                            
                            elif flavor == "SCUI_LS_SG_TRANS": # SCUI + LS_SG + TRANS
                                id1 = mrc_atoms[aui2id[line['AUI1']]]
                                id2 = mrc_atoms[aui2id[line['AUI2']]]
                                if (id1 in id_2_cluster_id) and (id2 in id_2_cluster_id) and (id_2_cluster_id[id1] == id_2_cluster_id[id2]):
                                    # Synonymous
                                    pred = 1
                                else:
                                    pred = 0

                            if ((line["Label"] == '0') or (line["Label"] == '1')):
                                val_true.append(int(str(line["Label"])))
                                val_pred.append(int(pred))
                                v_true = int(str(line["Label"]))
                                v_pred = int(pred)
                                if v_true == 1 :
                                    if v_pred == 1:
                                        TP += 1
                                    else:
                                        FN += 1
                                        print_pair_info(false_negative_file, line, atom1, atom2, id_2_cluster_id, id_2_cluster_info)
                                else:
                                    if v_pred == 1:
                                        FP += 1
                                        print_pair_info(false_positive_file, line, atom1, atom2, id_2_cluster_id, id_2_cluster_info)
                                    else:
                                        TN += 1

                                if line["Label"] == '0':
                                    nonsym_count += 1
                                else:
                                    sym_count += 1
                            else:
                                logger.info(line["Label"])
                        else:
                            #logger.info(line["AUI1"] + ": " + str(atom1))
                            #logger.info(line["AUI2"] + ": " + str(atom2))
                            none += 1

                        pbar.update(1)

    print ("Not existing pairs: %d"%none)
    print (TP)
    print (FP)
    print (TN)
    print (FN)
    
    _val_accuracy = (TP + TN)/(TP + TN + FP + FN)
    _val_precision = 0
    _val_recall = 0
    _val_f1 = 0
    if TP + FP > 0:
        _val_precision = TP/(TP+FP)
        
    if TP + FN > 0:
        _val_recall = TP/(TP+FN)
        
    if _val_recall + _val_precision > 0:
        _val_f1 = 2*(_val_recall * _val_precision) / (_val_recall + _val_precision)

        
    with open(fileouts['stat'] + '_' + flavor, 'a') as fo:
        fo.write("\n=== Data Distribution & Model Statistics ===")
        fo.write("\nFile: %s"%filein)
        fo.write("\nNon Synonym [0]:\t{}\n".format(nonsym_count))
        fo.write("Is Synonym  [1]:\t{}\n".format(sym_count))

        fo.write("Accuracy:\t\t{:.4f}\n".format(_val_accuracy))
        fo.write("Precision:\t\t{:.4f}\n".format(_val_precision))
        fo.write("Recall:  \t\t{:.4f}\n".format(_val_recall))
        fo.write("F1-Score:\t\t{:.4f}\n".format(_val_f1))
        fo.write("TP:\t\t{:.4f}\n".format(TP))
        fo.write("FP:\t\t{:.4f}\n".format(FP))
        fo.write("TN:\t\t{:.4f}\n".format(TN))
        fo.write("FN:\t\t{:.4f}\n".format(FN))

    #process_eval(val_pred, val_true, fileout+"_stat_v2")

def get_sg_lui_id(atom):
    ID = ''
    if atom['SG'] is not None:
        if atom['SG'][0] < atom['LUI']:
            ID = '(' + sg + '+' + line["LUI"] + ')'
        else:
            ID = '(' + line["LUI"] + '+' + sg + ')'
    return ID
 
def get_auis(all_clusters_dict, id2aui_dict,ID):
    aui_ids = list(all_clusters_dict[ID])
    auis = [id2aui_dict[int(aui_id)] for aui_id in aui_ids]
    auis.sort()
    return auis
    
def print_verification(fo, info, atom1, atom2, id_2_cluster_id, id_2_cluster_info, id_2_cluster_path, all_clusters_dict, id2aui_dict):

    fo.write("\n%s|%s|%s|%s\n"%(info[0], atom1['AUI'], atom2['AUI'], info[1]))
    fo.write("%s|%s|%s|%s|%s|%s\n"%(atom1['AUI'], atom1['CUI'], atom1['LUI'], atom1['SCUI'], ','.join(list(atom1['SG'])), atom1['STR']))
    fo.write("%s|%s|%s|%s|%s|%s\n\n"%(atom2['AUI'], atom2['CUI'], atom2['LUI'], atom2['SCUI'], ','.join(list(atom2['SG'])), atom2['STR']))
    
    #start_cluster_id = atom1['SCUI'] if atom1['SCUI'] != '' else get_sg_lui_id(atom1)
    #end_cluster_id = atom2['SCUI'] if atom2['SCUI'] != '' else get_sg_lui_id(atom2)
    #path = find_path(all_clusters_ids, id_2_cluster_info[atom1['AUI']].keys(), id2aui_dict, start_cluster_id = None, end_cluster_id = None)

    #fo.write("%s\n"%(path['path']))
    fo.write("%s\n"%(id_2_cluster_path[id_2_cluster_id[atom1['ID']]][0]))
    fo.write("%s\n"%(id_2_cluster_path[id_2_cluster_id[atom2['ID']]][0]))

    for cluster_id in id_2_cluster_path[id_2_cluster_id[atom1['ID']]][1]:
        fo.write('%s|%s|%s\n'%(atom1['AUI'], cluster_id, ','.join(get_auis(all_clusters_dict, id2aui_dict, cluster_id))))

    for cluster_id in id_2_cluster_path[id_2_cluster_id[atom2['ID']]][1]:
        fo.write('%s|%s|%s\n'%(atom2['AUI'], cluster_id, ','.join(get_auis(all_clusters_dict, id2aui_dict, cluster_id))))

    return

        
def print_pair_info(fo, line, atom1, atom2, id_2_cluster_info = None):
    
    fo.write("%s|%s|%s|%s\n"%(line['Type'], line['AUI1'], line['AUI2'], line['Label']))
    fo.write("%s|%s|%s|%s|%s|%s\n"%(line['AUI1'], atom1['CUI'], atom1['LUI'], atom1['SCUI'], ','.join(list(atom1['SG'])), atom1['STR'], id_2_cluster_info[atom1['ID']]))
    fo.write("%s|%s|%s|%s|%s|%s\n"%(line['AUI2'], atom2['CUI'], atom2['LUI'], atom2['SCUI'], ','.join(list(atom2['SG'])), atom2['STR'], id_2_cluster_info[atom2['ID']]))
    return
    
def process_eval(val_pred, val_true, stat_file):
    nonsym_count = len([val for val in val_true if val == 0])
    sym_count = len([val for val in val_true if val == 1])
    _val_accuracy = accuracy_score(val_true, val_pred)
    _val_precision = precision_score(val_true, val_pred)
    _val_recall = recall_score(val_true, val_pred)
    _val_f1 = f1_score(val_true, val_pred)
    _val_mcc = matthews_corrcoef(val_true, val_pred)
    _val_confusion_matrix = confusion_matrix(val_true, val_pred)

    tn, fp, fn, tp = _val_confusion_matrix.ravel()
    _val_specificity = tn / (tn+fp)
    _val_sensitivity = tp / (tp+fn)
    _val_false_positive_rate = fp / (fp+tn)

    target_names = ["Non Synonym [0]", "Is Synonym [1]"]
    final_report = classification_report(val_true, val_pred, target_names=target_names)

    with open(stat_file, 'w') as fo:
        fo.write("\n=== Data Distribution & Model Statistics ===")
        fo.write("\nNon Synonym [0]:\t{}\n".format(nonsym_count))
        fo.write("Is Synonym  [1]:\t{}\n".format(sym_count))

        fo.write("Accuracy:\t\t{:.4f}\n".format(_val_accuracy))
        fo.write("Precision:\t\t{:.4f}\n".format(_val_precision))
        fo.write("Recall:  \t\t{:.4f}\n".format(_val_recall))
        fo.write("F1-Score:\t\t{:.4f}\n".format(_val_f1))
        fo.write("MatthewCC:\t\t{:.4f}\n".format(_val_mcc))
        fo.write("Specificity:\t\t{:.4f}\n".format(_val_specificity))
        fo.write("Sensitivity:\t\t{:.4f}\n".format(_val_sensitivity))
        fo.write("FP. Rate:\t\t{:.4f}\n".format(_val_false_positive_rate))
        fo.write("Confusion Matrix:\n{}\n".format(_val_confusion_matrix))
        fo.write(final_report)
    return


def compute_rule_cluster_closure(job_name, paths, start_loop_cnt, submit_parameters, prepy_cmds):
    # Read MRCONSO_MASTER file
    # For every AUI, find all other AUIs sharing the same SCUI, LUI
    
    start_time = time.time()
    loop_cnt = start_loop_cnt
    
    all_clusters_dict = utils.load_pickle(paths['all_clusters_dict_pickle_fp'])
    merged_clusters_dict = {k:set([k]) for k in all_clusters_dict.keys()}
    utils.dump_pickle(merged_clusters_dict, paths['merged_clusters_pickle_fp'] + '_' + str(loop_cnt))
    if (FLAGS.regenerate_files is True) or (os.path.isfile(paths['merged_clusters_pickle_fp'] + '_' + str(loop_cnt) + '_keys') is False):
        utils.randomize_keys(paths['merged_clusters_pickle_fp'] + '_' + str(loop_cnt))
    
    while (True):
        start_loop = time.time()
        loop_cnt_str = '_' + str(loop_cnt)
        loop_cnt_next_str = '_' + str(loop_cnt + 1)
        
        
        merged_clusters_dict = utils.load_pickle(paths['merged_clusters_pickle_fp'] + loop_cnt_str)
        if len(merged_clusters_dict) == 0:
            utils.merge('excluded', paths['excluded_clusters_pickle_fp'] + '_', paths['excluded_clusters_pickle_fp'])
            utils.process_union(paths['excluded_clusters_pickle_fp'])
            break
            
        # There are clusters to be merged
        swarm_parameters = [
            " --workspace_dp=%s"%FLAGS.workspace_dp,
            " --umls_dl_dp=%s"%FLAGS.umls_dl_dp,
            " --closure_n_processes=%d"%FLAGS.closure_n_processes,
            " --job_name=%s"%(job_name),
            " --do_prep=false",
            " --do_compute_rule_closure_in_batch=true",
            " --do_eval=false",
            " --start_idx=start_index",
            " --end_idx=end_index",
            " --all_clusters_dict_pickle_fp=%s"%(paths['all_clusters_dict_pickle_fp']),
            " --merged_clusters_pickle_fp=%s"%(paths['merged_clusters_pickle_fp'] + loop_cnt_str),
            " --completed_inputs_fp=%s"%(paths['completed_inputs_fp'] + loop_cnt_next_str),
            " --merged_clusters_batch_pickle_fp=%s%s_start_index_end_index"%(paths['merged_clusters_batch_pickle_fp'], 
                                                                              loop_cnt_next_str),
            " --excluded_clusters_batch_pickle_fp=%s%s_start_index_end_index"%(paths['excluded_clusters_batch_pickle_fp'], 
                                                                                loop_cnt_next_str),
            " > %s"%(os.path.join(paths['rba_log_dp'], "%s%s_start_index_end_index.log "%(job_name, loop_cnt_next_str))),
        ]
        
        input_paras = {'inputs': paths['merged_clusters_pickle_fp'] + loop_cnt_str,
                      'inputs_keys': paths['merged_clusters_pickle_fp'] + '_keys' + loop_cnt_str}
        output_paras = {'merged': paths['merged_clusters_pickle_fp'] + loop_cnt_next_str, 
                        'excluded': paths['excluded_clusters_pickle_fp'] + loop_cnt_next_str,
                       }
        output_globs = {'merged': paths['merged_clusters_batch_pickle_fp'] + loop_cnt_next_str, 
                        'excluded': paths['excluded_clusters_batch_pickle_fp'] + loop_cnt_next_str,
                       }
        
        ninputs = 100 if FLAGS.debug is True else len(merged_clusters_dict)
        ntasks = 3 if FLAGS.debug is True else FLAGS.ntasks

        for k, fp in output_globs.items():
            utils.clear(fp)

        slurmjob = SlurmJob(job_name, FLAGS.run_slurm_job, prepy_cmds, swarm_parameters, submit_parameters,
                            ntasks, ninputs, loop_cnt_next_str, input_paras, output_paras, output_globs, paths, logger=logger)
        slurmjob.run()
        slurmjob.collect()
        
        'Process the output files by union the clusters from output lists'
        utils.process_union(paths['merged_clusters_pickle_fp'] + loop_cnt_next_str)
        utils.process_union(paths['excluded_clusters_pickle_fp'] + loop_cnt_next_str)
        
        utils.randomize_keys(paths['merged_clusters_pickle_fp'] + loop_cnt_next_str)
        
        logger.info("Finished %s loop %d in %d sec."%(job_name, loop_cnt, time.time() - start_loop))
        important_logger.info("Finished %s loop %d in %d sec."%(job_name, loop_cnt, time.time() - start_loop))
        loop_cnt += 1    
        
    excluded_clusters_dict = utils.load_pickle(paths['excluded_clusters_pickle_fp'])
    id_2_cluster_id_dict = dict()
    
    for cid, cluster in excluded_clusters_dict.items():
        for id1 in cluster:
            id_2_cluster_id_dict[id1] = cid
    
    utils.test_dict(id_2_cluster_id_dict, 'id_2_cluster_id_dict')
    utils.dump_pickle(id_2_cluster_id_dict, paths['trans_closure_pickle_fp'])
    logger.info("Finished %d loop with %d exclusive clusters in %d sec."%(loop_cnt, len(excluded_clusters_dict), time.time() - start_time))
    important_logger.info("Finished %d loop with %d exclusive clusters in %d sec."%(loop_cnt, len(excluded_clusters_dict), time.time() - start_time))
    del excluded_clusters_dict
    del id_2_cluster_id_dict
    return

def gen_id_2_cluster_info(paths):

    id2aui_dict = utils.load_pickle(paths['id2aui_pickle_fp'])
    all_clusters_dict = utils.load_pickle(paths['all_clusters_dict_pickle_fp'])
    id_2_cluster_id_dict = utils.load_pickle(paths['trans_closure_pickle_fp'])
    id_2_cluster_info_dict = dict()
    id_2_cluster_path_dict = dict()
    with tqdm(total=len(id_2_cluster_id_dict)) as pbar:
        for aui_id, cluster_ids in id_2_cluster_id_dict.items():
            aui = id2aui_dict[int(aui_id)]

            # Get SCUI and LUI/SG from cluster_id
            cluster_info = dict()
            tmp = cluster_ids.replace('(', '').replace(')','').split('+')
            cnt = 0        
            ids = []
            while cnt < len(tmp):
                if tmp[cnt] in all_clusters_dict:
                    ids.append(tmp[cnt])
                    cnt += 1
                else:
                 
                    tmp_id = '(' + tmp[cnt] + '+' + tmp[cnt+1] + ')'
                    if tmp_id in all_clusters_dict:
                        ids.append(tmp_id)
                        cnt += 2
            # Get common AUI IDs between any two adjacent cluster IDs
            if cluster_ids not in id_2_cluster_path_dict:
                id_2_cluster_path_dict[cluster_ids] = find_path(all_clusters_dict, ids, id2aui_dict)
            # Get AUI IDs
            for ID in ids:
                # Get aui_ids from this cluster
                auis = list(all_clusters_dict[ID])
                aui_ids = [id2aui_dict[int(aui2_id)] for aui2_id in auis]
                aui_ids.sort()
                cluster_info[ID] = aui_ids

            id_2_cluster_info_dict[aui] = cluster_info
            pbar.update(1)

    utils.dump_pickle(id_2_cluster_path_dict, paths['trans_closure_path_pickle_fp'])
    utils.dump_pickle(id_2_cluster_info_dict, paths['trans_closure_info_pickle_fp'])
    return

def find_path(all_clusters_dict, clusters_ids, id2aui_dict, start_cluster_id = None, end_cluster_id = None):
    start_id = start_cluster_id if start_cluster_id is not None else clusters_ids[0]
    all_auis = all_clusters_dict[start_id]
    path = start_id
    clusters_ids.remove(start_id)
    explore_ids = clusters_ids
    path_clusters = list()
    path_clusters.append(start_id)
    found_ID = start_id
    for num in range(len(clusters_ids)):
        for ID in explore_ids:
            common_auis = all_auis.intersection(all_clusters_dict[ID])
            if len(common_auis) > 0:
                found_ID = ID
                # Add to path
                path += '->{' + '|'.join([id2aui_dict[int(aui_id)] for aui_id in common_auis]) + '}->' + ID
                # Add to all_auis
                all_auis = all_auis.union(all_clusters_dict[ID])
                explore_ids.remove(ID)
                path_clusters.append(ID)
                break
        if (end_cluster_id is not None) and (found_ID == end_cluster_id):
            break
    
    return (path, path_clusters)

def compute_rule_cluster_closure_batch(start_idx, end_idx, all_clusters_dict_pickle_fp, completed_inputs_fp, merged_clusters_pickle_fp, 
                                       merged_clusters_batch_pickle_fp, excluded_clusters_batch_pickle_fp):
            
    start_time = time.time()
    all_clusters_dict = utils.load_pickle(all_clusters_dict_pickle_fp)
    merged_clusters_dict = utils.load_pickle(merged_clusters_pickle_fp)
    merged_clusters_dict_keys = utils.load_pickle(merged_clusters_pickle_fp + '_keys')

    #logger.info("Loading input to queue...")
    input_queue = queue.Queue()
    for idx, id1 in enumerate(merged_clusters_dict_keys):
        if (idx >= start_idx) and (idx <= end_idx):
            input_queue.put(id1)
    
    ninputs = input_queue.qsize()
    del merged_clusters_dict_keys
    
    output_queues_paras = {'completed': completed_inputs_fp, 
                          'merged': merged_clusters_batch_pickle_fp, 
                          'excluded': excluded_clusters_batch_pickle_fp,
                          } 
    node_parallel = NodeParallel(find_cluster_closure_batch, None, FLAGS.closure_n_processes,
                                 output_queues_paras,
                                 {'merged_clusters_dict': merged_clusters_dict,
                                  'all_clusters_dict': all_clusters_dict,
                                 }, dict(), logger = logger)  
    node_parallel.set_input_queue(input_queue)
    node_parallel.run()

    del merged_clusters_dict
    del node_parallel
    logger.info('Node finished %d inputs (%d, %d) in %d sec.'%(ninputs, start_idx, end_idx, time.time() - start_time))
    return

def find_cluster_closure_batch(input_queue, output_queues, being_processed, completed, **kwargs):
    
    merged_clusters_dict = kwargs['merged_clusters_dict']
    all_clusters_dict = kwargs['all_clusters_dict']
    pid = os.getpid()
    #logger.info("Process %d started."%pid)
    existing = 0
    num_inputs = 0
    while (True):
        k1 = input_queue.get()
        if (k1 != 'END'):
            num_inputs += 1
            being_processed[k1] = pid
            new_merged_clusters_dict, new_excluded_clusters_dict = dict(), dict()
            
            'Find all new possble k2 in new_cluster1 disjoint with cluster11'
            k1_mergeable = merged_clusters_dict[k1]  # cluster_id 
            new_k1_mergeable = set(k1_mergeable)
            for k2, k2_mergeable in merged_clusters_dict.items():

                if not new_k1_mergeable.isdisjoint(k2_mergeable):
                    new_k1_mergeable = new_k1_mergeable.union(k2_mergeable)

                if (k2 not in k1_mergeable): 
                    
                    if not all_clusters_dict[k1].isdisjoint(all_clusters_dict[k2]):
                        new_k1_mergeable.add(k2)
                    
            if len(new_k1_mergeable) > len(k1_mergeable):
                new_merged_clusters_dict[k1] = new_k1_mergeable
                output_queues['merged']['queue'].put(new_merged_clusters_dict)
            
            else:
                'No new k2 for k1, exclude it from the next hop'
                k1_mergeable = list(k1_mergeable)
                k1_mergeable.sort()
                new_id = '+'.join(k1_mergeable)
                k1_auis = all_clusters_dict[k1] # including auis from k1 cluster
                for k2 in k1_mergeable:
                    k1_auis = k1_auis.union(all_clusters_dict[k2])
                new_excluded_clusters_dict[new_id] = k1_auis
                output_queues['excluded']['queue'].put(new_excluded_clusters_dict)
            
            completed['queue'].put(k1)
            del being_processed[k1]
            del new_excluded_clusters_dict 
            del new_merged_clusters_dict
        else:
            completed['queue'].put('END')
            output_queues['merged']['queue'].put('END')
            output_queues['excluded']['queue'].put('END')    
            break

    #logger.info('Process %s processed %s inputs.'%(pid, num_inputs))
    return

def get_lui_sg_scui_cluster_dict(mrconso_master_rba_pickle_fp, all_clusters_dict_pickle_fp):
    if os.path.isfile(all_clusters_dict_pickle_fp):
        logger.info("Loading pickle %s ..."%(all_clusters_dict_pickle_fp))
        return utils.load_pickle(all_clusters_dict_pickle_fp)
    logger.info("Generating pickle %s ..."%(all_clusters_dict_pickle_fp))
    
    mrc_atoms = utils.load_pickle(mrconso_master_rba_pickle_fp)
    all_clusters_dict = dict()
    logger.info("Generating clusters of auis sharing the same scui or (lui,sg)")
    with tqdm(total = len(mrc_atoms)) as pbar:
        for aui, line in mrc_atoms.items():
            pbar.update(1)
            if (line['SCUI'] != ''):
                key = line['SCUI'].strip(' ')
                if key not in all_clusters_dict:
                    all_clusters_dict[key] = list()
                bisect.insort(all_clusters_dict[key], int(line['ID']))
            
            if line['SG'] is not None:
                for sg in line['SG']:
                    if line['LUI'] < sg:
                        key = '(' + line["LUI"] + '+' + sg + ')'
                    else:
                        key = '(' + sg + '+' + line["LUI"] + ')'
                    if key not in all_clusters_dict:
                        all_clusters_dict[key] = list()
                    bisect.insort(all_clusters_dict[key], int(line['ID']))
                    
    utils.test_dict(all_clusters_dict, 'all_clusters_dict')
    
    all_clusters_dict = {k:set(cluster) for k,cluster in all_clusters_dict.items()}    

    utils.test_dict(all_clusters_dict)
    print("All clusters: %d"%len(all_clusters_dict))
    important_logger.info("All clusters: %d"%len(all_clusters_dict))

    max_pairs = 0
    for k,v in all_clusters_dict.items():
        n = len(v)
        max_pairs += (n*(n+1))/2
    print("All pairs: %s"%max_pairs)
    important_logger.info("All pairs before clustering: %s"%max_pairs)
    
    utils.dump_pickle(all_clusters_dict, all_clusters_dict_pickle_fp)
    return all_clusters_dict # id -> [0, 3, 9, 20, etc.]

def compute_rule_pair_closure(mrc_atoms, trans_closure_pickle_fp):
    # Read MRCONSO_MASTER file
    # For every AUI, find all other AUIs sharing the same SCUI, LUI
    id_2_ids = get_id_2_ids_dict(mrc_atoms)
    reachable_ids = dict()
    
    # Find all pairs
    loop_cnt = 0
    found_cnt = 0
    manager = Manager()
    output_queue = manager.Queue()
    input_queue = manager.Queue()
        
    # Initialize the first round with all AUIs, from the next round, only reachable_ids will be explored
    for id1 in id_2_ids.keys():
        input_queue.put(id1)
        
    while (True):
        loop_cnt += 1
        logger.info("Start computing loop %d"%loop_cnt)
        logger.info("Loading input to queue...")
            
        for id1 in reachable_ids.keys():
            input_queue.put(id1)
        
        for idx in range(0, FLAGS.closure_n_processes):    
            input_queue.put("END")

        all_worker_processes_lst = []
        # Add worker processes
        for idx in range(0, FLAGS.closure_n_processes): 
            logger.info("Creating a new process.")
            p = Process(target=find_closure,
                        args=(input_queue, id_2_ids, reachable_ids, output_queue))
            all_worker_processes_lst.append(p)
    
        for p in all_worker_processes_lst:
            p.start()

        new_reachable_ids = dict()
        new_id_2_ids = dict()
        done_cnt = 0
        start_time = time.time()
        while (done_cnt < FLAGS.closure_n_processes):
            while not output_queue.empty():
                logger.info("Updating ... ")
                d = output_queue.get();
                if d is not None:
                    new_id_2_ids.update(d[0])
                    new_reachable_ids.update(d[1])
                    done_cnt += 1
            if input_queue.qsize() > 0:
                logger.info(input_queue.qsize())
            for p in all_worker_processes_lst:
                if p.is_alive():
                    if (time.time() - start_time) > 60:
                        start_time = time.time()
                        logger.info("Process %d is still working ... "%p.pid)
            time.sleep(10)
        logger.info("Done processing for the loop.")   
        for p in all_worker_processes_lst:
            p.join()
        
        logger.info("Processes finished the loop %d"%loop_cnt)
        
        if len(new_id_2_ids) > 0:
            id_2_ids.update(new_id_2_ids)
            reachable_ids.update(new_reachable_ids)
            logger.info('Found %d new pairs in loop %d'%(len(new_id_2_ids), loop_cnt))
            logger.info('Found %d new reachable ids in loop %d'%(len(new_id_2_ids), loop_cnt))
            del new_id_2_ids
            del new_reachable_ids
            for p in all_worker_processes_lst:
                del p
        else:
            break
        
    logger.info("Finished after %d loops in finding closure"%(loop_cnt))
    
    utils.dump_pickle(id_2_ids, trans_closure_pickle_fp)
    return id_2_ids

def find_closure(input_queue, id_2_ids, reachable_ids, output_queue):
    start_time = time.time()
    #logger.info("Process {} started.".format(os.getpid()))
    new_id_2_ids = dict()
    new_reachable_ids = dict()
    while (True):
        id1 = input_queue.get()
        if id1 == 'END':
            break
        if (id1 is not None):
            for id2 in id_2_ids[id1]:
                for id3 in id_2_ids[id2]:
                    if id3 not in id_2_ids[id1] and id3 != id1:
                        if (time.time() - start_time) > 60:
                            start_time = time.time()
                            total = len(id_2_ids[id1]) * len(id_2_ids[id2]) * len(id_2_ids[id3])
                            logger.info("Process {} processing total size {} from ({}, {}, {}) of ({}, {}, {})".format(os.getpid(), total, len(id_2_ids[id1]), len(id_2_ids[id2]), len(id_2_ids[id3]), id1, id2, id3))
                        if (id3 not in id_2_ids[id1]) and (id1 not in id_2_ids[id3]):
                            for _id in itertools.chain(id_2_ids[id1].keys(), id_2_ids[id2].keys(),id_2_ids[id3]):
                                if _id not in reachable_ids:
                                    new_reachable_ids[_id] = True
                            # Adding the new pair (id1, id3)
                            for _id1 in id_2_ids[id1]:
                                for _id3 in id_2_ids[id3]:
                                    if _id1 not in new_id_2_ids:
                                        new_id_2_ids[_id1] = dict()
                                    new_id_2_ids[_id1][_id3] = True
                                    if _id3 not in new_id_2_ids:
                                        new_id_2_ids[_id3] = dict()
                                    new_id_2_ids[_id3][_id1] = True
                            
    #logger.info("Process {} copying {} to queue.".format(os.getpid(), len(new_id_2_ids)))
    output_queue.put((new_id_2_ids, new_reachable_ids))
    del new_id_2_ids
    del new_reachable_ids
    #logger.info("Process {} finished.".format(os.getpid()))
    return

def get_id_2_ids_dict(mrc_atoms):
    all_clusters_dict = {}
    logger.info("Generating clusters of auis sharing the same scui or (lui,sg)")
    with tqdm(total = len(mrc_atoms)) as pbar:
        for aui, line in mrc_atoms.items():
            pbar.update(1)
            if (line['SCUI'] != ''):
                if (line['SCUI'] not in all_clusters_dict):
                    all_clusters_dict[line['SCUI']] = []
                bisect.insort(all_clusters_dict[line["SCUI"]], line['ID'])
            
            if line['SG'] is not None:
                for sg in line['SG']:
                    if (line["LUI"], sg) not in all_clusters_dict:
                        all_clusters_dict[(line["LUI"], sg)] = []
                    bisect.insort(all_clusters_dict[(line["LUI"], sg)], line['ID'])
    
    utils.test_dict(all_clusters_dict)
    print("All clusters: %d"%len(all_clusters_dict))
    max_pairs = 0
    for k,v in all_clusters_dict.items():
        n = len(v)
        max_pairs += (n*(n-1))/2
    print("All pairs: %s"%max_pairs)
    matrix_dim = len(mrc_atoms)
    id_2_ids = dict()
    
    with tqdm(total = len(all_clusters_dict)) as pbar:
        for cluster_id in all_clusters_dict.keys():
            pbar.update(1)
            if all_clusters_dict[cluster_id] is not None:
                comb = itertools.combinations(all_clusters_dict[cluster_id], 2)
                if comb is not None:
                    for id1, id2 in comb:
                        if id1 not in id_2_ids:
                            id_2_ids[id1] = dict()
                        id_2_ids[id1][id2] = True
                        if id2 not in id_2_ids:
                            id_2_ids[id2] = dict()
                        id_2_ids[id2][id1] = True
    utils.test_dict(id_2_ids)
    utils.test_big_item(id_2_ids, 1000)
    utils.test_big_item(id_2_ids, 10000)
    utils.test_big_item(id_2_ids, 100000)
    
    return id_2_ids

              
    
def get_mrc_atoms(mrconso_master_fp, mrconso_master_rba_pickle_fp):
    if (os.path.isfile(mrconso_master_rba_pickle_fp)):
        mrc_atoms = utils.load_pickle(mrconso_master_rba_pickle_fp)
        return mrc_atoms
    mrc_atoms = {}
    logger.info("Loading mrc_atoms from %s ..."%mrconso_master_fp)
    with tqdm(total = utils.count_lines(mrconso_master_fp)) as pbar:
        with open(mrconso_master_fp, 'r') as fi:
            reader = csv.DictReader(fi, fieldnames = FLAGS.mrconso_master_fields, delimiter=FLAGS.delimiter,doublequote=False,quoting=csv.QUOTE_NONE)
            idx = 0
            for line in reader:
                pbar.update(1)
                
                if line['SG'] is not None:
                    line['SG'] = set(line['SG'].split(','))
                    
                mrc_atoms[int(line['ID'])] = {'CUI':line["CUI"],'LUI':line["LUI"], 'AUI': line['AUI'],
                                              'SCUI': line["SCUI"], 'SG': line['SG'],\
                                             'STR': line['STR'], 'ID': int(line['ID'])}
    utils.dump_pickle(mrc_atoms, mrconso_master_rba_pickle_fp)                
    return mrc_atoms

def main(_):
    global utils
    utils = Utils()
    global important_logger 
    important_logger = utils.get_important_logger(logging.INFO, FLAGS.application_name, os.path.join(FLAGS.workspace_dp, FLAGS.important_info_fn))
    # Local folder, create if not existing
    paths = dict()
    paths['log_dp'] = os.path.join(FLAGS.workspace_dp, FLAGS.log_dn, FLAGS.rba_dn)
    Path(paths['log_dp']).mkdir(parents=True, exist_ok=True)

    paths['extra_dp'] = os.path.join(FLAGS.workspace_dp, FLAGS.extra_dn)
    Path(paths['extra_dp']).mkdir(parents=True, exist_ok=True)

    # For the executable files
    paths['bin_dp'] = os.path.join(FLAGS.workspace_dp, FLAGS.bin_dn)
    Path(paths['bin_dp']).mkdir(parents=True, exist_ok=True)

    paths['umls_meta_dp'] = FLAGS.umls_meta_dp if FLAGS.umls_meta_dp is not None else os.path.join(FLAGS.umls_dp, FLAGS.umls_meta_dn)

    paths['datasets_dp'] = FLAGS.datasets_dp if FLAGS.datasets_dp is not None else os.path.join(FLAGS.workspace_dp, FLAGS.datasets_dn)
    Path(paths['datasets_dp']).mkdir(parents=True, exist_ok=True)

    # For the dataset files
    paths['umls_dl_dp'] = FLAGS.umls_dl_dp if FLAGS.umls_dl_dp is not None else os.path.join(FLAGS.test_dataset_dp, FLAGS.umls_dl_dn)
    Path(paths['umls_dl_dp']).mkdir(parents=True, exist_ok=True)
    
    paths['log_filepath'] = os.path.join(paths['log_dp'],"%s.log"%(FLAGS.application_name))
    global logger 
    logger = utils.get_logger(logging.DEBUG, FLAGS.application_name, paths['log_filepath'])
    utils.set_logger(logger)
    
    paths['rba_dp'] = FLAGS.rba_dp if FLAGS.rba_dp is not None else os.path.join(FLAGS.workspace_dp, FLAGS.rba_dn)
    Path(paths['rba_dp']).mkdir(parents=True, exist_ok=True)
    
    paths['rba_bin_dp'] = FLAGS.rba_bin_dp if FLAGS.rba_bin_dp is not None else os.path.join(FLAGS.workspace_dp, FLAGS.bin_dn, FLAGS.rba_dn)
    Path(paths['rba_bin_dp']).mkdir(parents=True, exist_ok=True)
    paths['rba_log_dp'] = FLAGS.rba_log_dp if FLAGS.rba_log_dp is not None else os.path.join(FLAGS.workspace_dp, FLAGS.log_dn, FLAGS.rba_dn)
    Path(paths['rba_log_dp']).mkdir(parents=True, exist_ok=True)

    
    paths['mrconso_master_fp'] = os.path.join(paths['umls_dl_dp'], FLAGS.mrconso_master_fn)
    paths['mrconso_master_rba_pickle_fp'] = os.path.join(paths['umls_dl_dp'], FLAGS.mrconso_master_rba_pickle_fn)
    paths['id2aui_pickle_fp'] = os.path.join(paths['umls_dl_dp'], FLAGS.id2aui_pickle_fn)
    paths['aui2id_pickle_fp'] = os.path.join(paths['umls_dl_dp'], FLAGS.aui2id_pickle_fn)
    # Generate mrc_atoms
    
    paths['merged_clusters_bin_dp'] = os.path.join(paths['bin_dp'], FLAGS.rba_dn)
    Path(paths['merged_clusters_bin_dp']).mkdir(parents=True, exist_ok=True)
    paths['merged_clusters_batch_dp'] = os.path.join(paths['rba_dp'], FLAGS.merged_clusters_batch_dn)
    Path(paths['merged_clusters_batch_dp']).mkdir(parents=True, exist_ok=True)
    
    paths['all_clusters_dict_pickle_fp'] = os.path.join(paths['rba_dp'], FLAGS.all_clusters_dict_pickle_fn)
    paths['merged_clusters_pickle_fp'] = os.path.join(paths['rba_dp'], FLAGS.merged_clusters_pickle_fn)
    paths['excluded_clusters_pickle_fp'] = os.path.join(paths['rba_dp'], FLAGS.excluded_clusters_pickle_fn)
    paths['completed_inputs_fp'] = os.path.join(paths['rba_dp'], FLAGS.completed_inputs_fn)

    paths['merged_clusters_batch_pickle_fp'] = os.path.join(paths['merged_clusters_batch_dp'], FLAGS.merged_clusters_pickle_fn)
    paths['excluded_clusters_batch_pickle_fp'] = os.path.join(paths['merged_clusters_batch_dp'], FLAGS.excluded_clusters_pickle_fn)
    
    paths['trans_closure_pickle_fp'] = os.path.join(paths['rba_dp'], FLAGS.trans_closure_pickle_fn)
    paths['trans_closure_info_pickle_fp'] = os.path.join(paths['rba_dp'], FLAGS.trans_closure_info_pickle_fn)
    paths['trans_closure_path_pickle_fp'] = os.path.join(paths['rba_dp'], FLAGS.trans_closure_path_pickle_fn)
    paths['execute_py_fp'] = os.path.join(paths['bin_dp'], FLAGS.application_py_fn)

    # Parameters for generating swarm files
    job_name = FLAGS.job_name

    submit_parameters = [
        " -b 1",
        " --merge-output"
        " -g " + str(FLAGS.ram),
        " -t " + str(FLAGS.closure_n_processes),
        " --time 2-00:00:00 --logdir %s"%(paths['rba_log_dp']),
    ]
    prepy_cmds = ['source /data/nguyenvt2/libs/miniconda3/etc/profile.d/conda.sh', 
                  'conda activate %s'%FLAGS.conda_env]
    
    id_2_cluster_id = None
    mrc_atoms = get_mrc_atoms(paths['mrconso_master_fp'], paths['mrconso_master_rba_pickle_fp'])   
    if FLAGS.do_prep:
        
        if FLAGS.closure_approach == 'pair':
            mrc_atoms = get_mrc_atoms(paths['mrconso_master_fp'])
            id_2_ids = compute_rule_closure(mrc_atoms, paths['trans_closure_pickle_fp']) 
            
        elif FLAGS.closure_approach == 'cluster':
            logger.info("Loading all clusters ")
            get_lui_sg_scui_cluster_dict(paths['mrconso_master_rba_pickle_fp'], paths['all_clusters_dict_pickle_fp'])
            id_2_cluster_id = compute_rule_cluster_closure(job_name, paths, FLAGS.start_loop_cnt, submit_parameters, prepy_cmds)
            gen_id_2_cluster_info(paths)
    
    if FLAGS.do_compute_rule_closure_in_batch:
        compute_rule_cluster_closure_batch(FLAGS.start_idx, FLAGS.end_idx, FLAGS.all_clusters_dict_pickle_fp,
                                           FLAGS.completed_inputs_fp, FLAGS.merged_clusters_pickle_fp,  
                                           FLAGS.merged_clusters_batch_pickle_fp, FLAGS.excluded_clusters_batch_pickle_fp)
        
    if FLAGS.do_eval:

        paths['dataset_dn'] = FLAGS.test_dataset_dn if FLAGS.test_dataset_dn is not None else os.path.basename(FLAGS.test_dataset_dp)

        paths['dataset_rba_dp'] = os.path.join(paths['rba_dp'], paths['dataset_dn'])
        Path(paths['dataset_rba_dp']).mkdir(parents=True, exist_ok=True)

        paths['testfiles'] = []
        if (FLAGS.test_dataset_dp is not None) or (FLAGS.test_dataset_fp is not None):
            if FLAGS.test_dataset_dp is not None:
                paths['testfiles'] += glob.glob(os.path.join(FLAGS.test_dataset_dp, "*%s"%FLAGS.test_fn))
            if FLAGS.test_dataset_fp is not None:
                paths['testfiles'].append(FLAGS.test_dataset_fp)

        #mrc_atoms = utils.load_pickle(paths['mrconso_master_rba_pickle_fp'])
        #id_2_cluster_id = utils.load_pickle(paths['trans_closure_pickle_fp'])
        #id_2_cluster_info = utils.load_pickle(paths['trans_closure_info_pickle_fp'])
        rba_flavors = ["SCUI_LS_SG_TRANS", "LS_SG","SCUI", "SCUI_LS_SG"]
        logger.info("Computing lexical baseline for %s"%(paths['testfiles']))
        for test_fp in paths['testfiles']:
            fp = os.path.join(paths['dataset_rba_dp'], os.path.basename(test_fp))
            out_fp = {
                'FP': fp + "_false_positives",
                'FN': fp + "_false_negatives",
                'TP': fp + "_true_positives",
                'TN': fp + "_true_negatives",
                'pred': fp + "_predictions.PICKLE", 
                'stat': fp + "_stat.RRF",
            }
            if FLAGS.flavor == 'ALL':
                for flavor in rba_flavors:
                    #if flavor == 'SCUI_LS_SG_TRANS':
                    compute_rba_baseline(flavor, paths, test_fp, out_fp)
                    #compute_lexical_baseline(id_2_cluster_id, id_2_cluster_info, mrc_atoms, test_file, out_filepaths, flavor) 
            else:
                compute_rba_baseline(FLAGS.flavor, paths, test_fp, out_fp)

if __name__ == '__main__':
    app.run(main)

