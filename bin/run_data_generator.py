import csv
import re
import sys
import os
import math
import random
import shutil
import gc
import time
from pathlib import Path
import glob
from multiprocessing import Process, Queue, Manager
import queue
import copy

import numpy as np
from tqdm import tqdm
import bisect
import subprocess
import logging
import absl

from absl import app
from absl import flags

from common import SlurmJob, NodeParallel, Utils

FLAGS = flags.FLAGS
flags.DEFINE_string('f', '', 'kernel')

flags.DEFINE_string('server','FM','FM, Biowulf')
flags.DEFINE_string('application_name', 'run_data_generator', '')
flags.DEFINE_string('application_py_fn', 'run_data_generator.py', '')

flags.DEFINE_string('workspace_dp','/nfsvol2/projects/umls/deeplearning/aaai2020/2019AB','')
flags.DEFINE_string('important_info_fn','IMPORTANT_INFO.RRF','')
flags.DEFINE_string('umls_version_dp','/nfsvol2/projects/umls/deeplearning/aaai2020/2019AB/UMLS_VERSIONS/2020AA-ACTIVE','')
flags.DEFINE_string('dataset_version_dn','NEGPOS3', '')
# flags.DEFINE_string('umls_version_dp','/nfsvol2/projects/umls/deeplearning/aaai2020/2019AB/UMLS_VERSIONS/2020AA-ACTIVE','')
flags.DEFINE_string('umls_dl_dp', None, '')
flags.DEFINE_string('umls_dl_dn','META_DL', '')
flags.DEFINE_string('umls_meta_dp', None, '')
flags.DEFINE_string('umls_meta_dn','META','')
flags.DEFINE_string('datasets_dp', None, '')
flags.DEFINE_string('datasets_dn', "datasets", '')
flags.DEFINE_string('log_dn', 'logs','')
flags.DEFINE_string('bin_dn', 'bin','')
flags.DEFINE_string('extra_dn', 'extra', '')
# flags.DEFINE_string('workspace_dp','/data/nguyenvt2/aaai2020','')
# flags.DEFINE_string('umls_version_dp','/data/nguyenvt2/aaai2020/data/META_2019AB','')

flags.DEFINE_bool('gen_master_file', True, 'Your name.')
# flags.DEFINE_bool('gen_master_file', False, 'Your name.')
flags.DEFINE_bool('gen_pos_pairs', True, 'Your name.')
# flags.DEFINE_bool('gen_pos_pairs', False, 'Your name.')
flags.DEFINE_bool('gen_swarm_file', True, 'Your name.')
# flags.DEFINE_bool('gen_swarm_file', False, 'Your name.')
# flags.DEFINE_bool('exec_gen_neg_pairs_swarm', True, '')
flags.DEFINE_bool('exec_gen_neg_pairs_swarm', False, '')
# flags.DEFINE_bool('gen_neg_pairs', True, 'Your name.')
flags.DEFINE_bool('gen_neg_pairs', False, 'Your name.')
# flags.DEFINE_bool('gen_neg_pairs_batch', True, 'Your name.')
flags.DEFINE_bool('gen_neg_pairs_batch', False, 'Your name.')
flags.DEFINE_bool('gen_dataset', True, 'N')
# flags.DEFINE_bool('gen_dataset', False, 'N')
#flags.DEFINE_bool('run_slurm_job', True, '')
flags.DEFINE_bool('run_slurm_job', False, '')

flags.DEFINE_string('dataset_dp', None, '')
flags.DEFINE_string('dataset_dn', None, '')
flags.DEFINE_string('mrconso_fn','MRCONSO.RRF','')
flags.DEFINE_string('cui_sty_fn','MRSTY.RRF','')
flags.DEFINE_string('mrxns_fn','MRXNS_ENG.RRF','')
flags.DEFINE_string('mrxnw_fn','MRXNW_ENG.RRF','')
flags.DEFINE_string('sg_st_fn','SemGroups.txt','')

flags.DEFINE_string('mrx_nw_id_fn','NW_ID.RRF','')
flags.DEFINE_string('mrx_ns_id_fn','NS_ID.RRF','')
flags.DEFINE_string('nw_id_aui_fn','NW_ID_AUI.RRF','')
flags.DEFINE_string('mrconso_master_fn','MRCONSO_MASTER.RRF','')
flags.DEFINE_string('mrconso_master_randomized_fn','MRCONSO_MASTER_RANDOMIZED.RRF','')
flags.DEFINE_string('aui_info_gen_neg_pairs_pickle_fn', 'AUI_INFO_GEN_NEG_PAIRS.PICKLE', '')
flags.DEFINE_string('cui_to_aui_id_pickle_fn', 'CUI_AUI_ID.PICKLE', '')
flags.DEFINE_string('inputs_pickle_fn','INPUTS_DATA_GEN.PICKLE','')
flags.DEFINE_string('inputs_keys_ext', '_keys', '')
flags.DEFINE_string('inputs_pickle_fp', None, '')
flags.DEFINE_list('mrconso_master_fields', ["ID", "CUI", "LUI", "SUI", "AUI", "AUI_NUM", "SCUI", "NS_ID", "NS_LEN", "NS", "NORM_STR","STR", "SG"], '')
flags.DEFINE_list('mrconso_fields', ["CUI","LAT","TS","LUI","STT","SUI","ISPREF","AUI","SAUI","SCUI","SDUI","SAB","TTY","CODE","STR","SRL","SUPPRESS","CVF"], '')
flags.DEFINE_list('ds_fields', ['jacc', 'AUI1', 'AUI2', 'Label'],'')
flags.DEFINE_string("aui2id_pickle_fn", 'AUI2ID.PICKLE', "The output directory where the model checkpoints will be written.")
flags.DEFINE_string("id2aui_pickle_fn", 'ID2AUI.PICKLE', "The output directory where the model checkpoints will be written.")

flags.DEFINE_string('pos_pairs_fn','POS.RRF','')
flags.DEFINE_string('neg_file_prefix','NEG_FILES','')
flags.DEFINE_string('neg_batch_file_prefix','NEG_BATCH_FILES','')
flags.DEFINE_string('completed_inputs_fn','COMPLETED_INPUTS.RRF','')
flags.DEFINE_string('completed_inputs_fp', None, '')

flags.DEFINE_integer('gen_fold', 1, '')
flags.DEFINE_string('train_fn', 'TRAIN_DS.RRF', "The output directory where the model checkpoints will be written.")
flags.DEFINE_string('dev_fn', 'DEV_DS.RRF', "The output directory where the model checkpoints will be written.")
flags.DEFINE_string('test_fn', 'TEST_DS.RRF', "The output directory where the model checkpoints will be written.")

flags.DEFINE_string('swarm_fn','gen_neg_pairs.swarm','')
flags.DEFINE_string('submit_gen_neg_pairs_jobs_fn','gen_neg_pairs.submit','')
flags.DEFINE_string('data_generator_fn','run_data_generator.py','')

flags.DEFINE_bool('gen_neg_pairs_flavor_topn_sim', True, 'N')
flags.DEFINE_bool('gen_neg_pairs_flavor_ran_sim', True, '')
flags.DEFINE_bool('gen_neg_pairs_flavor_ran_nosim', True, '')

flags.DEFINE_string('training_type', "LEARNING_DS", '')
flags.DEFINE_string('gentest_type', "GENTEST_DS", '')

flags.DEFINE_string('neg_pairs_flavor_topn_sim', 'TOPN_SIM', 'TOPN_SIM, RAN_SIM, RAN_NOSIM')
flags.DEFINE_string('neg_pairs_flavor_ran_sim', 'RAN_SIM', '')
flags.DEFINE_string('neg_pairs_flavor_ran_nosim', 'RAN_NOSIM', '')
flags.DEFINE_string('neg_pairs_flavor_all', 'ALL', '')
flags.DEFINE_string('neg_pairs_flavor_ran', 'RAN', '')
# For SLURM
flags.DEFINE_string('user','nguyenvt2', '')
flags.DEFINE_string('job_name', None, '')
flags.DEFINE_string('job_id', None, '')
flags.DEFINE_string('conda_env', None, '')
flags.DEFINE_bool('debug', True, '')

flags.DEFINE_integer('ntasks', 490, '')
flags.DEFINE_integer('n_processes', 20, '')
flags.DEFINE_integer('ram', 80, '')
flags.DEFINE_integer('time_limit', None, '')
flags.DEFINE_integer('interval_time_between_create_process_task', 10, '')
flags.DEFINE_integer('interval_time_check_gen_neg_pairs_complete_task', 10, '')
flags.DEFINE_integer('interval_time_check_gen_neg_pairs_task', 10, '')

flags.DEFINE_integer('neg_to_pos_rate', 1, 'Your name.')
flags.DEFINE_integer('start_idx', 0, '')
flags.DEFINE_integer('end_idx', 100, '')
flags.DEFINE_string('delimiter', '|', '')
flags.DEFINE_integer('shuffle_cnt', 3, '')

csv.field_size_limit(sys.maxsize)


# Generate NW_ID.RRF file
def generate_nw_id_file(mrxnw_fn,mrx_nw_id_fn):
    nw_id_dict = dict()
    id_nw_dict = dict()
    nw_dict = dict()
    with open(mrxnw_fn,'r') as fi:
        reader = csv.DictReader(fi,fieldnames=["ENG","NW","CUI","LUI","SUI"],delimiter=FLAGS.delimiter)
        with tqdm(total = utils.count_lines(mrxnw_fn)) as pbar:
            for line in reader:
                pbar.update(1)
                if (line["NW"] is not None):
                        nw_dict[line["NW"]] = 1
    counter = 0
    with open(mrx_nw_id_fn,'w') as fo:
        with tqdm(total=len(nw_dict)) as pbar:
            for word,one in nw_dict.items():
                pbar.update(1)
                if word is not None:
                    id_nw_dict[counter] = word
                    nw_id_dict[word] = counter
                    fo.write(word + "|")
                    fo.write(str(counter) + "\n")
                counter += 1
    return nw_id_dict,id_nw_dict

def generate_ns_id_file(mrxns_fn,mrx_ns_id_fn,nw_id_dict):
    # Generate NS_ID.RRF
    ns_id_dict = {}
    ns_dict = {}
    ns_len_dict = {}
    with open(mrxns_fn,'r') as fi:#LAT,NSTR,CUI,LUI,SUI
        reader = csv.DictReader(fi,fieldnames=["ENG","NS","CUI","LUI","SUI"],delimiter=FLAGS.delimiter)
        with open(mrx_ns_id_fn,'w') as fo:
            with tqdm(total = utils.count_lines(mrxns_fn)) as pbar:
                for line in reader:
                    pbar.update(1)
                    ns = line["NS"].split(" ")
                    ns = list(set(ns))
                    ns_id = " ".join([str(nw_id_dict[nw]) for nw in ns])
                    ns_id_dict[line["SUI"]] = ns_id
                    ns_dict[line["SUI"]] = line["NS"]
                    ns_len_dict[line["SUI"]] = len(ns)
                    fo.write(line["SUI"] + "|")
                    fo.write(ns_id + "\n")
    return ns_id_dict,ns_dict,ns_len_dict
 
#def generate_manual_sg():
    # Generate a list of sources for the selected atoms        # Genarate the 
          
def generate_sg(sg_st_fn,cui_sty_fn):
    # Generate the SG field
    sty_2_SG = {}
    with open(sg_st_fn,'r') as fi: #ACTI|Activities & Behaviors|T052|Activity
        reader = csv.DictReader(fi,fieldnames=["SG","SG_STR","STY","STY_STR"],delimiter=FLAGS.delimiter)
        with tqdm(total = utils.count_lines(sg_st_fn)) as pbar:
            for line in reader:
                pbar.update(1)
                sty_2_SG[line["STY"]] = line["SG_STR"]

    cui_2_sg = {}
    with open(cui_sty_fn,'r') as fi: #CUI,TUI,STN,STY,ATUI,CVF
        reader = csv.DictReader(fi,fieldnames=["CUI","TUI","STN","STY","ATUI","CVF"],delimiter=FLAGS.delimiter)
        with tqdm(total = utils.count_lines(cui_sty_fn)) as pbar:
            for line in reader:
                pbar.update(1)
                if line["CUI"] in cui_2_sg:
                    cui_2_sg[line["CUI"]].append(sty_2_SG[line["TUI"]])
                else:
                    cui_2_sg[line["CUI"]] = [sty_2_SG[line["TUI"]]]

    # Remove duplidates in cui_2_sg
    for cui,sgs in cui_2_sg.items():
        sgs_set = list(set(sgs))
        cui_2_sg[cui] = sgs_set
    return cui_2_sg

def generate_aui_num(mrconso_fn):
    # Generate AUI_NUM
    cui_aui_dict = {}
    with tqdm(total = utils.count_lines(mrconso_fn)) as pbar:
        with open(mrconso_fn,'r') as fi:
            reader = csv.DictReader(fi, fieldnames = FLAGS.mrconso_fields,delimiter=FLAGS.delimiter,doublequote=False,quoting=csv.QUOTE_NONE)
            for line in reader:
                pbar.update(1)
                if ((line["LAT"] == "ENG") and (line["SUPPRESS"] == 'N')):
                    if line["CUI"] in cui_aui_dict:
                        cui_aui_dict[line["CUI"]].append(line["AUI"])
                    else:
                        cui_aui_dict[line["CUI"]] = [line["AUI"]]
    cui_aui_num_dict = {}                    
    for cui,aui_lst in cui_aui_dict.items():
        cui_aui_num_dict[cui] = len(aui_lst)
    return cui_aui_num_dict

def generate_aui_sim_num(mrconso_fn, nw_id_aui_dict, ns_id_dict):
    # Generate AUI_SIM_NUM
    aui_sim_num_dict = {}
    with tqdm(total = utils.count_lines(mrconso_fn)) as pbar:
        with open(mrconso_fn,'r') as fi:
            reader = csv.DictReader(fi, fieldnames = FLAGS.mrconso_fields,delimiter=FLAGS.delimiter,doublequote=False,quoting=csv.QUOTE_NONE)
            for line in reader:
                pbar.update(1)
                if ((line["LAT"] == "ENG") and (line["SUPPRESS"] == 'N')):
                    if line['SUI'] in ns_id_dict:
                        ns_id = ns_id_dict[line['SUI']]
                        aui_lst = []
                        if (ns_id.strip() != ''):
                            nw_lst = ns_id.split(' ')
                            for nw in nw_lst:
                                aui_lst += nw_id_aui_dict[int(nw)] 
                                
                        aui_sim_num_dict[line['SUI']] = len(list(set(aui_lst)))
                        del aui_lst
    return aui_sim_num_dict

# Generate NW_ID_AUI
def generate_nw_id_aui_fn(mrconso_master_fp, ns_id_dict,nw_id_aui_fn):
    nw_id_aui_dict = {}
    with tqdm(total = utils.count_lines(mrconso_master_fp)) as pbar:
        with open(mrconso_master_fp,'r') as fi:
            reader = csv.DictReader(fi, fieldnames = FLAGS.mrconso_master_fields, delimiter = FLAGS.delimiter,doublequote=False,quoting=csv.QUOTE_NONE)
            for line in reader:
                pbar.update(1)
                if line['SUI'] in ns_id_dict:
                    ns_id = ns_id_dict[line["SUI"]]
                    if (ns_id.strip() != ''):
                        nw_lst = [int(nw) for nw in ns_id.split(" ")]
                        for nw in nw_lst:
                            if nw in nw_id_aui_dict:
                                nw_id_aui_dict[nw].append(line['ID'])
                            else:
                                nw_id_aui_dict[nw] = [line['ID']]
                                
    with tqdm(total = len(nw_id_aui_dict)) as pbar:
        with open(nw_id_aui_fn,'w') as fo:
            for nw, aui_lst in nw_id_aui_dict.items():
                pbar.update(1)
                aui_set = set(aui_lst)
                aui_lst = list(aui_set)
                for aui_id in aui_lst:
                    fo.write(str(nw) + FLAGS.delimiter)
                    fo.write(str(aui_id) + "\n")
    return nw_id_aui_dict

# Generate AUI_LST sharing at least one normalized word with a given AUI
def generate_nw_id_aui_list(nw_id_aui_fn):
    nw_id_aui_pickle_fp = nw_id_aui_fn + ".PICKLE"
    if os.path.isfile(nw_id_aui_pickle_fp):
        nw_id_aui_dict = utils.load_pickle(nw_id_aui_pickle_fp)
        return nw_id_aui_dict
    nw_id_aui_dict = {}
    with tqdm(total = utils.count_lines(nw_id_aui_fn)) as pbar:
        with open(nw_id_aui_fn,'r') as fi:
            reader = csv.DictReader(fi, fieldnames=["NW", "AUI_ID"], delimiter = FLAGS.delimiter)
            for line in reader:
                pbar.update(1)
                if line["NW"] != '':
                    nw = int(line["NW"])
                    if nw in nw_id_aui_dict:
                        nw_id_aui_dict[nw].append(int(line["AUI_ID"]))
                    else:
                        nw_id_aui_dict[nw] = [int(line["AUI_ID"])]

    utils.dump_pickle(nw_id_aui_dict, nw_id_aui_pickle_fp)
    return nw_id_aui_dict

def generate_master_file(paths):
    
    logger.info("Generating NW_ID file from %s to %s ..."%(paths['mrxnw_fp'], paths['mrx_nw_id_fp']))
    nw_id_dict, id_nw_dict = generate_nw_id_file(paths['mrxnw_fp'], paths['mrx_nw_id_fp'])
    
    logger.info("Generating NS_ID file from %s to %s ..."%(paths['mrxns_fp'], paths['mrx_ns_id_fp']))
    ns_id_dict, ns_dict, ns_len_dict = generate_ns_id_file(paths['mrxns_fp'], paths['mrx_ns_id_fp'], nw_id_dict)
    
    logger.info("Generating SG from %s and %s..."%(paths['sg_st_fp'], paths['cui_sty_fp']))
    cui_2_sg = generate_sg(paths['sg_st_fp'], paths['cui_sty_fp'])
    
    logger.info("Generating AUI len from %s ..."%(paths['mrconso_fp']))
    cui_aui_num_dict = generate_aui_num(paths['mrconso_fp'])
        
    logger.info("Writing master file to %s ... "%paths['mrconso_master_fp'])
    cnt_nosim = 0
    # Generate the master file # CUI, LUI, SUI, AUI, AUI_NUM, AUI_SIM_NUM, SCUI, NS_ID, NS_LEN, NS, NORM_STR, STR, SG
    with open(paths['mrconso_fp'],'r') as fi: #CUI,LAT,TS,LUI,STT,SUI,ISPREF,AUI,SAUI,SCUI,SDUI,SAB,TTY,CODE,STR,SRL,SUPPRESS,CVF
        with open(paths['mrconso_master_fp'],'w') as fo: # CUI, LUI, SUI, AUI, AUI_NUM, SCUI, NS_ID, NS_LEN, NS, STR, SG
            reader = csv.DictReader(fi, fieldnames = FLAGS.mrconso_fields, delimiter = FLAGS.delimiter, doublequote=False, quoting=csv.QUOTE_NONE)
            with tqdm(total = utils.count_lines(paths['mrconso_fp'])) as pbar:
                cnt = 0
                for line in reader:
                    pbar.update(1)
                    if ((line["LAT"] == "ENG") and (line["SUPPRESS"] == 'N')):
                        string_tokens = re.sub(r'[^a-zA-Z0-9]', " ", line["STR"])
                        string_tokens = string_tokens.lower()
                        string_tokens = string_tokens.split()
                        string_joined = " ".join(string_tokens).strip()
                        if (string_joined != ''):
                            fo.write(str(cnt) + FLAGS.delimiter)
                            fo.write(line["CUI"] + FLAGS.delimiter)
                            fo.write(line["LUI"] + FLAGS.delimiter)
                            fo.write(line["SUI"] + FLAGS.delimiter)
                            fo.write(line["AUI"] + FLAGS.delimiter)
                            aui_num = cui_aui_num_dict[line["CUI"]]
                            fo.write(str(aui_num) + FLAGS.delimiter)
                            cnt_nosim += aui_num-1 if aui_num > 1 else 1
                            fo.write(line["SCUI"] + FLAGS.delimiter)
                            if line["SUI"] in ns_id_dict:
                                fo.write(ns_id_dict[line["SUI"]] + FLAGS.delimiter)
                                fo.write(str(ns_len_dict[line["SUI"]]) + FLAGS.delimiter)
                                fo.write(ns_dict[line["SUI"]] + FLAGS.delimiter)
                            else:
                                fo.write(FLAGS.delimiter + FLAGS.delimiter + FLAGS.delimiter)
                            fo.write(string_joined + FLAGS.delimiter)
                            fo.write(line["STR"] + FLAGS.delimiter)    
                            if line["CUI"] in cui_2_sg:
                                fo.write(",".join(cui_2_sg[line["CUI"]]) + "\n")
                            else:
                                fo.write("\n")
                            cnt += 1
    logger.info("Generating NW ID to AUI file %s to %s"%(paths['mrconso_master_fp'], paths['nw_id_aui_fp']))
    nw_id_aui_dict = generate_nw_id_aui_fn(paths['mrconso_master_fp'], ns_id_dict, paths['nw_id_aui_fp'])
            
    logger.info("cnt_nosim: %d"%(cnt_nosim))

def get_aui_lst(input_aui, nw_id_aui_dict, aui_info):
    input_aui_ns_id = aui_info[input_aui]["NS_ID"]
    aui_lst = []
    for nw in input_aui_ns_id:
        aui_lst += nw_id_aui_dict[int(nw)]  
    return list(set(aui_lst))

def get_aui_info(mrconso_master_fn):
    aui_ns_id_dict = {}
    #cnt = 0
    id_aui_dict = {}
    with open(mrconso_master_fn,'r') as fi:
        reader = csv.DictReader(fi, fieldnames = FLAGS.mrconso_master_fields, delimiter=FLAGS.delimiter,doublequote=False,quoting=csv.QUOTE_NONE)
        with tqdm(total = utils.count_lines(mrconso_master_fn)) as pbar:
            for line in reader:
                pbar.update(1)
                #line['AUI_ID'] = cnt
                id_aui_dict[line['ID']] = line['AUI']
                if line["NS_ID"] != '':
                    line["NS_ID"] = set([int(nw) for nw in line["NS_ID"].split(" ")])
                else:
                    line["NS_ID"] = set()
                aui_ns_id_dict[line["AUI"]] = line
                #cnt += 1
    return aui_ns_id_dict, id_aui_dict

def get_aui_info_gen_neg_pairs(mrconso_master_fn, aui_info_gen_neg_pairs_pickle_fp):
    aui_ns_id_dict = {}
    cnt = 0
    id_aui_dict = {}
    
    if os.path.isfile(aui_info_gen_neg_pairs_pickle_fp):
        aui_ns_id_dict = utils.load_pickle(aui_info_gen_neg_pairs_pickle_fp)
        return aui_ns_id_dict
    with open(mrconso_master_fn,'r') as fi:
        reader = csv.DictReader(fi, fieldnames = FLAGS.mrconso_master_fields, delimiter=FLAGS.delimiter,doublequote=False,quoting=csv.QUOTE_NONE)
        with tqdm(total = utils.count_lines(mrconso_master_fn)) as pbar:
            for line in reader:
                pbar.update(1)
                
                if line["NS_ID"] != '':
                    line["NS_ID"] = set([int(nw) for nw in line["NS_ID"].split(" ")])
                else:
                    line["NS_ID"] = set()
                aui_info = {}
                aui_info['AUI'] = line['AUI']
                aui_info['AUI_NUM'] = line['AUI_NUM']
                aui_info['CUI'] = line['CUI']
                aui_info['NS_ID'] = line['NS_ID']
                
                aui_ns_id_dict[int(line["ID"])] = aui_info
    utils.dump_pickle(aui_ns_id_dict, aui_info_gen_neg_pairs_pickle_fp)
    return aui_ns_id_dict

def get_cui_to_aui_dict(mrconso_master_fn, cui_to_aui_id_pickle_fp):
    cui_to_aui_dict = {}
    if os.path.isfile(cui_to_aui_id_pickle_fp):
        cui_to_aui_dict = utils.load_pickle(cui_to_aui_id_pickle_fp)
        return cui_to_aui_dict

    with open(mrconso_master_fn,'r') as fi:
        reader = csv.DictReader(fi, fieldnames = FLAGS.mrconso_master_fields, delimiter=FLAGS.delimiter,doublequote=False,quoting=csv.QUOTE_NONE)
        with tqdm(total = utils.count_lines(mrconso_master_fn)) as pbar:
            for line in reader:
                pbar.update(1)
                if line['CUI'] not in cui_to_aui_dict:
                    cui_to_aui_dict[line["CUI"]] = list()
                cui_to_aui_dict[line['CUI']].append(int(line['ID']))

    utils.dump_pickle(cui_to_aui_dict, cui_to_aui_id_pickle_fp)
    return cui_to_aui_dict


def reverse_insort(a, x, comp, lo=0, hi=None):
    """Insert item x in list a, and keep it reverse-sorted assuming a
    is reverse-sorted.

    If x is already in a, insert it to the right of the rightmost x.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if x[comp] > a[mid][comp]: hi = mid
        else: lo = mid+1
    a.insert(lo, x)

def compute_jaccard(ns1,ns2):
    u = len(ns1 | ns2)
    i = len(ns1 & ns2)
    if u == 0:
        return 0
    return round(i/u,2)

def generate_negative_pairs(job_name, paths, prepy_cmds, submit_parameters):

    start_time = time.time()
 
    # Get all AUIs 
    if os.path.isfile(paths['inputs_pickle_fp']):
        all_inputs = utils.load_pickle(paths['inputs_pickle_fp'])
        all_inputs_keys = utils.randomize_keys(paths['inputs_pickle_fp'])
    else:
        all_inputs = dict()
        with open(paths['mrconso_master_fp'],'r') as fi:
            reader = csv.DictReader(fi,fieldnames = FLAGS.mrconso_master_fields,delimiter=FLAGS.delimiter,doublequote=False,quoting=csv.QUOTE_NONE)
            for line in reader:
                all_inputs[int(line['ID'])] = int(line['AUI_NUM']) - 1 if int(line['AUI_NUM']) > 1 else 1
        utils.dump_pickle(all_inputs, paths['inputs_pickle_fp'])
        all_inputs_keys = utils.randomize_keys(paths['inputs_pickle_fp'])

    #logger.info("Loading NW ID to AUI list from %s"%paths['nw_id_aui_fp'])
    nw_id_aui_dict = generate_nw_id_aui_list(paths['nw_id_aui_fp'])
    #logger.info("Loading AUI info from %s"%paths['mrconso_master_fp'])
    aui_info = get_aui_info_gen_neg_pairs(paths['mrconso_master_fp'], paths['aui_info_gen_neg_pairs_pickle_fp'])

    #logger.info("Loading CUI to AUI ID dict ...")
    cui_to_aui_id_dict = get_cui_to_aui_dict(paths['mrconso_master_fp'], paths['cui_to_aui_id_pickle_fp'])
    logger.info("neg_to_pos_rate: %d"%FLAGS.neg_to_pos_rate)
    debug = 'true' if FLAGS.debug is True else 'false'
    swarm_parameters = [
        " --workspace_dp=%s"%FLAGS.workspace_dp,
        " --umls_version_dp=%s"%FLAGS.umls_version_dp,
        " --umls_dl_dp=%s"%FLAGS.umls_dl_dp,
        " --n_processes=%d"%FLAGS.n_processes,
        " --job_name=%s"%(job_name),
        " --start_idx=start_index",
        " --end_idx=end_index",
        " --dataset_version_dn=%s"%(FLAGS.dataset_version_dn),
        " --gen_master_file=false",
        " --gen_pos_pairs=false",  
        " --gen_neg_pairs=false",
        " --gen_neg_pairs_batch=true",
        " --gen_dataset=false",
        " --neg_to_pos_rate=%d"%FLAGS.neg_to_pos_rate,
        " --conda_env=%s"%FLAGS.conda_env,
        " --debug=%s"%debug,
        " --inputs_pickle_fp=%s"%paths['inputs_pickle_fp'],
        " --completed_inputs_fp=%s"%(paths['completed_inputs_fp']),
        " > %s"%(os.path.join(paths['data_gen_log_dp'], "%s_start_index_end_index.log "%(job_name))),
        ]

    input_paras = {'inputs': paths['inputs_pickle_fp'],
                  'inputs_keys': paths['inputs_pickle_fp'] + FLAGS.inputs_keys_ext}

    output_paras = {'training_type': paths['neg_pairs_training_flavor_fp'], 
                    'gentest_type': paths['neg_pairs_gentest_flavor_fp']}
    output_globs = {'training_type': paths['neg_pairs_training_flavor_glob'], 
                    'gentest_type': paths['neg_pairs_gentest_flavor_glob']}

    ninputs = 100 if FLAGS.debug is True else len(all_inputs)
    ntasks = 3 if FLAGS.debug is True else FLAGS.ntasks

    for k, fp in output_globs.items():
        for flavor, flavor_glob in fp.items():
            utils.clear(flavor_glob)

    slurmjob = SlurmJob(job_name, FLAGS.run_slurm_job, prepy_cmds, swarm_parameters, submit_parameters, ntasks, ninputs, None, input_paras, output_paras, output_globs, paths, logger=logger)
    slurmjob.run()
    
    'Process the output files by union the clusters from output lists'
    #generate_dataset(paths)

def generate_negative_pairs_node_batch(paths, start_idx, end_idx, neg_to_pos_rate, n_processes=2):

    start_time = time.time()
    all_inputs_dict_keys = utils.load_pickle(paths['inputs_pickle_fp'] + FLAGS.inputs_keys_ext)

    if FLAGS.debug is True:
        logger.info("Loading input to queue...")
    input_queue = queue.Queue()
    for idx, id1 in enumerate(all_inputs_dict_keys):
        if (idx >= start_idx) and (idx <= end_idx):
            input_queue.put(id1)

    ninputs = input_queue.qsize()
    del all_inputs_dict_keys

    #logger.info("Loading NW ID to AUI list from %s"%paths['nw_id_aui_fp'])
    nw_id_aui_dict = generate_nw_id_aui_list(paths['nw_id_aui_fp'])
    #logger.info("Loading AUI info from %s"%paths['mrconso_master_fp'])
    aui_info = get_aui_info_gen_neg_pairs(paths['mrconso_master_fp'], paths['aui_info_gen_neg_pairs_pickle_fp'])

    #logger.info("Loading CUI to AUI ID dict ...")
    cui_to_aui_id_dict = get_cui_to_aui_dict(paths['mrconso_master_fp'], paths['cui_to_aui_id_pickle_fp'])
    logger.info("neg_to_pos_rate: %d"%FLAGS.neg_to_pos_rate)

    output_queues_paras = {'completed': paths['completed_inputs_fp'],

                           'write_tr_queues': paths['neg_pairs_training_flavor_batch_fp'],
                           'write_tr_queues': paths['neg_pairs_training_flavor_batch_fp'],
                           'write_ge_queues': paths['neg_pairs_gentest_flavor_batch_fp'],
                          }
    node_parallel = NodeParallel(generate_negative_pairs_batch, write_flavor_to_file, FLAGS.n_processes,
                                 output_queues_paras, 
                                 {'paths': paths, 'neg_to_pos_rate': neg_to_pos_rate,
                                  'nw_id_aui_dict':nw_id_aui_dict, 'aui_info':aui_info,
                                  'cui_to_aui_id_dict':cui_to_aui_id_dict,
                                 }, {'paths': paths, 'aui_info': aui_info,}, logger = logger)
    node_parallel.set_input_queue(input_queue)
    node_parallel.run()

    del node_parallel     
    logger.info('Node finished %d inputs (%d, %d) in %d sec.'%(ninputs, start_idx, end_idx, time.time() - start_time))
    return
    

def generate_negative_pairs_batch(input_queue, output_queues, being_processed, completed, **kwargs):
    start_time = time.time()
    paths = kwargs['paths']
    neg_to_pos_rate = kwargs['neg_to_pos_rate']
    nw_id_aui_dict = kwargs['nw_id_aui_dict']
    aui_info = kwargs['aui_info']
    cui_to_aui_id_dict = kwargs['cui_to_aui_id_dict']
   
    num = 0
    p_start = time.time()
    start_time = time.time()
    p_id = os.getpid()

    aui2s_nosim = {aui_id:True for aui_id in aui_info.keys()}
    aui2s_ran = {aui_id:True for aui_id in aui_info.keys()}
    aui2s_sim = dict()

    aui1, k, n, cnt_sim, cnt_no_sim, max_size, sec = 0, 0, 0, 0, 0, 0, 0
    auis, topn2n = [], []

    while (input_queue.empty() is False):
        start_time = time.time()
        aui1 = input_queue.get()
        if aui1 == 'END':
            break
        being_processed[aui1] = p_id
        n = int(aui_info[aui1]["AUI_NUM"])-1
        #logger.info("aui_info[{}]: {} with n: {}".format(aui1, aui_info[aui1], n))
        if n == 0:
            n = 1 # To cover all AUIs
        k = neg_to_pos_rate*n

        aui2s_sim_id_removed, aui2s_nosim_id_removed, aui2s_ran_id_removed = dict(), dict(), dict()
 
        auis = get_aui_lst(aui1,nw_id_aui_dict,aui_info)
        cnt_sim = len(auis)
        #logger.info("auis[{}]: {}".format(aui1, auis))
        aui2s_sim = {aui2:True for aui2 in auis}
        auis.clear()

        # Remove auis from the same CUI, will be added back at the end of the loop
        #logger.info("cui_to_aui_id_dict[{}]: {}".format(aui1, [id_aui_dict[id2] for id2 in cui_to_aui_id_dict[aui_info[aui1]['CUI']]]))
        for id2 in cui_to_aui_id_dict[aui_info[aui1]['CUI']]:
            if id2 in aui2s_sim:
                aui2s_sim_id_removed[id2] = aui2s_sim[id2]
                del aui2s_sim[id2]
            if id2 in aui2s_nosim:
                aui2s_nosim_id_removed[id2] = aui2s_nosim[id2]
                del aui2s_nosim[id2]
            if id2 in aui2s_ran:
                aui2s_ran_id_removed[id2] = aui2s_ran[id2]
                del aui2s_ran[id2]

        # ============ GENERATING training ==============
        top2n = compute_topn_sim(aui1, k*2, aui2s_sim, aui_info, nw_id_aui_dict)
        # Take the first half to ds and the second half to gentest
        max_size = len(top2n) 
        topn_sim_pairs_training = top2n[0:k]
        if max_size > 0:
            topn_sim_pairs_gentest = top2n[k:max_size]
        else:
            topn_sim_pairs_gentest = []
        
        cnt_sim_neg = len(aui2s_sim)

        # ============ GENERATING GENTEST ==============
        # Remove the existing pairs from training
        ran_sim_pairs_training = compute_ran_sim(aui1, k, aui2s_sim, aui_info, nw_id_aui_dict) 
        for pair in (topn_sim_pairs_training + ran_sim_pairs_training):
            id2 = pair['AUI2']
            if id2 in aui2s_sim:
                aui2s_sim_id_removed[id2] = aui2s_sim[id2]
                del aui2s_sim[id2]
            del id2

        output_queues['write_tr_queues']['queue'].put({FLAGS.neg_pairs_flavor_topn_sim:topn_sim_pairs_training})
        del topn_sim_pairs_training
        #gc.collect()
        output_queues['write_ge_queues']['queue'].put({FLAGS.neg_pairs_flavor_topn_sim:topn_sim_pairs_gentest})
        del topn_sim_pairs_gentest
        #gc.collect()

        output_queues['write_tr_queues']['queue'].put({FLAGS.neg_pairs_flavor_ran_sim:ran_sim_pairs_training})
        del ran_sim_pairs_training
        #gc.collect()
        ran_sim_pairs_gentest = compute_ran_sim(aui1, k, aui2s_sim, aui_info, nw_id_aui_dict)
        output_queues['write_ge_queues']['queue'].put({FLAGS.neg_pairs_flavor_ran_sim:ran_sim_pairs_gentest})
        del ran_sim_pairs_gentest
        del aui2s_sim_id_removed
        del aui2s_sim
        #gc.collect()
 
        ran_nosim_pairs_training = compute_ran_nosim(aui1, k, aui2s_nosim, aui_info, nw_id_aui_dict)
        for pair in ran_nosim_pairs_training:
            id2 = pair["AUI2"]
            if id2 in aui2s_nosim:
                aui2s_nosim_id_removed[id2] = aui2s_nosim[id2]
                del aui2s_nosim[id2]
            del id2

        output_queues['write_tr_queues']['queue'].put({FLAGS.neg_pairs_flavor_ran_nosim:ran_nosim_pairs_training})
        del ran_nosim_pairs_training
        #gc.collect()

        ran_nosim_pairs_gentest = compute_ran_nosim(aui1, k, aui2s_nosim, aui_info, nw_id_aui_dict)
        output_queues['write_ge_queues']['queue'].put({FLAGS.neg_pairs_flavor_ran_nosim:ran_nosim_pairs_gentest})
        for id2, aui2 in aui2s_nosim_id_removed.items():
            aui2s_nosim[id2] = aui2
        del ran_nosim_pairs_gentest
        del aui2s_nosim_id_removed
        #gc.collect()
   
        ran_pairs_training = compute_ran(aui1, k, aui2s_ran, aui_info, nw_id_aui_dict)
        for pair in ran_pairs_training:
            id2 = pair["AUI2"]
            if id2 in aui2s_ran:
                aui2s_ran_id_removed[id2] = aui2s_ran[id2]
                del aui2s_ran[id2]
            del id2

        output_queues['write_tr_queues']['queue'].put({FLAGS.neg_pairs_flavor_ran:ran_pairs_training})
        del ran_pairs_training
        #gc.collect()

        ran_pairs_gentest = compute_ran(aui1, k, aui2s_ran, aui_info, nw_id_aui_dict)
        output_queues['write_ge_queues']['queue'].put({FLAGS.neg_pairs_flavor_ran:ran_pairs_gentest})
        for id2, aui2 in aui2s_ran_id_removed.items():
            aui2s_ran[id2] = aui2
        del ran_pairs_gentest
        del aui2s_ran_id_removed
        #gc.collect()
        
        #logger.info(top_n)
        sec = time.time() - start_time
        num += 1
        completed['queue'].put(FLAGS.delimiter.join([str(aui1), str(cnt_sim), str(cnt_sim_neg)]))
        #logger.debug("pid: {}, num: {}, aui1: {}, n: {}, sec: {}, ".format(p_id, num, aui1, n, int(sec)) + 
        #     "topn_tr : {} vs. rsim_tr: {} vs. rnosim_tr: {} vs. ran_tr: {} ".format(len(topn_sim_pairs_training), len(ran_sim_pairs_training), len(ran_nosim_pairs_training), len(ran_pairs_training)) +
        #     "topn_ge : {} vs. rsim_ge: {} vs. rnosim_ge: {} vs. ran_ge: {}".format(len(topn_sim_pairs_gentest), len(ran_sim_pairs_gentest), len(ran_nosim_pairs_gentest), len(ran_pairs_gentest)))

        del being_processed[aui1]
        gc.collect()

    completed['queue'].put('END')
    output_queues['write_tr_queues']['queue'].put('END')
    output_queues['write_ge_queues']['queue'].put('END')
    p_end = time.time() - p_start
    logger.info("Process %s finished %d in %d sec."%(p_id, num, p_end))
    return
    
def compute_topn_sim(input_aui, k, aui2s_sim, aui_info, nw_id_aui_dict):
    top_n = []
    nws_1 = aui_info[input_aui]
    #logger.info("Len of nws_1: %d"%(len(nws_1)))
    #logger.info("Len of auis_info: %d"%(len(aui_info)))

    label = '0'
    jacc = 0
    for id2, aui2 in aui2s_sim.items():
        if id2 not in aui_info:
            #logger.info("aui_info[%d]: %s"%(id2, aui_info[id2]))
            logger.info("%d not in aui_info"%(id2))
        jacc = compute_jaccard(nws_1["NS_ID"],aui_info[id2]["NS_ID"])
        reverse_insort(top_n, {'jacc':jacc, 'AUI1': input_aui, 'AUI2':id2, 'Label':label},'jacc')
        top_n = top_n[0:k]
    del jacc
    del label
    del nws_1
    #gc.collect()
    return top_n

def compute_ran_sim(input_aui, k, aui2s_sim, aui_info, nw_id_aui_dict):
    neg_pairs = []
    nws_1 = aui_info[input_aui]
    #logger.info(len(auis))
    label = '0'
    jacc = 0
    
    # If not having enough ids
    if (len(aui2s_sim) <= k):
        for id2,aui2 in aui2s_sim.items():
            jacc = compute_jaccard(nws_1["NS_ID"],aui_info[id2]["NS_ID"])
            neg_pairs.append({'jacc':jacc, 'AUI1': input_aui, 'AUI2':id2, 'Label':label})
    else:
        selected = random.sample(aui2s_sim.items(),k)
        for id2,aui2 in selected:
            jacc = compute_jaccard(nws_1["NS_ID"],aui_info[id2]["NS_ID"])
            neg_pairs.append({'jacc':jacc, 'AUI1': input_aui, 'AUI2':id2, 'Label':label})    
        del selected
    del jacc
    del label
    del nws_1
    #gc.collect()
    return neg_pairs

def compute_ran_nosim(input_aui, k, aui2s_nosim, aui_info, nw_id_aui_dict):
    neg_pairs = []
    nws_1 = aui_info[input_aui]
    #logger.info(len(auis))
    jacc = 0
    label = '0'
    
    # If not having enough ids
    if (len(aui2s_nosim) <= k):
        for id2,aui2 in aui2s_nosim.items():
            neg_pairs.append({'jacc':jacc, 'AUI1': input_aui, 'AUI2':id2, 'Label':label})
    else:
        selected = random.sample(aui2s_nosim.items(),k)
        for id2,aui2 in selected:
            neg_pairs.append({'jacc':jacc, 'AUI1': input_aui, 'AUI2':id2, 'Label':label})    
        del selected
    if (len(neg_pairs) != k):
        logger.debug("Must check %s: , k: %d vs. ran_nosim: %d"%(input_aui, k, len(neg_pairs)))
    del jacc
    del label
    del nws_1   
    #gc.collect() 
    return neg_pairs

def compute_ran(input_aui, k, aui2s_ran, aui_info, nw_id_aui_dict):
    neg_pairs = []
    nws_1 = aui_info[input_aui]
    #logger.info(len(auis))
    label = '0'
    jacc = 0

    # If not having enough ids
    if (len(aui2s_ran) <= k):
        for id2,aui2 in aui2s_ran.items():
            jacc = compute_jaccard(nws_1["NS_ID"],aui_info[id2]["NS_ID"])
            neg_pairs.append({'jacc':jacc, 'AUI1': input_aui, 'AUI2':id2, 'Label':label})
    else:
        selected = random.sample(aui2s_ran.items(),k)
        for id2,aui2 in selected:
            jacc = compute_jaccard(nws_1["NS_ID"],aui_info[id2]["NS_ID"])
            neg_pairs.append({'jacc':jacc, 'AUI1': input_aui, 'AUI2':id2, 'Label':label})
        del selected
    del jacc
    del label
    del nws_1
    #gc.collect()
    return neg_pairs

def write_flavor_to_file(num_workers, okey, output_queues_paras, output_queue, **kwargs):
    paths = kwargs['paths']
    fp = output_queues_paras[okey]['fp']
    logger.info('Process %s created for %s'%(okey, fp))
    aui_info = kwargs['aui_info']

    cnt = dict()
    aui1s = dict()
    files = dict()
    flavors = [flavor for flavor in fp.keys() if flavor != FLAGS.neg_pairs_flavor_all]

    for flavor in flavors:
        cnt[flavor] = 0
        aui1s[flavor] = 0
        files[flavor] = open(fp[flavor], 'a')

    cnt_end = 0
    done = False
    while (True):
        if cnt_end == num_workers:
                done = True
                break
        while (output_queue['queue'].empty() is False):
            #logger.debug("Writer %s done receiving from %s workers"%(okey, cnt_end))
            if cnt_end == num_workers:
                done = True
                break
            pairs = output_queue['queue'].get()
            if pairs != 'END':
                for flavor, lst in pairs.items():                            
                    aui1s[flavor] += 1
                    for v in lst:
                        files[flavor].write(str(v["jacc"]) + FLAGS.delimiter)
                        files[flavor].write(aui_info[v["AUI1"]]['AUI'] + FLAGS.delimiter)
                        files[flavor].write(aui_info[v["AUI2"]]['AUI'] + FLAGS.delimiter + v['Label'] + "\n")
                        cnt[flavor] += 1
                    lst.clear()
            else:
                cnt_end += 1
                if cnt_end == num_workers:
                    done = True
                    break
            del pairs
            #gc.collect()
        if done is True:
            break   
    output_queues_paras[okey]['status'] = True         
    for flavor in flavors:
        files[flavor].close()
        logger.info("Writer finished %d pairs for %d aui1s in %s"%(cnt[flavor], aui1s[flavor], fp[flavor]))
    return

def write_pairs_to_file(pairs_ds, pairs_fn):
    if len(pairs_ds) > 0:
        with open(pairs_fn,'a') as fo:
            for k in pairs_ds:                            
                fo.write(str(k["jacc"]) + FLAGS.delimiter)
                fo.write(k["AUI1"] + FLAGS.delimiter)
                fo.write(k["AUI2"] + FLAGS.delimiter + k['Label'] + "\n")

def write_list_to_file(pairs_ds, pairs_fn):
    with open(pairs_fn,'w') as fo:
            for k in dedup_list(pairs_ds): 
                fo.write(k)
                #fo.write('\n')
    return

def read_file_to_list(pairs_fp):
    pairs = []
    logger.info("Loading file %s ..."%pairs_fp)      
    with open(pairs_fp) as f:
        for line in f:
            inputs = line.split(FLAGS.delimiter)
            if len(inputs) == 4:
                pairs.append(line)
    return pairs
                
def read_file_to_pairs(pairs_fp):
    pairs = []
    logger.info("Loading file %s ..."%pairs_fp)                
    with open(pairs_fp, 'r') as fi:
        reader = csv.DictReader(fi,fieldnames=["jacc", "AUI1", "AUI2", "Label"], delimiter=FLAGS.delimiter)
        with tqdm(total = utils.count_lines(pairs_fp)) as pbar:
            for line in reader:
                pbar.update(1)
                if len(line) == 4: 
                    # Eliminate insatisfiable input pairs
                    pairs.append(line)
    
    return pairs

def dedup_list(ds):
    dedup_ds = {}
    for item in ds:
        dedup_ds[item] = True

    return list(dedup_ds.keys())

def randomize_and_write_list(ds, ds_fp, shuffle_cnt = 3):
    for i in range(shuffle_cnt):
        logger.info("Shuffling round %d ..." %i)
        random.shuffle(ds)

    logger.info("Writing to file %s ..."%ds_fp)
    with tqdm(total = len(ds)) as pbar:
        with open(ds_fp, 'w') as fo:
            for pair in ds:
                fo.write(pair)
                #fo.write('\n')
                pbar.update(1)
    logger.info("Done writing")
    return
    
def shuffle_file(file_in, file_out):
    lines = open(file_in).readlines()
    random.shuffle(lines)
    open(file_out, 'w').writelines(lines)
    return
    
def randomize_and_write_ds(ds, ds_fp, shuffle_cnt = 3):
    """Input: ds is a list with these fields
                jacc, aui1, aui2, and label
    """
    
    for i in range(shuffle_cnt):
        logger.info("Shuffling round %d ..." %i)
        random.shuffle(ds)
        
    logger.info("Writing to file %s ..."%ds_fp)
    with tqdm(total=len(ds)) as pbar:
        with open(ds_fp, 'w') as fo:
            for pair in ds:
                fo.write(str(pair["jacc"]) + FLAGS.delimiter)
                fo.write(pair["AUI1"] + FLAGS.delimiter)
                fo.write(pair["AUI2"] + FLAGS.delimiter + pair["Label"] + "\n")
                pbar.update(1)
    logger.info("Done writing")
    return
        
def randomize_and_write_file(ds_fp_in, ds_fp_out, shuffle_cnt = 3):
    """Input: ds is a list with these fields
                jacc, aui1, aui2, and label
    """
    ds = []
    logger.info("Reading file ..." %ds_fp_in)
    with tqdm(total = utils.count_lines(ds_fp_in)) as pbar:
        with open(ds_fp_in,'r') as fi:
            reader = csv.DictReader(fi,fieldnames=["jacc", "AUI1", "AUI2", "Label"], delimiter=FLAGS.delimiter)
            with tqdm(total = utils.count_lines(ds_fp)) as pbar:
                for line in reader:
                    pbar.update(1)
                    ds.append(line)

    for i in range(shuffle_cnt):
        logger.info("Shuffling round %d ..." %i)
        random.shuffle(ds)
        
    logger.info("Writing to file %s ..."%ds_fp)    
    with tqdm(total=len(ds)) as pbar:
        with open(ds_fp_out, 'w') as fo:
            for pair in ds:
                fo.write(pair["jacc"] + FLAGS.delimiter)
                fo.write(pair["AUI1"] + FLAGS.delimiter)
                fo.write(pair["AUI2"] + FLAGS.delimiter + pair["Label"] + "\n")
    
    logger.info("Done writing.")
        
def concat_files(glob_fp, out_fp):
    with open(out_fp, 'wb') as outfile:
        for filename in glob_fp:
            with open(filename, 'rb') as readfile:
                shutil.copyfileobj(readfile, outfile)
    return
   
def generate_dataset(paths, fold=1):
    """ The input files have the pairs already randomized
    """
    # Collecting pos_pairs 
    pos_pairs = read_file_to_list(paths['pos_pairs_fp'])
    
    # Split to train, test, dev sets
    logger.info("Spliting the pos ...")
    pos_dev_pairs, pos_test_pairs, pos_train_pairs = split_ds(pos_pairs, fold)
    del pos_pairs

    all_neg_dev_pairs = [] 
    all_neg_test_pairs = []
    all_neg_train_pairs = []
    all_neg_gentest_pairs = []
    # Collecting neg_pairs from different flavors
    for flavor in paths['neg_pairs_training_flavor_fp'].keys():
    
        # Generating the datasets
        if flavor != FLAGS.neg_pairs_flavor_all:
            
            # Collecting data from files in glob
            logger.info("Concatenating files from %s to %s"%(paths['neg_pairs_training_flavor_glob'][flavor], paths['neg_pairs_training_flavor_fp'][flavor]))
            concat_files(glob.glob(paths['neg_pairs_training_flavor_glob'][flavor]), paths['neg_pairs_training_flavor_fp'][flavor])
            # Read the data neg_fp
            neg_pairs_training = read_file_to_list(paths['neg_pairs_training_flavor_fp'][flavor])
            # Generating the val and test sets for training
            logger.info("Spliting the neg ...")
            neg_dev_pairs, neg_test_pairs, neg_train_pairs = split_ds(neg_pairs_training, fold)
            if flavor != FLAGS.neg_pairs_flavor_ran:
                all_neg_train_pairs += neg_train_pairs
                all_neg_test_pairs += neg_test_pairs
                all_neg_dev_pairs += neg_dev_pairs
    
            train_fp = os.path.join(paths['ds_training_flavor_dp'][flavor], "%s_%s_%s"%(FLAGS.training_type, flavor, FLAGS.train_fn))
            logger.info("Randomizing and writing the %s ..."%train_fp)
            randomize_and_write_list(pos_train_pairs + neg_train_pairs, train_fp)
            logger.info("Deleting variables for neg_train %s"%flavor)
            del neg_train_pairs

            dev_fp = os.path.join(paths['ds_training_flavor_dp'][flavor], "%s_%s_%s"%(FLAGS.training_type, flavor, FLAGS.dev_fn))
            logger.info("Randomizing and writing the %s ..."%dev_fp)
            randomize_and_write_list(pos_dev_pairs + neg_dev_pairs, dev_fp)
            logger.info("Deleting variables for neg_dev %s"%flavor)
            del neg_dev_pairs

            test_fp = os.path.join(paths['ds_training_flavor_dp'][flavor], "%s_%s_%s"%(FLAGS.training_type, flavor, FLAGS.test_fn))
            logger.info("Randomizing and writing the %s ..."%test_fp)
            write_list_to_file(pos_test_pairs + neg_test_pairs, test_fp)
            logger.info("Deleting variables for neg_test %s"%flavor)
            del neg_test_pairs
            # Leave for the pos_test_pairs for joining with other tests below
            
            gentest_fp = os.path.join(paths['ds_gentest_flavor_dp'], "%s_%s_%s"%(FLAGS.gentest_type, flavor, FLAGS.test_fn))
            logger.info("Generating GENTEST file for %s in %s"%(flavor, gentest_fp))
            concat_files(glob.glob(paths['neg_pairs_gentest_flavor_glob'][flavor]), paths['neg_pairs_gentest_flavor_fp'][flavor])
            neg_pairs_gentest = read_file_to_list(paths['neg_pairs_gentest_flavor_fp'][flavor])
            if flavor != FLAGS.neg_pairs_flavor_ran:
                all_neg_gentest_pairs += neg_pairs_gentest
            
            write_list_to_file(pos_test_pairs + neg_pairs_gentest, gentest_fp)
            logger.info("Deleting variables for neg_gentest %s"%flavor)
            del neg_pairs_gentest
            
        else:
            train_fp = os.path.join(paths['ds_training_flavor_dp'][flavor], "%s_%s_%s"%(FLAGS.training_type, flavor, FLAGS.train_fn))
            logger.info("Randomizing and writing the %s ..."%train_fp)
            randomize_and_write_list(pos_train_pairs + dedup_list(all_neg_train_pairs), train_fp)
            logger.info("Deleting variables for all_neg_train %s"%flavor)
            del all_neg_train_pairs

            dev_fp = os.path.join(paths['ds_training_flavor_dp'][flavor], "%s_%s_%s"%(FLAGS.training_type, flavor, FLAGS.dev_fn))
            logger.info("Randomizing and writing the %s ..."%dev_fp)
            randomize_and_write_list(pos_dev_pairs + dedup_list(all_neg_dev_pairs), dev_fp)
            logger.info("Deleting variables for all_neg_dev %s"%flavor)
            del all_neg_dev_pairs

            test_fp = os.path.join(paths['ds_training_flavor_dp'][flavor], "%s_%s_%s"%(FLAGS.training_type, flavor, FLAGS.test_fn))
            logger.info("Randomizing and writing the %s ..."%test_fp)
            write_list_to_file(pos_test_pairs + dedup_list(all_neg_test_pairs), test_fp)
            logger.info("Deleting variables for all_neg_test %s"%flavor)
            del all_neg_test_pairs
            # Leave for the pos_test_pairs for joining with other tests below
            
            gentest_fp = os.path.join(paths['ds_gentest_flavor_dp'], "%s_%s_%s"%(FLAGS.gentest_type, flavor, FLAGS.test_fn))
            logger.info("Generating GENTEST file for %s in %s"%(flavor, gentest_fp))
            write_list_to_file(pos_test_pairs + dedup_list(all_neg_gentest_pairs), gentest_fp)
            logger.info("Deleting variables for all_neg_gentest %s"%flavor)
            del all_neg_gentest_pairs

    logger.info("Deleting variables pos_test_pairs")
    del pos_train_pairs
    del pos_dev_pairs
    del pos_test_pairs
    logger.info("GC ollecting ... ")
    gc.collect()
    logger.info("Done.")
    return
    
    
def split_ds(pairs, fold=1):
    # Split the neg_pairs
    cnt = int(len(pairs)/5)
    
    dev_pairs = pairs[0:cnt] 
    test_pairs = pairs[cnt+1:cnt*2]
    train_pairs = pairs[cnt*2:len(pairs)]
        
    return dev_pairs, test_pairs, train_pairs
    
def generate_pos_pairs(paths):
    aui_info, id_aui_dict = get_aui_info(paths['mrconso_master_fp'])
    cui_aui_dict = {}
    logger.info("Loading %s "%paths['mrconso_master_fp'])
    with open(paths['mrconso_master_fp'],'r') as fi:
        reader = csv.DictReader(fi, fieldnames = FLAGS.mrconso_master_fields, delimiter=FLAGS.delimiter,doublequote=False,quoting=csv.QUOTE_NONE)
        with tqdm(total = utils.count_lines(paths['mrconso_master_fp'])) as pbar:
            for line in reader:
                pbar.update(1)
                if line["CUI"] in cui_aui_dict:
                    if line["AUI"] not in cui_aui_dict[line["CUI"]]:
                        cui_aui_dict[line["CUI"]].append(line["AUI"])
                else:
                    cui_aui_dict[line["CUI"]] = [line["AUI"]]
    label = '1'
    pos_pairs = []
    logger.info("Computing jaccard for each pos pair")
    with tqdm(total=len(cui_aui_dict)) as pbar:
        for cui in cui_aui_dict:
            pbar.update(1)
            lst_auis = cui_aui_dict[cui]
            for i in range(len(lst_auis)):
                for j in range(len(lst_auis)-(i+1)):
                    jacc = compute_jaccard(aui_info[lst_auis[i]]["NS_ID"],aui_info[lst_auis[i+j+1]]["NS_ID"])
                    pos_pairs.append({'jacc':jacc,'AUI1': lst_auis[i],'AUI2':lst_auis[i+j+1],'Label': label})
    
    logger.info("Randomizing and writing to %s file"%paths['pos_pairs_fp'])
    randomize_and_write_ds(pos_pairs, paths['pos_pairs_fp'])
    
    logger.info("Deleting pos_pairs ...")
    del pos_pairs
    logger.info("GC collecting ... ")
    gc.collect()
    logger.info("Done generating pos pairs.")
    return


def get_dataset_dn():
    dataset_dn = FLAGS.dataset_dn if FLAGS.dataset_dn is not None else os.path.basename(FLAGS.umls_version_dp)
    return dataset_dn

        
def main(_):
    global utils
    utils = Utils()
    paths = dict()
    # Local folder, create if not existing
    paths['log_dp'] = os.path.join(FLAGS.workspace_dp, FLAGS.log_dn)
    Path(paths['log_dp']).mkdir(parents=True, exist_ok=True)
    paths['data_gen_log_dp'] = os.path.join(FLAGS.workspace_dp, FLAGS.log_dn, FLAGS.job_name)
    Path(paths['data_gen_log_dp']).mkdir(parents=True, exist_ok=True)

    paths['extra_dp'] = os.path.join(FLAGS.workspace_dp, FLAGS.extra_dn)
    Path(paths['extra_dp']).mkdir(parents=True, exist_ok=True)
    # File path to SemGroups.txt
    paths['sg_st_fp'] = os.path.join(paths['extra_dp'], FLAGS.sg_st_fn)

    # For the executable files
    paths['bin_dp'] = os.path.join(FLAGS.workspace_dp, FLAGS.bin_dn)
    Path(paths['bin_dp']).mkdir(parents=True, exist_ok=True)
    
    # Paths to METATHESAURUS filfes
    paths['umls_meta_dp'] = FLAGS.umls_meta_dp if FLAGS.umls_meta_dp is not None else os.path.join(FLAGS.umls_version_dp, FLAGS.umls_meta_dn)
    
    paths['mrxnw_fp'] = os.path.join(paths['umls_meta_dp'], FLAGS.mrxnw_fn)
    paths['mrxns_fp'] = os.path.join(paths['umls_meta_dp'], FLAGS.mrxns_fn)
    paths['mrconso_fp'] = os.path.join(paths['umls_meta_dp'], FLAGS.mrconso_fn)
    paths['cui_sty_fp'] = os.path.join(paths['umls_meta_dp'], FLAGS.cui_sty_fn)

    
    paths['datasets_dp'] = FLAGS.datasets_dp if FLAGS.datasets_dp is not None else os.path.join(FLAGS.workspace_dp, FLAGS.datasets_dn)
    Path(paths['datasets_dp']).mkdir(parents=True, exist_ok=True)

    # For the dataset files
    paths['dataset_dn'] = get_dataset_dn()
    paths['dataset_dp'] = os.path.join(paths['datasets_dp'], paths['dataset_dn'])
    Path(paths['dataset_dp']).mkdir(parents=True, exist_ok=True)

    paths['data_generator_fp'] = os.path.join(paths['bin_dp'], FLAGS.data_generator_fn)
    paths['bin_umls_version_dp'] = os.path.join(paths['bin_dp'], paths['dataset_dn'])
    Path(paths['bin_umls_version_dp']).mkdir(parents=True, exist_ok=True)
    paths['swarm_fp'] = os.path.join(paths['bin_umls_version_dp'], "%s_%s"%(FLAGS.dataset_version_dn, FLAGS.swarm_fn))
    paths['submit_gen_neg_pairs_jobs_fp'] = os.path.join(paths['bin_umls_version_dp'], "%s_%s"%(FLAGS.dataset_version_dn, FLAGS.submit_gen_neg_pairs_jobs_fn))
    

    # File paths to program data files
    paths['umls_dl_dp'] = FLAGS.umls_dl_dp if FLAGS.umls_dl_dp is not None else os.path.join(FLAGS.umls_version_dp, FLAGS.umls_dl_dn)
    Path(paths['umls_dl_dp']).mkdir(parents=True, exist_ok=True)

    paths['mrx_nw_id_fp'] = os.path.join(paths['umls_dl_dp'], FLAGS.mrx_nw_id_fn)
    paths['mrx_ns_id_fp'] = os.path.join(paths['umls_dl_dp'], FLAGS.mrx_ns_id_fn)
    paths['nw_id_aui_fp'] = os.path.join(paths['umls_dl_dp'], FLAGS.nw_id_aui_fn)    
    paths['mrconso_master_fp'] = os.path.join(paths['umls_dl_dp'], FLAGS.mrconso_master_fn)  
    paths['mrconso_master_randomized_fp'] = os.path.join(paths['umls_dl_dp'], FLAGS.mrconso_master_randomized_fn)  
    paths['aui_info_gen_neg_pairs_pickle_fp'] = os.path.join(paths['umls_dl_dp'], FLAGS.aui_info_gen_neg_pairs_pickle_fn)
    paths['cui_to_aui_id_pickle_fp'] = os.path.join(paths['umls_dl_dp'], FLAGS.cui_to_aui_id_pickle_fn)
    paths['inputs_pickle_fp'] = os.path.join(paths['umls_dl_dp'], FLAGS.inputs_pickle_fn)

    paths['pos_pairs_fp'] = os.path.join(paths['umls_dl_dp'], FLAGS.pos_pairs_fn)

    paths['dataset_version_dp'] = os.path.join(FLAGS.umls_version_dp, FLAGS.dataset_version_dn)
    Path(paths['dataset_version_dp']).mkdir(parents=True, exist_ok=True)

    paths['neg_files_dp'] = os.path.join(paths['dataset_version_dp'], FLAGS.neg_file_prefix)
    Path(paths['neg_files_dp']).mkdir(parents=True, exist_ok=True)
    paths['neg_batch_files_dp'] = os.path.join(paths['dataset_version_dp'], FLAGS.neg_batch_file_prefix)
    Path(paths['neg_batch_files_dp']).mkdir(parents=True, exist_ok=True)
        
    paths['neg_pairs_flavors'] = [
        FLAGS.neg_pairs_flavor_topn_sim,
        FLAGS.neg_pairs_flavor_ran_sim,
        FLAGS.neg_pairs_flavor_ran_nosim,
        FLAGS.neg_pairs_flavor_all,
        FLAGS.neg_pairs_flavor_ran,
    ]
    # ========= FOR NEG BATCH ================
    paths['neg_pairs_training_flavor_batch_fp'] = dict()
    paths['neg_pairs_gentest_flavor_batch_fp'] = dict()
    
    for flavor in paths['neg_pairs_flavors']:
        paths['neg_pairs_training_flavor_batch_fp'][flavor] = os.path.join(paths['neg_batch_files_dp'], "%s_%s_%s_%d_%d.RRF"%(FLAGS.training_type,flavor, FLAGS.neg_batch_file_prefix, FLAGS.start_idx, FLAGS.end_idx)) 
        paths['neg_pairs_gentest_flavor_batch_fp'][flavor] = os.path.join(paths['neg_batch_files_dp'], "%s_%s_%s_%d_%d.RRF"%(FLAGS.gentest_type, flavor, FLAGS.neg_batch_file_prefix, FLAGS.start_idx, FLAGS.end_idx)) 

    # ====== FOR NEG GLOB ==================
    paths['neg_pairs_training_flavor_glob'] = {flavor:os.path.join(paths['neg_batch_files_dp'], "%s_%s_%s*"%(FLAGS.training_type, flavor, FLAGS.neg_batch_file_prefix)) for flavor in paths['neg_pairs_flavors']}
    paths['neg_pairs_gentest_flavor_glob'] = {flavor:os.path.join(paths['neg_batch_files_dp'], "%s_%s_%s*"%(FLAGS.gentest_type, flavor, FLAGS.neg_batch_file_prefix)) for flavor in paths['neg_pairs_flavors']}

    # ====== FOR COMPLETED_AUIS ==================
    paths['completed_inputs_fp'] = os.path.join(paths['neg_batch_files_dp'], FLAGS.completed_inputs_fn)
    
    # ========= FOR NEG FINAL FILES COLLECTED FROM BATCHES ================    
    paths['neg_pairs_training_flavor_fp'] = dict()
    paths['neg_pairs_gentest_flavor_fp'] = dict()
    for flavor in paths['neg_pairs_flavors']:
        paths['neg_pairs_training_flavor_fp'][flavor] = os.path.join(paths['neg_files_dp'], "%s_%s_%s.RRF"%(FLAGS.training_type, flavor, FLAGS.neg_file_prefix))
        paths['neg_pairs_gentest_flavor_fp'][flavor] = os.path.join(paths['neg_files_dp'], "%s_%s_%s.RRF"%(FLAGS.gentest_type, flavor, FLAGS.neg_file_prefix))
    
    # FOR THE FINAL DATASETS
    paths['ds_training_flavor_dp'] = dict()
    for flavor in paths['neg_pairs_flavors']:
        paths['ds_training_flavor_dp'][flavor] = os.path.join(paths['dataset_version_dp'], FLAGS.training_type, flavor)
        Path(paths['ds_training_flavor_dp'][flavor]).mkdir(parents=True, exist_ok=True)
        
    paths['ds_gentest_flavor_dp'] = os.path.join(paths['dataset_version_dp'], FLAGS.gentest_type)
    Path(paths['ds_gentest_flavor_dp']).mkdir(parents=True, exist_ok=True)
    
    # Logging
    paths['log_filepath'] = os.path.join(paths['log_dp'],"%s.log"%(FLAGS.application_name))
    global logger 
    logger = utils.get_logger(logging.DEBUG, FLAGS.application_name, paths['log_filepath'])
    utils.set_logger(logger)

    submit_parameters = [
        " -b 1",
        " --merge-output"
        " -g " + str(FLAGS.ram),
        " -t " + str(FLAGS.n_processes),
        " --time 2-00:00:00 --logdir %s"%(os.path.join(FLAGS.workspace_dp, FLAGS.log_dn, FLAGS.job_name)),
    ]
    #prepy_cmds = ['source /data/nguyenvt2/libs/miniconda3/etc/profile.d/conda.sh',
    #              'conda activate %s'%FLAGS.conda_env]
    prepy_cmds = []
    paths['execute_py_fp'] = os.path.join(paths['bin_dp'], FLAGS.application_py_fn)

    if FLAGS.gen_master_file:
        generate_master_file(paths)
                
    if FLAGS.gen_pos_pairs:
        logger.info("Generating %s from %s"%(paths['mrconso_master_fp'], paths['pos_pairs_fp']))
        generate_pos_pairs(paths)
    
    if FLAGS.gen_neg_pairs:
        logger.info("Generating neg pairs files from %s ... "%(paths['mrconso_master_fp']))
        start_time = time.time()

        generate_negative_pairs(FLAGS.job_name, paths, prepy_cmds, submit_parameters)
        end_time = time.time()
        logger.info("Generating neg pairs in %d sec."%(end_time - start_time))

    if FLAGS.gen_dataset:
        generate_dataset(paths)    

    if FLAGS.gen_neg_pairs_batch:
        generate_negative_pairs_node_batch(paths, FLAGS.start_idx, FLAGS.end_idx, FLAGS.neg_to_pos_rate, FLAGS.n_processes)
        
    logger.info("Finished.")

if __name__ == '__main__':
    app.run(main)
