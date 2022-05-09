from multiprocessing import Manager, Queue, Process, Lock
from pathlib import Path
import subprocess
import logging
import pickle
import os
import inspect
import math
import glob
import time
import random
import csv
import gc
from tqdm import tqdm

class SlurmJob():
    def __init__(self, job_name, run_slurm_job, prepy_cmds, swarm_parameters, submit_parameters, ntasks, ninputs, fp_suffix, input_paras, output_paras, output_globs, paths, job_id = None, max_concurrent = 1000, logger = None):
        self.prepy_cmds = prepy_cmds # source /data/nguyenvt2/libs/miniconda3/etc/profile.d/conda.sh; conda activate {};
                                    # --module=python/3.7
        self.execute_py_fp = paths['execute_py_fp']
        self.swarm_parameters = swarm_parameters
        self.submit_parameters = submit_parameters
        self.bin_dp = paths['bin_dp']
        self.job_name = job_name
        self.run_slurm_job = run_slurm_job
        self.ntasks = ntasks
        self.ninputs = ninputs
        self.fp_suffix = fp_suffix if fp_suffix is not None else ''
        self.input_paras = input_paras
        self.output_paras = output_paras
        self.output_globs = output_globs
        self.max_concurrent = max_concurrent
        self.logger = logger
        self.utils = Utils()
        self.utils.set_logger(logger)
      
        self.swarm_fp = os.path.join(self.bin_dp, self.job_name + '.swarm' + self.fp_suffix)
        self.submit_fp = os.path.join(self.bin_dp, self.job_name + '.submit' + self.fp_suffix)

        self.swarm_files = list()
        self.submit_files = list()
    
        self.start_time = time.time()
        self.time_limit = 10*20*60*60

    def run(self, submit=True):
        'Generate swarm files and submit_job files'
        self.swarm_files, self.submit_jobs_files = self.gen()

        #'Remove existing files'
        #for okey, fp in self.output_paras.items():
            #self.logger.debug("Deleting files %s*"%(fp))
            #clear(fp)
            #self.logger.debug("Deleting files %s*"%(self.output_globs[okey]))
            #clear(self.output_globs[okey])

        'Execute tasks' 
        if (self.run_slurm_job is True):
            if (self.check() is True):
                for f in self.submit_files:
                    self.logger.debug("Excuting %s with slurm"%f)
                    results = self.execute_task(f)
            self.logger.debug("Waiting for the job %s"%self.job_name)
            self.wait(self.job_name, 600)
            
        else:
            # Execute swarm files
            for f in self.swarm_files:
                self.logger.debug("Executing %s with python"%f)
                self.execute_task(f)

    def collect(self):
        'Collect results and pickle them into output_paras' 
        for okey, fp in self.output_paras.items():      
            self.utils.collect(okey, self.output_globs[okey], fp)


    def execute_task(self, fp):
        'Executing the task'
        subprocess.run(['cat','%s'%fp], check=True)
        subprocess.run(['sh','%s'%fp], check=True)
        return 
       
    def execute(self):
        # Executing the task
        files = glob.glob('%s*'%submit_jobs_fp)
        for fp in files:
            subprocess.run(['cat','%s'%fp], check=True)
            subprocess.run(['sh','%s'%fp], check=True)
        return

    def wait(self, job_name, interval = 10, job_id = None):
        
        time.sleep(interval)
        t = time.time()
        while True:
            # Break if this takes longer than time limit
            #if (time.time() - t > time_limit):
            #    self.resume_jobs = True
            #    break
            # Check if the job is done
            if (self.check(job_name, job_id)):
                break
            time.sleep(interval)
        return

    def check(self, job_name = None, job_id=None):
        # Greb the jobs using sjobs
        if job_id is not None:
            chk_cmd = 'sjobs | grep %s | wc -l '%(job_id)
        else:
            if job_name is None:
                job_name = self.job_name
            grep_str = '%s'%(job_name)
            chk_cmd = 'sjobs | grep %s | wc -l '%(grep_str[0:8])

        chk_cmd_response = subprocess.getoutput(chk_cmd)
        self.logger.debug(chk_cmd_response)
        
        if int(chk_cmd_response.strip()) == 0:
            return True
        return False

    def delete(self, swarm_fp = None, submit_fp = None):
        fps = glob.glob(os.path.join(self.bin_dp, self.job_name + "*.swarm"))
        fps += glob.glob(os.path.join(self.bin_dp, self.job_name + "*.sh"))
        for fp in fps:
            f = Path(fp)
            f.unlink()

    def gen(self, swarm_fp = None, submit_fp = None, override = False):
        if len(self.submit_files) > 0:
            if not override:
                return self.submit_files
        batch_size = math.ceil(self.ninputs/self.ntasks)
        cnt = 0
        cur_file = 0
        num_files = self.ntasks/self.max_concurrent

        self.swarm_files = []
        self.submit_files = []
        swarm_fp = swarm_fp if swarm_fp is not None else self.swarm_fp
        submit_fp = submit_fp if submit_fp is not None else self.submit_fp
        for i in range(0, self.ntasks):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size - 1
            if end_idx > self.ninputs:
                end_idx = self.ninputs - 1

            new_swarm_parameters = []
            for idx in range(len(self.swarm_parameters)):
                    new_swarm_parameters.append(self.swarm_parameters[idx].replace('start_index', str(start_idx)).replace('end_index', str(end_idx)))

            if cnt == 0:
                new_swarm_fp = swarm_fp + "_" + str(cur_file)
                self.swarm_files.append(new_swarm_fp)
                new_submit_fp = submit_fp + "_" + str(cur_file)
                self.submit_files.append(new_submit_fp)

                with open(new_swarm_fp,'w') as fo1:
                    fo1.write("#!/bin/bash\n")      
                    # prepy_cmds = ""  
                    #fo1.write("{}; python {} ".format("; ".join(self.prepy_cmds), self.execute_py_fp))
                    fo1.write("python {} ".format(self.execute_py_fp))
                    fo1.write(" ".join(new_swarm_parameters))
                    fo1.write("\n")

                with open(new_submit_fp,'w') as fo2:
                    fo2.write("swarm -f " + new_swarm_fp)
                    fo2.write(" --job-name=%s"%self.job_name)
                    fo2.write(" ".join(self.submit_parameters))
                    fo2.write("\n")
            else:
                with open(new_swarm_fp,'a') as fo3:
                    #prepy_cmds = ""
                    #fo3.write("{}; python {}".format("; ".join(self.prepy_cmds), self.execute_py_fp))
                    fo3.write("python {}".format(self.execute_py_fp))
                    fo3.write(" ".join(new_swarm_parameters))
                    fo3.write("\n")
            cnt += 1
            if cnt == self.max_concurrent:
                cnt = 0
                cur_file += 1
        
        return self.swarm_files, self.submit_files
    

class NodeParallel():
    def __init__(self, worker_target, output_target, num_processes, output_queues_paras, worker_target_kwargs, output_target_kwargs, logger = None):
        self.manager = Manager()
        self.input_queue = self.manager.Queue()
        self.output_queues = dict()
        self.output_queues_paras = self.manager.dict()
        self.being_processed = self.manager.dict()
        self.completed = dict()
        self.processes = list()
        self.worker_id_processes = dict()
        self.worker_processes = list()
        
        self.worker_target = worker_target
        self.output_target = output_target if output_target is not None else self.write
        self.worker_kwargs = worker_target_kwargs
        self.output_kwargs = output_target_kwargs
        self.num_processes = num_processes
        self.ninputs = None 
        self.logger= logger
        self.utils = Utils()
        self.utils.set_logger(logger)
       
        # Initialize the output queue
        for okey, fp in output_queues_paras.items():
            if okey == 'completed':
                self.completed['queue'] = self.manager.Queue()
                self.completed['fp'] = fp
                self.completed['status'] = False
            else:
                self.output_queues[okey] = dict()
                self.output_queues[okey]['queue'] = self.manager.Queue()
                self.output_queues_paras[okey] = self.manager.dict()
                self.output_queues_paras[okey]['fp'] = fp
                self.output_queues_paras[okey]['status'] = False
        self.utils.test_dict(output_queues_paras)
        self.utils.test_dict(self.output_queues_paras, 'init output queues paras')
        self.num_workers = self.num_processes - len(self.output_queues) - 2 # 1 for main, 1 for the manager

    def run(self):
        start_time = time.time()
        self.logger.debug("Starting %d processes ..."%self.num_processes)
        self.start()
    
        self.logger.debug("Processes started computing")
        self.processing()

        self.logger.debug("Processes finished in %d sec."%(time.time() - start_time))
        self.join()
        return
        
    def create(self):
        if len(self.processes) > 0:
            return

        for okey, okey_paras in self.output_queues.items(): 
            p = Process(target = self.output_target, args = (self.num_workers, okey, self.output_queues_paras, self.output_queues[okey],), kwargs = self.output_kwargs)
            self.processes.append(p)
        self.logger.debug("%d writer processes created"%(len(self.output_queues)))                                                                                 
        for idx in range(self.num_workers):
            p = Process(target = self.worker_target, args = (self.input_queue, self.output_queues, self.being_processed, self.completed,), kwargs = self.worker_kwargs)
            self.processes.append(p)
            self.worker_processes.append(p)
        self.logger.debug("%d compute processes created"%(self.num_workers))

    def is_input_done(self):
        return (self.input_queue.qsize() == 0)

    def is_output_done(self):
        for okey in self.output_queues_paras.keys():
            if self.output_queues_paras[okey]['status'] is False:
                return False
        return True

    def is_completed_done(self):
        return self.completed['status']

    def is_being_processed_done(self):
        return (len(self.being_processed) == 0)

    def is_done(self):
        self.logger.debug(self.is_output_done())
        if (self.is_being_processed_done() is False) or (self.is_input_done() is False) or (self.is_output_done() is False) or (self.is_completed_done() is False):
            return False
        return True

    def processing(self):

        self.logger.debug("Start processing ...")
        num_completed = 0
        completed_fo = open(self.completed['fp'],'w')
        completed_fo.close()       
	#self.logger.debug("Remove files %s*"%(para['fp']))
            #clear(para['fp'])
        while (self.is_done() is False):
            for input_id, p_id in self.being_processed.items():
                if self.worker_id_processes[p_id].is_alive() is False:
                    self.input_queue.put(input_id)
                    del self.being_processed[input_id]
                    self.logger.debug("Process %s is terminated by exception while processing %s."%(p_id, input_id))
                    del self.worker_id_processes[p_id]

                    # Add a new process
                    p = Process(target = self.worker_target, args = (self.input_queue, self.output_queues, self.being_processed, self.completed, ), kwargs = self.worker_kwargs)
                    self.processes.append(p)
                    self.worker_processes.append(p)
                    p.start()

                    time.sleep(10)
                    self.worker_id_processes[p.pid] = p
                    self.logger.debug("Adding process %s"%p.pid)
                    
            num_completed = self.track(num_completed)
            self.logger.debug("Inputs completed: %d"%num_completed)
            time.sleep(10)

        completed_fo.close()
        self.logger.debug("Finished with %d inputs"%num_completed)
        return

    def write(self, num_workers, okey, output_queues_paras, output_queue):
        self.logger.debug("Process %s started"%okey)
        output_dict = dict()
        repeated_keys = 0
        num_updates = 0
        fp = output_queues_paras[okey]['fp']
        num_end = 0
        while (True):
            done = False
            while (output_queue['queue'].empty() is False):
                d = output_queue['queue'].get()
                if d == "END":
                    num_end += 1
                    if num_end == num_workers:
                        done = True
                        break
                else:
                    num_updates += 1
                    for k, value in d.items():
                        if k not in output_dict:
                            output_dict[k] = list()
                        else: 
                            repeated_keys += 1
                        output_dict[k].append(value)
                del d
                #gc.collect()
            if done is True:
                break

        self.utils.dump_pickle(output_dict, fp)  
        output_queues_paras[okey]['status'] = True    
        self.logger.debug("Writer %s finished writing %d items (%d repeated, %d updates) from %s workers to %s"%(okey, len(output_dict), repeated_keys, num_updates, num_workers, fp))
        self.utils.test_dict(output_dict, "sample outputs from %s"%(fp))
 
        return

    def track(self, cnt):
        completed_fo = open(self.completed['fp'],'a')
        while ((self.completed['queue'].empty() is False) and (self.completed['status'] is False)):
            info = self.completed['queue'].get()
            if info == 'END':
                break
            completed_fo.write(info) 
            completed_fo.write("\n")                 
            cnt += 1
            del info
        completed_fo.close()
        if cnt == self.ninputs:
            self.completed['status'] = True
        return cnt
   

    def start(self):
        if len(self.processes) == 0:
            self.create()
        for p in self.processes:
            p.start()
        
        for p in self.worker_processes:
            self.worker_id_processes[p.pid] = p
        return
        
    def join(self):
        for p in self.processes:
            p.join()
        return
       
    def set_input_queue(self, input_queue):
        self.ninputs = input_queue.qsize()
        while (input_queue.empty() is False):
            d = input_queue.get()
            self.input_queue.put(d)
        for idx in range(self.num_workers):
            self.input_queue.put("END")

        return

class Utils:

    def __init__(self, logger = None):
        self.logger = logger

    def randomize_keys(self, d_fp):
        d = self.load_pickle(d_fp)
        keys = [k for k in d.keys()]
        random.shuffle(keys)
        random.shuffle(keys)
        random.shuffle(keys)
        self.dump_pickle(keys, d_fp + '_keys')
 
        return

    def merge_clusters(self, ori_fp, fp, final_fp):
        ori_dict = self.load_pickle(ori_fp)
        if (os.path.isfile(final_dict)):
            final_dict = self.load_pickle(final_fp)
        else:
            final_dict = dict()

        merged_dict = self.load_pickle(fp)
        merged_dict_keys = [k for k in merged_dict.keys()]

        new_merged_dict = dict()
        cnt = 0
        while cnt < len(merged_dict_keys):
            k1 = merged_dict_keys[cnt]
            k1_mergeable_k2 = merged_dict[k1]
            if len(k1_mergeable_k2) > 0:
                new_cluster = dict()

                # get all clusters in k1_mergeable_k2 
                for k2 in k1_mergeable_k2:
                    # recursively find all mergeable k2_mergeable_k3
                    new_cluster = new_cluster.union(merged_dict[k2])

                # generate the new key
                new_cluster_ids = list(k1_mergeable_k2.union({k1}))
                new_cluster_ids.sort()
                new_cluster_key = '_'.join(new_cluster_ids)
               
                # update merged_dict with the new key replacing all keys
            
    
                new_merged_dict[new_cluster_key] = new_cluster

            cnt += 1
        self.dump_pickle(new_merged_dict, fp)
        return    

    def process_union(self,fp):
        d = self.load_pickle(fp)
        out = dict()
        for k, lst in d.items():
            union_all = set()
            for cluster in lst: # list of items
                union_all = union_all.union(set(cluster))
            out[k] = union_all

        self.dump_pickle(out, fp)
        del d
        del out
        return

    def collect(self, okey, okey_glob, fp): 
        'Collect list of items from write()'

        #Collect the results from output_paras
        self.logger.debug("Started collecting results for %s from %s"%(okey, fp + '*'))
        output_dict = dict()
        output_files = glob.glob(okey_glob + '*')
        num_updates = 0
        repeated_keys = 0
        for f in output_files:
            d = self.load_pickle(f)
            num_updates += 1
            for k, item_lst in d.items():
                if k not in output_dict:
                    output_dict[k] = list()
                else:
                    repeated_keys += 1
                for item in item_lst:
               	    output_dict[k].append(item)
            del d
        self.logger.debug("Collected %d keys (%d repeated, %d updates) for %s from %s"%(len(output_dict), repeated_keys, num_updates, okey, fp + '*'))
        self.dump_pickle(output_dict, fp)
        del output_dict

        return

    def merge(self, okey, okey_glob, fp):
        'Merge items from glob' 
        self.logger.debug("Started merging results for %s from %s"%(okey, fp + '*'))
        output_dict = dict()
        output_files = glob.glob(okey_glob + '*')
        num_updates = 0
        repeated_keys = 0
        for f in output_files:
            d = self.load_pickle(f)
            num_updates += 1
            for k, item in d.items():
                if k not in output_dict:
                    output_dict[k] = list()
                else:
                    repeated_keys += 1
                output_dict[k].append(item)
            del d
        self.logger.debug("Merged %d keys (%d repeated, %d updates) for %s from %s"%(len(output_dict), repeated_keys, num_updates, okey, fp + '*'))
        self.dump_pickle(output_dict, fp)
        del output_dict

        return

    def read_file_to_id_ds(self, fp, aui2id):
        pickle_fp = fp + '.PICKLE'
        partition = dict()
        if (os.path.isfile(pickle_fp)):
            self.logger.debug("Loading file %s ..."%pickle_fp)
            partition = self.load_pickle(pickle_fp)
            #return partition
        else:
            self.logger.debug("Loading file %s ..."%fp)

            partition = dict()
            with open(fp, 'r') as fi:
                reader = csv.DictReader(fi, fieldnames = ["jacc", "AUI1", "AUI2", "Label"], delimiter = '|')
                with tqdm(total = self.count_lines(fp)) as pbar:
                    for line in reader:
                        pbar.update(1)
                        #if ((aui2id[line['AUI1']] in aui2vec) and (aui2id[line['AUI2']] in aui2vec)):
                        ID = (aui2id[line['AUI1']], aui2id[line['AUI2']])
                        partition[ID] = (float(line['jacc']), int(line['Label']))

            self.dump_pickle(partition, pickle_fp)
        return partition                  

    def compute_scores(self, y_true, y_pred):
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

    def cal_scores(self, TP, TN, FP, FN):
        accuracy = (TP + TN)/(TP + TN + FP + FN)
        precision, recall, f1 = 0, 0, 0

        if TP + FP > 0:
            precision = TP/(TP+FP)

        if TP + FN > 0:
            recall = TP/(TP+FN)

        if recall + precision > 0:
            f1 = 2*(recall * precision) / (recall + precision)

        return round(accuracy,4), round(precision,4), round(recall,4), round(f1,4)

    def clear(self, path):
        fps = glob.glob(path + '*')
        for fp in fps:
            f = Path(fp)
            f.unlink()

    def count_lines(self, filein):
        return sum(1 for line in open(filein))
   
    def set_logger(self, logger):
        self.logger = logger

    def get_logger(self, log_level, name, filepath):
        # get TF logger
    
        log = logging.getLogger(name)
        log.setLevel(log_level)

        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
        ch = logging.StreamHandler()
        ch.setLevel(level=logging.DEBUG)
        ch.setFormatter(formatter)
        
        # create file handler which logs even debug messages
        fh = logging.FileHandler(filepath)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
    
        log.addHandler(ch)
        log.addHandler(fh) 
        return log

    def get_important_logger(self, log_level, name, filepath):
        # get TF logger

        log = logging.getLogger(name)
        log.setLevel(log_level)

        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # create file handler which logs even debug messages
        fh = logging.FileHandler(filepath)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)

        log.addHandler(fh)
        return log

    def dump_pickle(self, obj, pickle_fp):
        self.logger.debug("Dumping pickle at %s"%pickle_fp)
        with open(pickle_fp, 'wb') as f:
            pickle.dump(obj, f, protocol = 4)

        if type(obj) == list:
            self.test_list(obj)
        elif type(obj) == dict:
            self.test_dict(obj)
        return obj

    def load_pickle(self, pickle_fp):
        self.logger.debug("Loading %s"%pickle_fp)
        with open(pickle_fp, 'rb') as f:
            obj = pickle.load(f)
        if type(obj) == list:
            self.test_list(obj)
        elif type(obj) == dict:
            self.test_dict(obj)
        return obj

    def test_big_item(self, d, n):
        cnt = 0
        for k, v in d.items():
            if len(v) > n:
                cnt += 1
                self.logger.debug("Big AUI: item has len > {}: [{}] = {} " %(n, id1, len(id2s)))
        return 

    def test_dict(self, d, dn=None):
        for i, (k,v) in enumerate(d.items()):
            if i < 2:
                self.logger.debug('{}({}): {} -> {}'.format(inspect.stack()[1][3], dn, k, v))
        return

    def test_list(self, l, dn=None):
        for i, v in enumerate(l):
            if i < 2:
                self.logger.debug('{}({}): [{}] = {}'.format(inspect.stack()[1][3], dn, i, v))
        return

    def test_type(self, t, dn=None):
        self.logger.debug('{}({}): type({}) = {}'.format(inspect.stack()[1][3],dn, t, type(t)))
    
    def test_member(self, e, d, dn=None):
        if e not in d:
            self.logger.debug('{}({}): {} not in {}'.format(inspect.stack()[1][3], dn, e, d))
        else:
            self.logger.debug('{}({}): t[{}] = {}'.format(inspect.stack()[1][3], dn, e, d[e]))

    def shuffle_file(self, file_in, file_out):
        lines = open(file_in).readlines()
        random.shuffle(lines)
        open(file_out, 'w').writelines(lines)
        return


