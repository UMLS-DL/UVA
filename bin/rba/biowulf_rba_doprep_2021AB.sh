#!/bin/sh
source /data/nguyenvt2/libs/miniconda3/etc/profile.d/conda.sh; \
conda activate tf_uva; \
WORKSPACE=/data/Bodenreider_UMLS_DL/UVA; \
UMLS_VERSION=2021AB-ACTIVE; \
python $WORKSPACE/bin/run_rba_evaluator.py \
--workspace_dp=$WORKSPACE \
--umls_dp=$WORKSPACE/UMLS_VERSIONS \
--umls_dl_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/META_DL \
--test_dataset_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/NEGPOS1/GENTEST_DS  \
--rba_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/RBA \
--do_prep=true \
--do_compute_rule_closure_in_batch=false \
--do_eval=false \
--closure_n_processes=16 \
--ntasks=200 \
--conda_env=tf230_uva \
--ram=120 \
--job_name=21AB_BMC \
--run_slurm_job=true \
--debug=false \
--regenerate_files=true


