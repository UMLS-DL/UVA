#!/bin/bash
# Generating datasets
WORKSPACE=/data/Bodenreider_UMLS_DL/UVA ;\
UMLS_VERSION=2021AB-ACTIVE ;\
python $WORKSPACE/bin/run_data_generator.py \
--job_name=21AB_GEN --server=Biowulf \
--workspace_dp=$WORKSPACE \
--umls_version_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION \
--umls_dl_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/META_DL \
--gen_master_file=false \
--gen_pos_pairs=false \
--gen_swarm_file=false \
--exec_gen_neg_pairs_swarm=false \
--gen_neg_pairs=false \
--gen_dataset=true \
--run_slurm_job=true \
--ntasks=499 \
--n_processes=20 \
--ram=180 \
--dataset_version_dn=NEGPOS1 \
--neg_to_pos_rate=1 \
--debug=false

