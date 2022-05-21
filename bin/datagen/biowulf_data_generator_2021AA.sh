#!/bin/bash
# Generating datasets
WORKSPACE=/data/Bodenreider_UMLS_DL/UVA; \
UMLS_VERSION=2021AA-ACTIVE; \
python $WORKSPACE/bin/run_data_generator.py \
--job_name=21AA_GEN --server=Biowulf \
--workspace_dp=$WORKSPACE \
--umls_version_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION \
--umls_dl_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/META_DL \
--gen_master_file=false \
--gen_pos_pairs=false \
--gen_swarm_file=true \
--exec_gen_neg_pairs_swarm=true \
--gen_neg_pairs=true \
--gen_dataset=true \
--run_slurm_job=true \
--ntasks=499 \
--n_processes=20 \
--ram=180 \
--dataset_version_dn=NEGPOS1_run2 \
--neg_to_pos_rate=1 \
--debug=false

