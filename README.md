# UMLS Vocabulary Alignment

## Download folder
Three UVA datasets 2020AA, 2021AA, and 2021AB are available for download.
The official link with UMLS authentication from the NLM is under processing. For now, these datasets can be downloaded at [UVA](https://drive.google.com/drive/folders/1P72Q2FNo4MKEgIBVv2lGQinJzwHF2cuG?usp=sharing).

```
$ cd $WORKSPACE/UMLS_VERSIONS/download
$ ls
2020AA-ACTIVE.tar.gz
2021AA-ACTIVE.tar.gz
2021AB-ACTIVE.tar.gz
biowordvec.txt 
$ cp biowordvec.txt $WORKSPACE/extra/
$ mkdir 2021AA-ACTIVE
$ mkdir 2021AA-ACTIVE/NEGPOS1
$ cp 2021AA-ACTIVE.tar.gz 2021AA-ACTIVE/NEGPOS1/
$ cd 2021AA-ACTIVE/NEGPOS1/
$ tar -xzvf 2021AA-ACTIVE.tar.gz
$ ls 2021AA-ACTIVE/
2021AA-ACTIVE/GENTEST_DS_RAN_SIM_TEST_DS.RRF
2021AA-ACTIVE/LEARNING_DS_ALL_TRAIN_DS.RRF
2021AA-ACTIVE/LEARNING_DS_ALL_TEST_DS.RRF
2021AA-ACTIVE/GENTEST_DS_ALL_TEST_DS.RRF
2021AA-ACTIVE/LEARNING_DS_ALL_DEV_DS.RRF
2021AA-ACTIVE/GENTEST_DS_RAN_NOSIM_TEST_DS.RRF
2021AA-ACTIVE/GENTEST_DS_TOPN_SIM_TEST_DS.RRF
```


## How to generate UVA dataset?

### Requirements
Step 1: install a version of UMLS, for example, 2021AA-ACTIVE. UMLS can be downloaded at [https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html]
During the installation process, active subset of vocabularies can be selected.

Step 2: after the installation process finishes, copy the .RRF files in the resulting META directory into $WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/META.

```
$ cd $WORKSPACE/UMLS_VERSIONS/
$ ls
2020AA-ACTIVE 2021AA-ACTIVE 2021AB-ACTIVE
$ cd 2021AA-ACTIVE
$ ls
META
$ cd META
MRCONSO.RRF  MRREL.RRF  MRSTY.RRF  MRXNS_ENG.RRF  MRXNW_ENG.RRF
# The above .RRF files are mandatory for the UVA project.
```
###  Generating a UVA dataset
Below is the command for generating the 2021AA-ACTIVE dataset using 499 nodes with 20 threads and 180GB of RAM in each node.

```
$ cat $WORKSPACE/bin/datagen/biowulf_data_generator_2021AA.sh
#!/bin/bash
# Generating datasets
WORKSPACE=/data/Bodenreider_UMLS_DL/UVA; \
UMLS_VERSION=2021AA-ACTIVE; \
python $WORKSPACE/bin/run_data_generator.py \
--job_name=21AA_GEN --server=Biowulf \
--workspace_dp=$WORKSPACE \
--umls_version_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION \
--umls_dl_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/META_DL \
--gen_master_file=true \
--gen_pos_pairs=true \
--gen_swarm_file=true \
--exec_gen_neg_pairs_swarm=true \
--gen_neg_pairs=true \
--gen_dataset=true \
--run_slurm_job=true \
--ntasks=499 \
--n_processes=20 \
--ram=180 \
--dataset_version_dn=NEGPOS1 \
--neg_to_pos_rate=1 \
--debug=false
```
Below is the sample code to deploy the above job to NIH Biowulf HPC
```
swarm -f $WORKSPACE/bin/datagen/biowulf_data_generator_2021AA.sh 
-b 1 -g 240 -t 12 --time 2-20:00:00 \
--logdir $WORKSPACE/logs
```

## References
1. Vinh Nguyen, Olivier Bodenreider. _UVA Resources for Biomedical Vocabulary Alignment at Scale in the UMLS Metathesaurus_. 2022. Submitted to ISWC. 2022
2. Vinh Nguyen, Hong Yung Yip, Goonmeet Bajaj, Thilini Wijesiriwardene, Vishesh Javangula, Srinivasan Parthasarathy, Amit Sheth, Olivier Bodenreider. _Context-Enriched Learning Models for Aligning Biomedical Vocabularies at Scale in the UMLS Metathesaurus_. Proceedings of the Web Conference 2022 (WWW'22). ACM. 2022
3. Vinh Nguyen, Hong Yung Yip, Olivier Bodenreider. Biomedical Vocabulary Alignment at Scale in the UMLS Metathesaurus. Proceedings of the Web Conference 2021 (WWW'21). ACM. 2021
4. Vinh Nguyen, Olivier Bodenreider. Adding an Attention Layer Improves the Performance of a Neural Network Architecture for Synonymy Prediction in the UMLS Metathesaurus. Proceedings of the MedInfo Conference 2021.
5. Goonmeet Bajaj, Vinh Nguyen, Thilini Wijesiriwardene, Hong Yung Yip, Vishesh Javangula, Srinivasan Parthasarathy, Amit Sheth, Olivier Bodenreider. Evaluating Biomedical BERT Models for Vocabulary Alignment at Scale in the UMLS Metathesaurus. arXiv preprint arXiv:2109.13348. 2021
6. Thilini Wijesiriwardene, Vinh Nguyen, Goonmeet Bajaj, Hong Yung Yip, Vishesh Javangula, Yuqing Mao, Kin Wah Fung, Srinivasan Parthasarathy, Amit P Sheth, Olivier Bodenreider. UBERT: A Novel Language Model for Synonymy Prediction at Scale in the UMLS Metathesaurus. arXiv preprint arXiv:2204.12716. 2021


