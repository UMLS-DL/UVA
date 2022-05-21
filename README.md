# UMLS Vocabulary Alignment

## Latest Updates
* The [UVA Challenge infopage](https://github.com/UMLS-DL/UVA/wiki/UVA-Challenge) is created
* The [How to Install UMLS page](https://github.com/UMLS-DL/UVA/wiki/How-to-install-UMLS) is created
* A paper title `UVA Resources for Biomedical Vocabulary Alignment at Scale in the UMLS Metathesaurus` has been submitted to the ISWC Resource track. A copy of the paper submission has been submitted to the arxiv and will be available soon.

## Requirements
Step 1: install a version of UMLS (e.g., 2022AA-ACTIVE) by following this [tutorial](https://github.com/UMLS-DL/UVA/wiki/How-to-install-UMLS). UMLS can be downloaded at the [UMLS Download](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html).

During the installation process, active subset of vocabularies can be selected.

Step 2: after the installation process finishes, copy the .RRF files in the resulting META directory into $WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/META.

<pre>
$ cd $WORKSPACE/UMLS_VERSIONS/
$ ls
2020AA-ACTIVE 2021AA-ACTIVE 2021AB-ACTIVE
$ cd 2021AA-ACTIVE
$ ls
META
$ cd META
<b>MRCONSO.RRF  MRREL.RRF  MRSTY.RRF  MRXNS_ENG.RRF  MRXNW_ENG.RRF</b>
</pre>
The above .RRF files are **mandatory** for the UVA project.

Step 3: install two conda profiles: `uva_kge` and `tf_uva`
```
$ sh $WORKSPACE/bin/install_pykeen_env.sh
$ sh $WORKSPACE/bin/install_tf230_uva.sh
```

## Download folder for existing UVA datasets
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
2021AA-ACTIVE/GENTEST_DS_RAN_SIM_TEST_DS.RRF
2021AA-ACTIVE/LEARNING_DS_ALL_TRAIN_DS.RRF
2021AA-ACTIVE/LEARNING_DS_ALL_TEST_DS.RRF
2021AA-ACTIVE/GENTEST_DS_ALL_TEST_DS.RRF
2021AA-ACTIVE/LEARNING_DS_ALL_DEV_DS.RRF
2021AA-ACTIVE/GENTEST_DS_RAN_NOSIM_TEST_DS.RRF
2021AA-ACTIVE/GENTEST_DS_TOPN_SIM_TEST_DS.RRF
$ ls 2021AA-ACTIVE/
```

If using a downloaded UVA dataset, a MRCONSO_MASTER.RRF file needs to be generated for the baselines as follows.
<pre>
#!/bin/bash
# Generating datasets
WORKSPACE=/data/Bodenreider_UMLS_DL/UVA; \
UMLS_VERSION=2021AA-ACTIVE; \
python $WORKSPACE/bin/run_data_generator.py \
--job_name=21AA_GEN --server=Biowulf \
--workspace_dp=$WORKSPACE \
--umls_version_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION \
--umls_dl_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/META_DL \
<b>--gen_master_file=true </b> \
<b>--gen_pos_pairs=false</b> \
<b>--gen_swarm_file=false</b> \
<b>--exec_gen_neg_pairs_swarm=false</b> \
<b>--gen_neg_pairs=false</b> \
<b>--gen_dataset=false</b> \
--run_slurm_job=false \
--ntasks=499 \
--n_processes=20 \
--ram=180 \
--dataset_version_dn=NEGPOS1 \
--neg_to_pos_rate=1 \
--debug=false
</pre>
The resulting MRCONSO_MASTER.RRF file is located in the $WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/META_DL


## How to generate a new UVA dataset?
Below is the command for generating the 2021AA-ACTIVE dataset using 499 nodes with 20 threads and 180GB of RAM in each node.

<pre>
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
<b>--gen_master_file=true</b> \
<b>--gen_pos_pairs=true</b> \
<b>--gen_swarm_file=true</b> \
<b>--exec_gen_neg_pairs_swarm=true</b> \
<b>--gen_neg_pairs=true</b> \
<b>--gen_dataset=true</b> \
--run_slurm_job=true \
--ntasks=499 \
--n_processes=20 \
--ram=180 \
<b>--dataset_version_dn=NEGPOS1</b> \
--neg_to_pos_rate=1 \
--debug=false
</pre>
The resulting dataset version is located inside the $WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/NEGPOS1 with two folders GENTEST_DS and LEARNING_DS.

Below is the sample code to deploy the above data generator job to NIH Biowulf HPC
```
$ cat $WORKSPACE/bin/datagen/biowulf_data_generator_2021AA.swarm
swarm -f $WORKSPACE/bin/datagen/biowulf_data_generator_2021AA.sh 
-b 1 -g 240 -t 12 --time 2-20:00:00 \
--logdir $WORKSPACE/logs
```

## How to train a LexLM model using a UVA dataset?
Below are the commands for preparing, training, and testing a LexLM model to be executed in a sequence. The paths need to be adjusted according to the project setting if using the downloaded dataset path parameter $train_dataset_fp instead of $train_dataset_dp.

The training and testing job was deployed to a server with 240 GB of system RAM and a Tesla V100X 32G GPU. The batch size may be reduced or increased based on the GPU RAM.

<pre>
WORKSPACE=/data/Bodenreider_UMLS_DL/UVA; \
UMLS_VERSION=2021AA-ACTIVE; \
python $WORKSPACE/bin/run_umls_classifier.py \
--workspace_dp=$WORKSPACE \
--umls_version_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION \
--umls_dl_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/META_DL \
--dataset_version_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/NEGPOS1 \
--train_dataset_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/NEGPOS1/LEARNING_DS/ALL \
--run_id="lexlm_uva_run1"  \
--n_epoch=100  \
--batch_size=8192  \
<b>--do_train=false</b> \
<b>--do_predict=false</b> \
<b>--do_prep=true</b>  \
--generator_workers=16
</pre>

<pre>
conda activate tf_uva; \
WORKSPACE=/data/Bodenreider_UMLS_DL/UVA; \
UMLS_VERSION=2021AA-ACTIVE; \
python  $WORKSPACE/bin/run_umls_classifier.py \
--workspace_dp=$WORKSPACE \
--umls_version_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION \
--umls_dl_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/META_DL \
--dataset_version_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/NEGPOS1 \
--train_dataset_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/NEGPOS1/LEARNING_DS/ALL \
--run_id="lexlm_uva_run1_2021AA-ACTIVE"  \
--exp_flavor=1 \
--n_epoch=100  \
--batch_size=8192  \
<b>--do_prep=false</b>  \
<b>--do_train=true</b> \
<b>--do_predict=false</b>  \
--generator_workers=8
</pre>

<pre>
WORKSPACE=/data/Bodenreider_UMLS_DL/UVA; \
UMLS_VERSION=2021AA-ACTIVE; \
python  $WORKSPACE/bin/run_umls_classifier.py \
--workspace_dp=$WORKSPACE \
--umls_version_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION \
--umls_dl_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/META_DL \
--dataset_version_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/NEGPOS1 \
--train_dataset_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/NEGPOS1/LEARNING_DS/ALL \
--test_dataset_fp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/NEGPOS1/GENTEST_DS/GENTEST_DS_ALL_TEST_DS.RRF \
--run_id="lexlm_uva_run1_$UMLS_VERSION"  \
--exp_flavor=1 \
--n_epoch=100  \
--batch_size=8192  \
<b>--do_prep=false</b>  \
<b>--do_train=false</b> \
<b>--do_predict=true</b>  \
--start_epoch_predict=1 \
--end_epoch_predict=100 \
--generator_workers=8 
</pre>
## How to train a ConLM model using a UVA dataset?
Below are the commands for preparing, training, and testing a ConLM model to be executed in a sequence. The paths need to be adjusted according to the project setting if using the downloaded dataset path parameter $train_dataset_fp instead of $train_dataset_dp.

The training and testing job was deployed to a server with 240 GB of system RAM and a Tesla V100X 32G GPU. The batch size may be reduced or increased based on the GPU RAM.

<pre>
$ cd $WORKSPACE/bin/conlm
conda activate uva_kge; \
WORKSPACE=/data/Bodenreider_UMLS_DL/UVA; \
UMLS_VERSION=2021AA-ACTIVE; \
VARIANT="All_Triples"; \
MODEL="ComplEx" ; \
EPOCHS=100 ; \
OPTIMIZER="Adam" ; \
KGE_HOME=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/META_DL/CONLM; \
mkdir -p $KGE_HOME; \
cp $WORKSPACE/bin/conlm/UMLS_Parser.py $WORKSPACE/bin/conlm/gen_dataset.sh $WORKSPACE/bin/conlm/gen_direct.sh $WORKSPACE/bin/conlm/SemGroups.txt $KGE_HOME/ ; \
cd $KGE_HOME; \
sh $KEG_HOME/gen_dataset.sh; 
</pre>
<pre>
python $WORKSPACE/bin/run_train_kge.py \
--root_dir=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/META_DL/CONLM \
--kge_model=$MODEL \
--optimizer=$OPTIMIZER \
--kge_triple_variants=$VARIANT \
--lr=0.01 \
--embedding_dim=25 \
--loss="marginranking" \
--margin=1 \
--training_loop="slcwa" \
--num_epochs=100 \
--batch_size=1024 \
--evaluator="rankbased" \
--negative_sampler="basic" \
--num_negs_per_pos=50 \
--create_inverse_triples=False \
--enable_eval_filtering=True; 
</pre>
<pre>
python $WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/META_DL/CONLM/UMLS_Parser.py \
--KGE_Home=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/META_DL/CONLM \
--meta_dl_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/META_DL \
--Task="gen_aui2convec" \
--Model=$MODEL \
--Optimizer=$OPTIMIZER \
--Variant=$VARIANT \
--context_dim=50 \
--num_epochs=$EPOCHS; 
</pre>
<pre>
conda activate tf_uva; \
python $WORKSPACE/bin/run_umls_classifier.py \
--workspace_dp=$WORKSPACE \
--KGE_Home=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/META_DL/CONLM \
--umls_version_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION \
--umls_dl_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/META_DL \
--dataset_version_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/NEGPOS1 \
--train_dataset_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/NEGPOS1/LEARNING_DS/ALL \
<b>--exp_flavor=2</b> \
--context_vector_dim=150 \
--ConVariant=$VARIANT \
--Model=$MODEL \
--ConOptimizer=$OPTIMIZER \
--ConEpochs=$EPOCHS \
--do_prep=False \
--run_id="conlm_uva_run1_$UMLS_VERSION" \
--n_epoch=$EPOCHS \
--batch_size=8192 \
<b>--do_train=True</b> \
<b>--do_predict=False</b> \
--generator_workers=8 \
</pre>
<pre>
conda activate tf_uva ; \
python $WORKSPACE/bin/run_umls_classifier.py \
--workspace_dp=$WORKSPACE \
--KGE_Home=$KGE_HOME \
--umls_version_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION \
--umls_dl_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/META_DL \
--dataset_version_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/NEGPOS1 \
--test_dataset_fp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/NEGPOS1/GENTEST_DS/GENTEST_DS_ALL_TEST_DS.RRF \
--train_dataset_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/NEGPOS1/LEARNING_DS/ALL \
<b>--exp_flavor=2</b> \
--context_vector_dim=150 \
--ConVariant=$VARIANT \
--Model=$MODEL \
--ConOptimizer=$OPTIMIZER \
--ConEpochs=$EPOCHS \
--do_prep=False \
--run_id="conlm_uva_run1_$UMLS_VERSION" \
--n_epoch=$EPOCHS \
--batch_size=8192 \
<b>--do_train=False</b> \
<b>--do_predict=True</b> \
--generator_workers=8 \
--start_epoch_predict=1 \
--end_epoch_predict=100
</pre>
The resulting training performance and checkpoints of each run is located inside the `TRAINING` folder of each $UMLS_VERSION.

## References

[1] Vinh Nguyen, Olivier Bodenreider. _UVA Resources for Biomedical Vocabulary Alignment at Scale in the UMLS Metathesaurus_. 2022. Submitted to ISWC. 2022.

[2] Vinh Nguyen, Hong Yung Yip, Goonmeet Bajaj, Thilini Wijesiriwardene, Vishesh Javangula, Srinivasan Parthasarathy, Amit Sheth, Olivier Bodenreider. _Context-Enriched Learning Models for Aligning Biomedical Vocabularies at Scale in the UMLS Metathesaurus_. Proceedings of the Web Conference 2022 (WWW'22). ACM. 2022.

[3] Vinh Nguyen, Hong Yung Yip, Olivier Bodenreider. _Biomedical Vocabulary Alignment at Scale in the UMLS Metathesaurus_. Proceedings of the Web Conference 2021 (WWW'21). ACM. 2021

[4] Vinh Nguyen, Olivier Bodenreider. _Adding an Attention Layer Improves the Performance of a Neural Network Architecture for Synonymy Prediction in the UMLS Metathesaurus_. Proceedings of the MedInfo Conference 2021.

[5] Bajaj, Goonmeet, Vinh Nguyen, Thilini Wijesiriwardene, Hong Yung Yip, Vishesh Javangula, Amit Sheth, Srinivasan Parthasarathy, and Olivier Bodenreider. _Evaluating Biomedical Word Embeddings for Vocabulary Alignment at Scale in the UMLS Metathesaurus Using Siamese Networks_. In Proceedings of the Third Workshop on Insights from Negative Results in NLP, pp. 82-87. 2022.

[6] Goonmeet Bajaj, Vinh Nguyen, Thilini Wijesiriwardene, Hong Yung Yip, Vishesh Javangula, Srinivasan Parthasarathy, Amit Sheth, Olivier Bodenreider. _Evaluating Biomedical BERT Models for Vocabulary Alignment at Scale in the UMLS Metathesaurus_. arXiv preprint arXiv:2109.13348. 2021

[7] Thilini Wijesiriwardene, Vinh Nguyen, Goonmeet Bajaj, Hong Yung Yip, Vishesh Javangula, Yuqing Mao, Kin Wah Fung, Srinivasan Parthasarathy, Amit P Sheth, Olivier Bodenreider. _UBERT: A Novel Language Model for Synonymy Prediction at Scale in the UMLS Metathesaurus_. arXiv preprint arXiv:2204.12716. 2021


## Contact
Any issue found while running the code can be posted in the [Issues](https://github.com/UMLS-DL/UVA/issues).

Any question about the UVA project, please email Vinh Nguyen at vinh.nguyen@nih.gov.
