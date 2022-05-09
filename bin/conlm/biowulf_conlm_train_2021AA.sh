source /data/nguyenvt2/libs/miniconda3/etc/profile.d/conda.sh; \
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
conda activate tf230_uva; \
python $WORKSPACE/bin/run_umls_classifier.py \
--workspace_dp=$WORKSPACE \
--KGE_Home=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/META_DL/CONLM \
--umls_version_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION \
--umls_dl_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/META_DL \
--dataset_version_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/NEGPOS1 \
--train_dataset_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/NEGPOS1/LEARNING_DS/ALL \
--exp_flavor=2 \
--context_vector_dim=150 \
--ConVariant=$VARIANT \
--Model=$MODEL \
--ConOptimizer=$OPTIMIZER \
--ConEpochs=$EPOCHS \
--do_prep=False \
--run_id="conlm_uva_run1_$UMLS_VERSION" \
--n_epoch=$EPOCHS \
--batch_size=8192 \
--do_train=True \
--do_predict=False \
--generator_workers=8 \
--logs_fp="$WORKSPACE/logs/train_conlm_uva_run1_"$UMLS_VERSION"_MODEL_LEARNING_DS_ALL_8192b_100e_exp2_All_Triples_TransE_SGD_100_150_uva_final_run1.log" 
##sh $KEG_HOME/gen_dataset.sh; \
##python $WORKSPACE/bin/run_train_kge.py \
##--root_dir=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/META_DL/CONLM \
##--kge_model=$MODEL \
##--optimizer=$OPTIMIZER \
##--kge_triple_variants=$VARIANT \
##--lr=0.01 \
##--embedding_dim=25 \
##--loss="marginranking" \
##--margin=1 \
##--training_loop="slcwa" \
##--num_epochs=100 \
##--batch_size=1024 \
##--evaluator="rankbased" \
##--negative_sampler="basic" \
##--num_negs_per_pos=50 \
##--create_inverse_triples=False \
##--enable_eval_filtering=True; \
##python $WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/META_DL/CONLM/UMLS_Parser.py \
##--KGE_Home=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/META_DL/CONLM \
##--meta_dl_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/META_DL \
##--Task="gen_aui2convec" \
##--Model=$MODEL \
##--Optimizer=$OPTIMIZER \
##--Variant=$VARIANT \
##--context_dim=50 \
##--num_epochs=$EPOCHS; 
