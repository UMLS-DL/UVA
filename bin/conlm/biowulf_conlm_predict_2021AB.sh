source /data/nguyenvt2/libs/miniconda3/etc/profile.d/conda.sh; \
WORKSPACE=/data/Bodenreider_UMLS_DL/UVA; \
UMLS_VERSION=2021AB-ACTIVE; \
VARIANT="All_Triples"; \
MODEL="ComplEx"; \
EPOCHS=100; \
OPTIMIZER="Adam"; \
KGE_HOME=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/META_DL/CONLM; \
conda activate tf230_uva; \
python $WORKSPACE/bin/run_umls_classifier.py \
--workspace_dp=$WORKSPACE \
--KGE_Home=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/META_DL/CONLM \
--umls_version_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION \
--umls_dl_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/META_DL \
--dataset_version_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/NEGPOS1 \
--test_dataset_fp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/NEGPOS1/GENTEST_DS/GENTEST_DS_ALL_TEST_DS.RRF \
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
--do_train=False \
--do_predict=True \
--generator_workers=8 \
--start_epoch_predict=1 \
--end_epoch_predict=100 \
--logs_fp="$WORKSPACE/logs/train_conlm_uva_run1_"$UMLS_VERSION"_MODEL_LEARNING_DS_ALL_8192b_100e_exp2_All_Triples_TransE_SGD_100_150_uva_final_run1.log"
