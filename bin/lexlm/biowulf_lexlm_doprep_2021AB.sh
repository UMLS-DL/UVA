WORKSPACE=/data/Bodenreider_UMLS_DL/UVA; \
UMLS_VERSION=2021AB-ACTIVE; \
python $WORKSPACE/bin/run_umls_classifier.py \
--workspace_dp=$WORKSPACE \
--umls_version_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION \
--umls_dl_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/META_DL \
--dataset_version_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/NEGPOS1 \
--train_dataset_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/NEGPOS1/LEARNING_DS/ALL \
--ds_to_pickle_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/NEGPOS1/LEARNING_DS/ALL   \
--run_id="lexlm_uva_run1"  \
--n_epoch=100  \
--batch_size=8192  \
--do_train=false \
--do_predict=false \
--do_prep=true  \
--generator_workers=16 > $WORKSPACE/logs/prep_$UMLS_VERSION_NEGPOS1_LEARNING_DS_ALL_8192b_100e_exp1_nodropout_run1.log

