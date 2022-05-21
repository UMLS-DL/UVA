source /data/nguyenvt2/libs/miniconda3/etc/profile.d/conda.sh; \
conda activate tf230_uva; \
WORKSPACE=/data/Bodenreider_UMLS_DL/UVA; \
UMLS_VERSION=2021AB-ACTIVE; \
python  $WORKSPACE/bin/run_umls_classifier.py \
--workspace_dp=$WORKSPACE \
--umls_version_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION \
--umls_dl_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/META_DL \
--dataset_version_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/NEGPOS1 \
--train_dataset_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/NEGPOS1/LEARNING_DS/ALL \
--run_id="lexlm_uva_run1_2021AB-ACTIVE"  \
--exp_flavor=1 \
--n_epoch=100  \
--batch_size=8192  \
--do_prep=false  \
--do_train=true \
--do_predict=false  \
--generator_workers=8 > $WORKSPACE/logs/train_$UMLS_VERSION_MODEL_LEARNING_DS_ALL_8192b_100e_exp1_nodropout_run1.log

