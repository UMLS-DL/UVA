source /data/nguyenvt2/libs/miniconda3/etc/profile.d/conda.sh; \
conda activate tf230_uva; \
WORKSPACE=/data/Bodenreider_UMLS_DL/UVA; \
UMLS_VERSION=2021AA-ACTIVE; \
python  $WORKSPACE/bin/run_umls_classifier.py \
--workspace_dp=$WORKSPACE \
--umls_version_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION \
--umls_dl_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/META_DL \
--dataset_version_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/NEGPOS1 \
--train_dataset_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/NEGPOS1/LEARNING_DS/ALL \
--test_dataset_fp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/NEGPOS1/GENTEST_DS/GENTEST_DS_ALL_TEST_DS.RRF \
--test_dataset_fp_rba_predictions=/data/nguyenvt2/aaai2020/RBA/GENTEST_DS/2020AA-ACTIVE_ALL_MODEL_GENTEST_DS_TEST_DS.RRF_predictions.PICKLE_SCUI_LS_SG_TRANS \
--run_id="lexlm_uva_run1_$UMLS_VERSION"  \
--exp_flavor=1 \
--n_epoch=100  \
--batch_size=8192  \
--do_prep=false  \
--do_train=false \
--do_predict=true  \
--start_epoch_predict=100 \
--end_epoch_predict=100 \
--generator_workers=8 > $WORKSPACE/logs/predict_gentest_$UMLS_VERSION_MODEL_LEARNING_DS_ALL_8192b_100e_exp1_nodropout_run1.log; \
python  $WORKSPACE/bin/run_umls_classifier.py \
--workspace_dp=$WORKSPACE \
--umls_version_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION \
--umls_dl_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/META_DL \
--dataset_version_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/NEGPOS1 \
--train_dataset_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/NEGPOS1/LEARNING_DS/ALL \
--test_dataset_fp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/NEGPOS1/GENTEST_DS/GENTEST_DS_TOPN_SIM_TEST_DS.RRF \
--test_dataset_fp_rba_predictions=/data/nguyenvt2/aaai2020/RBA/GENTEST_DS/2020AA-ACTIVE_ALL_MODEL_GENTEST_DS_TEST_DS.RRF_predictions.PICKLE_SCUI_LS_SG_TRANS \
--run_id="lexlm_uva_run1_$UMLS_VERSION"  \
--exp_flavor=1 \
--n_epoch=100  \
--batch_size=8192  \
--do_prep=false  \
--do_train=false \
--do_predict=true  \
--start_epoch_predict=99 \
--end_epoch_predict=100 \
--generator_workers=8 > $WORKSPACE/logs/predict_gentest_$UMLS_VERSION_MODEL_LEARNING_DS_TOPN_SIM_8192b_100e_exp1_nodropout_run1.log ; \
python  $WORKSPACE/bin/run_umls_classifier.py \
--workspace_dp=$WORKSPACE \
--umls_version_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION \
--umls_dl_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/META_DL \
--dataset_version_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/NEGPOS1 \
--train_dataset_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/NEGPOS1/LEARNING_DS/ALL \
--test_dataset_fp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/NEGPOS1/GENTEST_DS/GENTEST_DS_RAN_SIM_TEST_DS.RRF \
--test_dataset_fp_rba_predictions=/data/nguyenvt2/aaai2020/RBA/GENTEST_DS/2020AA-ACTIVE_ALL_MODEL_GENTEST_DS_TEST_DS.RRF_predictions.PICKLE_SCUI_LS_SG_TRANS \
--run_id="lexlm_uva_run1_$UMLS_VERSION"  \
--exp_flavor=1 \
--n_epoch=100  \
--batch_size=8192  \
--do_prep=false  \
--do_train=false \
--do_predict=true  \
--start_epoch_predict=99 \
--end_epoch_predict=100 \
--generator_workers=8 > $WORKSPACE/logs/predict_gentest_$UMLS_VERSION_MODEL_LEARNING_DS_RAN_SIM_8192b_100e_exp1_nodropout_run1.log; \
python  $WORKSPACE/bin/run_umls_classifier.py \
--workspace_dp=$WORKSPACE \
--umls_version_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION \
--umls_dl_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/META_DL \
--dataset_version_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/NEGPOS1 \
--train_dataset_dp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/NEGPOS1/LEARNING_DS/ALL \
--test_dataset_fp=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/NEGPOS1/GENTEST_DS/GENTEST_DS_RAN_NOSIM_TEST_DS.RRF \
--test_dataset_fp_rba_predictions=/data/nguyenvt2/aaai2020/RBA/GENTEST_DS/2020AA-ACTIVE_RAN_NOSIM_MODEL_GENTEST_DS_TEST_DS.RRF_predictions.PICKLE_SCUI_LS_SG_TRANS \
--run_id="lexlm_uva_run1_$UMLS_VERSION"  \
--exp_flavor=1 \
--n_epoch=100  \
--batch_size=8192  \
--do_prep=false  \
--do_train=false \
--do_predict=true  \
--start_epoch_predict=99 \
--end_epoch_predict=100 \
--generator_workers=8 > $WORKSPACE/logs/predict_gentest_$UMLS_VERSION_MODEL_LEARNING_DS_ALL_8192b_100e_exp1_nodropout_run1.log

