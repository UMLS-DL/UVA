export WORKSPACE=/data/Bodenreider_UMLS_DL/UVA
export UMLS_VERSION=2021AA-ACTIVE
echo "Generating Intermediatary and Training Files..."
python UMLS_Parser.py --Task="gen_dataset" --KGE_Home=$WORKSPACE/UMLS_VERSIONS/$UMLS_VERSION/META_DL/CONLM

export TRAINING_DATA_DIR=2-BYOD_UMLS
echo "Making Training Data directory..."
mkdir $TRAINING_DATA_DIR

echo "CD into $TRAINING_DATA_DIR directory..."
cd $TRAINING_DATA_DIR

echo "Making Triple directories..."
mkdir All_Triples AUI_SCUI_Triples AUI_SCUI_ParentSCUI_Triples SCUI_ParentSCUI_Triples SCUI_SG_Triples
cd ..

echo "Copying files to $TRAINING_DATA_DIR Triples directories..."
cp entity2id.txt relation2id.txt $TRAINING_DATA_DIR/All_Triples 
cp entity2id.txt relation2id.txt $TRAINING_DATA_DIR/AUI_SCUI_Triples 
cp entity2id.txt relation2id.txt $TRAINING_DATA_DIR/AUI_SCUI_ParentSCUI_Triples 
cp entity2id.txt relation2id.txt $TRAINING_DATA_DIR/SCUI_ParentSCUI_Triples 
cp entity2id.txt relation2id.txt $TRAINING_DATA_DIR/SCUI_SG_Triples
cp All_Triples.txt $TRAINING_DATA_DIR/All_Triples
cp AUI_SCUI_Triples.txt $TRAINING_DATA_DIR/AUI_SCUI_Triples
cp AUI_SCUI_ParentSCUI_Triples.txt $TRAINING_DATA_DIR/AUI_SCUI_ParentSCUI_Triples
cp SCUI_ParentSCUI_Triples.txt $TRAINING_DATA_DIR/SCUI_ParentSCUI_Triples
cp SCUI_SG_Triples.txt $TRAINING_DATA_DIR/SCUI_SG_Triples

mv $TRAINING_DATA_DIR/All_Triples/All_Triples.txt $TRAINING_DATA_DIR/All_Triples/train2id.txt
mv $TRAINING_DATA_DIR/AUI_SCUI_Triples/AUI_SCUI_Triples.txt $TRAINING_DATA_DIR/AUI_SCUI_Triples/train2id.txt
mv $TRAINING_DATA_DIR/AUI_SCUI_ParentSCUI_Triples/AUI_SCUI_ParentSCUI_Triples.txt $TRAINING_DATA_DIR/AUI_SCUI_ParentSCUI_Triples/train2id.txt
mv $TRAINING_DATA_DIR/SCUI_ParentSCUI_Triples/SCUI_ParentSCUI_Triples.txt $TRAINING_DATA_DIR/SCUI_ParentSCUI_Triples/train2id.txt
mv $TRAINING_DATA_DIR/SCUI_SG_Triples/SCUI_SG_Triples.txt $TRAINING_DATA_DIR/SCUI_SG_Triples/train2id.txt

echo "Cleaning up current directory..."
mv SemGroups.txt SemGroups.RRF
rm -rf *.txt
mv SemGroups.RRF SemGroups.txt

rm -rf ../$TRANING_DATA_DIR
mv $TRANING_DATA_DIR ..
rm -rf $TRAINING_DATA_DIR
echo "All done!"
