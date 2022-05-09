echo "Generating Intermediatary and Training Files..."
sh gen_direct.sh
python UMLS_Parser.py --Task="gen_dataset" --KGE_Home="."

echo "Making Triple directories..."
mkdir -p All_Triples AUI_SCUI_Triples SCUI_ParentSCUI_Triples SCUI_SG_Triples

echo "Copying files to $TRAINING_DATA Triples directories..."
cp entity2id.txt relation2id.txt ./All_Triples 
cp entity2id.txt relation2id.txt ./AUI_SCUI_Triples 
cp entity2id.txt relation2id.txt ./AUI_SCUI_ParentSCUI_Triples 
cp entity2id.txt relation2id.txt ./SCUI_ParentSCUI_Triples 
cp entity2id.txt relation2id.txt ./SCUI_SG_Triples
cp All_Triples_Train.txt ./All_Triples
cp AUI_SCUI_Triples_Train.txt ./AUI_SCUI_Triples
cp SCUI_ParentSCUI_Triples_Train.txt ./SCUI_ParentSCUI_Triples
cp SCUI_SG_Triples_Train.txt ./SCUI_SG_Triples

mv ./All_Triples/All_Triples_Train.txt ./All_Triples/train2id.txt
mv ./AUI_SCUI_Triples/AUI_SCUI_Triples_Train.txt ./AUI_SCUI_Triples/train2id.txt
mv ./SCUI_ParentSCUI_Triples/SCUI_ParentSCUI_Triples_Train.txt ./SCUI_ParentSCUI_Triples/train2id.txt
mv ./SCUI_SG_Triples/SCUI_SG_Triples_Train.txt ./SCUI_SG_Triples/train2id.txt
