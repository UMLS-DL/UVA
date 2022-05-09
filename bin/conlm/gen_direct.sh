awk -F'|' '{print $5"|"$7}' ../MRCONSO_MASTER.RRF > AUI_SCUI_direct_MRCONSO_MASTER_META_DL.RRF
awk -F'|' '{print $2"|"$4}' ../../META/MRHIER.RRF > AUI_PAUI_direct_MRHIER.RRF
awk -F'|' '{print $2"|"$7}' ../MRCONSO_MASTER.RRF > CUI_SCUI_direct_MRCONSO.RRF
awk -F'|' '{print $1"|"$4}' ../../META/MRSTY.RRF > CUI_STY_direct_MRSTY.RRF
