source /data/nguyenvt2/libs/miniconda3/etc/profile.d/conda.sh
conda create -n uva_kge python=3.7 -y
conda activate uva_kge
conda install cudatoolkit=10.0 -y
conda install cudnn=7.3.1 -y
yes | pip install tensorflow-gpu==1.15
conda install numpy -y
conda install absl-py -y
yes | pip install pykeen==1.5.0
