source /data/nguyenvt2/libs/miniconda3/etc/profile.d/conda.sh
conda create -n tf_uva python=3.7 -y
conda activate tf_uva
conda install cudatoolkit=10.1 -y
conda install cudnn=7.6.5 -y
yes | pip install tensorflow-gpu==2.3.0
conda install numpy -y
conda install absl-py -y
conda install tqdm -y
conda install pandas -y
conda install scikit-learn -y
conda install matplotlib -y

