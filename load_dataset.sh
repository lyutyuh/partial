# Based on https://github.com/nikitakit/tetra-tagging/blob/master/examples/training.ipynb

%%bash
if [ ! -e self-attentive-parser ]; then
  git clone https://github.com/nikitakit/self-attentive-parser &> /dev/null
fi
rm -rf train dev test EVALB/
cp self-attentive-parser/data/02-21.10way.clean ./data/train
cp self-attentive-parser/data/22.auto.clean ./data/dev
cp self-attentive-parser/data/23.auto.clean ./data/test
# The evalb program needs to be compiled
cp -R self-attentive-parser/EVALB EVALB
rm -rf self-attentive-parser
cd EVALB && make &> /dev/null
# To test that everything works as intended, we check that the F1 score when
# comparing the dev set with itself is 100.
./evalb -p nk.prm ../data/dev ../data/dev | grep FMeasure | head -n 1


# Download the dependency data
# 1. The SPRML dataset
wget https://polybox.ethz.ch/index.php/s/Qf4lCxxuNotEhg7/download -O data/spmrl.tar.gz
# Run the script to extract the data
tar -xvf data/spmrl.tar.gz -C data/spmrl
./data/spmrl/SPMRL_SHARED_2014_NO_ARABIC/extract_all.sh

# 2. The PTB-dependency dataset
wget https://polybox.ethz.ch/index.php/s/oDhHVx6uZU89ySX/download -O data/dependency_ptb.tar.gz
tar -xvf data/dependency_ptb.tar.gz -C data/
