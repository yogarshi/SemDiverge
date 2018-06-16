
# Path to VDPWI -- SET THIS
vdpwi_dir=../VDPWI-NN-TORCH

# Paths to embeddings -- SET THIS
embeds_dir=
embeds_pre=
source_embeds=${embeds_dir}/${embeds_pre}.en
target_embeds=${embeds_dir}/${embeds_pre}.fr

# Paths to training and test data - SET THESE
data_dir=../data/

train_dir=${data_dir}/train_dir
dev_dir=${data_dir}/dev_dir
test_dir=${data_dir}/test_dir

source_train=${train_dir}/all.tok.lc.en
target_train=${train_dir}/all.tok.lc.fr

source_dev=${test_dir}/all.tok.lc.en
target_dev=${test_dir}/all.tok.lc.fr

source_test=${test_dir}/all.tok.lc.en
target_test=${test_dir}/all.tok.lc.fr


# Preprocess embeddings

echo "Reformating embeddings"
python add_suffix.py --input ${source_embeds} --suffix "@en@" --output ${source_embeds}.prep --only_first
python add_suffix.py --input ${target_embeds} --suffix "@fr@" --output ${target_embeds}.prep --only_first
cat ${source_embeds}.prep ${target_embeds}.prep > ${embeds_dir}/${embeds_pre}.all
rm ${source_embeds}.prep ${target_embeds}.prep

echo "Converting embeddings to torch readable format"
th ${vdpwi_dir}/scripts/convert-wordvecs.lua ${embeds_dir}/${embeds_pre}.all ${data_dir}/${embeds_pre}.vocab ${data_dir}/${embeds_pre}.th


# Preprocess training and test data
python add_suffix.py --input ${source_train} --suffix "@en@" --output ${train_dir}/a.toks
python add_suffix.py --input ${target_train} --suffix "@fr@" --output ${train_dir}/b.toks
lines=`wc -l ${train_dir}/a.toks | cut -d' ' -f1`
echo ${lines}
rm ${train_dir}/id.txt
seq $lines >> ${train_dir}/id.txt
sed 's/1/5/g' ${train_dir}/labels > ${train_dir}/sim.txt

python add_suffix.py --input ${source_dev} --suffix "@en@" --output ${test_dir}/a.toks
python add_suffix.py --input ${target_dev} --suffix "@fr@" --output ${test_dir}/b.toks
lines=`wc -l ${dev_dir}/a.toks | cut -d' ' -f1`
rm ${dev_dir}/id.txt
seq $lines >> ${dev_dir}/id.txt
sed 's/1/5/g' ${dev_dir}/labels > ${dev_dir}/sim.txt

python add_suffix.py --input ${source_test} --suffix "@en@" --output ${test_dir}/a.toks
python add_suffix.py --input ${target_test} --suffix "@fr@" --output ${test_dir}/b.toks
lines=`wc -l ${test_dir}/a.toks | cut -d' ' -f1`
rm ${test_dir}/id.txt
seq $lines >> ${test_dir}/id.txt
sed 's/1/5/g' ${test_dir}/labels > ${test_dir}/sim.txt

python ${vdpwi_dir}/scripts/build_vocab.py ${data_dir}

ln -s ${train_dir} ${data_dir}/train
ln -s ${dev_dir} ${data_dir}/dev
ln -s ${test_dir} ${data_dir}/test

