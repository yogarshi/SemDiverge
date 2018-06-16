#PBS -l pmem=10g
#PBS -m abe
#PBS -q batch
#PBS -l walltime=24:00:00
#PBS -l cput=48:00:00
set -x


# Paramters to be set

training_dir=../data/train_dir
test_dir=../data/test_dir
random_seed=1
# Ratio of positive to negative examples - 5 is a good idea
ratio=5


# Create dictionaries from word aligned parallel data
python create_dict.py	--src_sents_path ../data/dict_data/fr-en.fr.gz \
						--trg_sents_path ../data/dict_data/fr-en.en.gz \
						--al_path ../data/dict_data/fr-en.al.gz \
						--output_path_s2t ../data/dict_data/dict.f2e \
						--output_path_t2s ../data/dict_data/dict.e2f


cp ../data/dict_data/dict.f2e "${training_dir}"
cp ../data/dict_data/dict.e2f "${training_dir}"

# Create negative examples
python create_negative_examples.py --train_dir "${training_dir}" --test_dir "${test_dir}" --seed ${random_seed} --ratio $ratio



