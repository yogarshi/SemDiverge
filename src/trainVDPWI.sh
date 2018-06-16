root_dir=`pwd`

vdpwi_dir=/fs/clip-software/user-supported/VDPWI/VDPWI-NN-Torch

cd ${vdpwi_dir}

mkdir saved_model
mkdir predictions

th ${vdpwi_dir}/trainVDPWI.lua	--embeds /fs/clip-xling/CLTE/SemDiverge/MMDataVDPWI/data/bivec/ --prefix bivec \
								--folder /fs/clip-xling/CLTE/SemDiverge/MMDataVDPWI/data/fr-en/  --epochs 40 

cd ${root_dir}
