#!/bin/bash

date

## experiment naming variables -- SET THESE 
base_dir=.
exp_dir=$base_dir/exp
lang_in=en
lang_out=fr

## experiment files --
data_dir=data
train=OpenSubtitles2016.ALL.en-fr.tok.tc.bpe.en
tune=
test=


gpu_n=0
avg_metric=perplexity # "perplexity",accuracy,bleu
avg_n=8
beam=5


## software bits
softw_path=/fs/clip-software/user-supported
moses_scripts_path=$softw_path/mosesdecoder/3.0/scripts

## Training
if [ ! -d $exp_dir/model ]; then
	echo "    ** Training"
	python3 -m sockeye.train \
		-s $data_dir/$train.$lang_in \
		-t $data_dir/$train.$lang_out \
		-vs $data_dir/$tune.$lang_in \
		-vt $data_dir/$tune.$lang_out \
		-o $exp_dir/model \
		--num-words 50000:50000 \
		--encoder rnn \
		--decoder rnn \
		--num-layers 1:1 \
		--rnn-cell-type gru \
		--rnn-num-hidden 1000 \
		--num-embed 512:512 \
		--rnn-attention-type mlp \
		--max-seq-len 50:50 \
		--batch-size 80 \
		--checkpoint-frequency 35000 \
		--max-num-checkpoint-not-improved 8 \
		--monitor-bleu 1000 \
		--device-ids $gpu_n \
		--seed 1234 \
		--disable-device-locking
fi;

decode_dir=$exp_dir/decode.$avg_metric.$avg_n.$beam
mkdir -p $decode_dir

if [ ! -f $decode_dir/params.avg ]; then
	echo "    ** Averaging parameters"
	python3 -m sockeye.average \
		-n $avg_n \
		--metric $avg_metric \
		--strategy best \
		--output $decode_dir/params.avg \
		$exp_dir/model
fi;
ln -sf $decode_dir/params.avg $exp_dir/model/params.best

## Decoding
if [ ! -f $decode_dir/$test.tok.tc.bpe.trans ]; then
	echo "    ** Decoding"
	python3 -m sockeye.translate \
		--beam-size $beam         \
		--device-ids -$gpu_n   \
		--models $exp_dir/model \
		--input $data_dir/$test.tok.tc.bpe.$lang_in \
		--output $decode_dir/$test.tok.tc.bpe.trans
fi;

## evaluation
if [ ! -f $decode_dir/bleu.log- ]; then
	echo "=== Evaluation with tokenization, truecasing and BPE encoding ===" >> $decode_dir/bleu.log
	$moses_scripts_path/generic/multi-bleu.perl \
		-lc $data_dir/$test.tok.tc.bpe.$lang_out \
		< $decode_dir/$test.tok.tc.bpe.trans      \
		>> $decode_dir/bleu.log

	echo "=== Evaluation with original segmentation ===" >> $decode_dir/bleu.log
	cat $decode_dir/$test.tok.tc.bpe.trans  \
		| sed -r 's/(@@ )|(@@ ?$)//g'        \
		| $moses_scripts_path/recaser/detruecase.perl   \
		| $moses_scripts_path/tokenizer/detokenizer.perl \
			-l $lang_out \
		> $decode_dir/$test.trans
	$moses_scripts_path/generic/multi-bleu.perl \
		-lc $data_dir/$test.$lang_out \
		< $decode_dir/$test.trans      \
		>> $decode_dir/bleu.log

	cat $decode_dir/bleu.log
fi;

date
