__author__ = 'yogarshi'
import argparse
import cPickle as pickle
import gzip
import sys
from collections import Counter


def main(argv):
	parser = argparse.ArgumentParser(description="")
	parser.add_argument("--src_sents_path", help='')
	parser.add_argument("--trg_sents_path", help='')
	parser.add_argument("--al_path", help='')
	parser.add_argument("--output_path_s2t", help='')
	parser.add_argument("--output_path_t2s", help='')

	args = parser.parse_args(args=argv)
	trans_dict_s2t = {}
	trans_dict_t2s = {}
	count = 0
	# Build the dictionary as a {string : Counter}
	with gzip.open(args.src_sents_path) as f_src, \
			gzip.open(args.trg_sents_path) as f_trg, \
			gzip.open(args.al_path) as f_al:
		sys.stdout.write("Building dictionaries")
		for en_line in f_src:
			count += 1
			if count % 100000 == 0:
				sys.stdout.flush()
				sys.stdout.write(".")
			src_line = en_line.strip().split()
			trg_line = f_trg.readline().strip().split()
			al_line = f_al.readline().strip().split()

			for each_alignment in al_line:
				src_pos, trg_pos = each_alignment.strip().split('-')
				src_word = src_line[int(src_pos)]
				trg_word = trg_line[int(trg_pos)]
				if src_word not in trans_dict_s2t:
					trans_dict_s2t[src_word] = Counter()
				trans_dict_s2t[src_word][trg_word] += 1
				if trg_word not in trans_dict_t2s:
					trans_dict_t2s[trg_word] = Counter()
				trans_dict_t2s[trg_word][src_word] += 1

	pickle.dump(trans_dict_s2t, open(args.output_path_s2t,'w'))
	pickle.dump(trans_dict_t2s, open(args.output_path_t2s,'w'))

	with open(args.output_path_s2t + ".txt", 'w') as fout:
		for each_key in trans_dict_s2t:
			fout.write(each_key)
			fout.write("\t")
			fout.write(", ".join(trans_dict_s2t[each_key]))
			fout.write("\n")
	with open(args.output_path_t2s + ".txt", 'w') as fout:
		for each_key in trans_dict_t2s:
			fout.write(each_key)
			fout.write("\t")
			fout.write(", ".join(trans_dict_t2s[each_key]))
			fout.write("\n")



if __name__ == "__main__":
	main(sys.argv[1:])
