__author__ = 'yogarshi'
import cPickle as pickle
import random

from utils import *


def word_filter(sp, dict_e2f, dict_f2e, len_ratio=2., trans_ratio=.5):
	"""
	Returns true for a sentence pair if it satisfies certain word overlap
	criteria
	:param sp: 			tuple of sentence pairs
	:param dict_e2f:	Translation dictionary from english to french
	:param dict_f2e:	Translation dictionary from french to english
	:param len_ratio: 	The maximum ratio between the lengths of the two
						sentences
	:param trans_ratio:	The minimum fraction of words that have to have a
						translation in the other sentence
	:return: true if both criteria are satisfied, else false
	"""

	s1, s2 = sp[0].split(), sp[1].split()
	l1, l2 = len(s1), len(s2)

	# Check length ratio
	if l1 > 2 * l2 or l2 > 2 * l1:
		return False

	# Check overlap
	if fraction_of_words(s1, s2, dict_e2f) * 1.0 / l1 < trans_ratio:
		return False

	if fraction_of_words(s2, s1, dict_f2e) * 1.0 / l2 < trans_ratio:
		return False

	return True


def main(argv):
	parser = argparse.ArgumentParser(description="")
	parser.add_argument("--train_dir", help='Folder containing training files')
	parser.add_argument("--test_dir", help="Folder containing test files")
	parser.add_argument("--seed",default=1,type=int)
	parser.add_argument("--ratio", default=5,type=int)
	args = parser.parse_args(args=argv)

	###
	# The training and test directory should contain the following files
	# en_pos.tok.lc :	The english side of the corpus in tokenized lower case
	# 				format
	# fr_pos.tok.lc : 	The french (foreign) side of the corpus in tokenized
	# 				lower case format
	# dict.e2f		: 	Translation dictionary from English to French
	# dict.f2e		: 	Translation dictionary from French to English
	###

	# Set random seed for replicability - 42
	random.seed(args.seed)

	# Load dictionaries
	print "Loading dictionaries.."
	dict_e2f_path = os.path.join(args.train_dir, "dict.e2f")
	dict_f2e_path = os.path.join(args.train_dir, "dict.f2e")
	dict_e2f = pickle.load(open(dict_e2f_path))
	print "Loaded e2f dict with {0} entries".format(len(dict_e2f))
	dict_f2e = pickle.load(open(dict_f2e_path))
	print "Loaded f2e dict with {0} entries".format(len(dict_f2e))

	# Load positive training data
	en_sents = [x.strip().replace('|',' ') for x in
				open(os.path.join(args.train_dir, "en_pos.tok.lc"))]
	fr_sents = [x.strip().replace('|',' ') for x in
				open(os.path.join(args.train_dir, "fr_pos.tok.lc"))]
	assert len(en_sents) == len(fr_sents)
	print "Loaded {0} positive examples".format(len(en_sents))

	# Load test data to ensure no training data leakage
	en_sents_test = [x.strip() for x in
				open(os.path.join(args.test_dir, "all.tok.lc.en"))]
	fr_sents_test = [x.strip() for x in
				open(os.path.join(args.test_dir, "all.tok.lc.fr"))]


	# Create negative examples
	print "Creating negative examples ..."
	positive_pairs_test = zip(en_sents_test,fr_sents_test)
	positive_pairs = [sp for sp in zip(en_sents,fr_sents) if sp not in positive_pairs_test]
	print "{0} training pairs discarded to avoid training set overlap".format(len(positive_pairs) - len(en_sents))
	negative_pairs = set()
	count = 0
	total_negs = 0

	# Hacky way to generate balanced negative examples - sample a positive
	# sentence, a negative sentence. With high probability, we will not have
	# seen this sentence pair earlier. If it satisfies our criteria for a good
	# negative example, add it to our set. If not, discard and repeat.
	while len(negative_pairs) < args.ratio * len(positive_pairs):
		if count % 10000 == 0:
			print count
		count += 1
		sp = (random.choice(en_sents), random.choice(fr_sents))
		if sp in positive_pairs:
			continue
		if not word_filter(sp, dict_e2f, dict_f2e):
			continue
		negative_pairs.add(sp)
	print "Created {0} negative examples from {1} positive examples".format(
		len(negative_pairs), count)

	# if len(negative_pairs) > 5* len(positive_pairs):
	# 	print "Downsampling negative pairs ..."
	# 	negative_pairs = random.sample(negative_pairs,5*len(positive_pairs))

	# Write to file
	with open(os.path.join(args.train_dir, "en_neg.tok.lc"), 'w') as fout1, \
			open(os.path.join(args.train_dir, "fr_neg.tok.lc"), 'w') as fout2:
		for en_sent, fr_sent in negative_pairs:
			fout1.write(en_sent)
			fout1.write("\n")
			fout2.write(fr_sent)
			fout2.write("\n")

	with open(os.path.join(args.train_dir, "en_pos.tok.lc"),
			  'w') as fout1, \
			open(os.path.join(args.train_dir, "fr_pos.tok.lc"),
				 'w') as fout2:
		for en_sent, fr_sent in positive_pairs:
			fout1.write(en_sent)
			fout1.write("\n")
			fout2.write(fr_sent)
			fout2.write("\n")

	# Create labels
	with open(os.path.join(args.train_dir, "labels"), 'w') as fout:
		fout.write('\n'.join(['1'] * len(positive_pairs)))
		fout.write('\n')
		fout.write('\n'.join(['0'] * len(negative_pairs)))


if __name__ == "__main__":
	main(sys.argv[1:])
