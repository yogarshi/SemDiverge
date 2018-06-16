__author__ = 'yogarshi'
import argparse
import sys

def add_to_all(args):
	with open(args.output, 'w') as fout, open(args.input) as fin:
		for each_line in fin:
			x = each_line.strip().split()
			fout.write(' '.join(map(lambda p: p + args.suffix, x)))
			fout.write('\n')

def add_to_first(args):
	with open(args.output, 'w') as fout, open(args.input) as fin:
		x = fin.readline()
		for each_line in fin:
			x = each_line.strip().split()
			fout.write(' '.join([x[0] + args.suffix] + x[1:]))
			fout.write('\n')


def main(argv):
	parser = argparse.ArgumentParser(
		description="Add a suffix after every token in the input file")
	parser.add_argument("--input", help='')
	parser.add_argument("--suffix", help='')
	parser.add_argument("--output")
	parser.add_argument("--only_first",action='store_true',help='Only add prefix to first token')
	args = parser.parse_args(args=argv)


	if args.only_first:
		add_to_first(args)
	else:
		add_to_all(args)


if __name__ == "__main__":
	main(sys.argv[1:])
