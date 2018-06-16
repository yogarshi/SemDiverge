__author__ = 'yogarshi'
import sys, os
import numpy as np
import argparse

def fraction_of_words(s1,s2,trans_dict):
	count = 0
	for w1 in s1:
		try:
			for each_w in trans_dict[w1]:
				if each_w in s2:
					count += 1
					break
		except KeyError:
			continue
	return count



# def main(argv):
# 	parser = argparse.ArgumentParser(description="")
# 	parser.add_argument("", help='')
# 	args = parser.parse_args(args=argv)
#
#
# if __name__ == "__main__":
# 	main(sys.argv[1:])
