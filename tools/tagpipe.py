# -*- coding: UTF-8 -*-
from __future__ import print_function
import argparse
import codecs
import utils


def parse_cl():
    cla_parser = argparse.ArgumentParser()
    cla_parser.add_argument('-i', help='input file name')
    cla_parser.add_argument('-o', help='output file name')
    cla_parser.add_argument('-m', action='store',
                            default='model.mor', help='model directory')
    cla_parser.add_argument(
            '-f',
            type=int, action='store', default=0,
            help='output format: 0 - plain (default); \
            1 - conllu; 2 - conllx; [anything else[ - plain.')
    args = cla_parser.parse_args()
    if args.f not in range(3):
        args.f = 0
    return args


def main():

    args = parse_cl()

    # create a morphological analyzer instance
    analyzer = utils.AnalyzerDD()
    analyzer.load_model(args.m)

    # create a morphological tagger instance
    tagger = utils.TaggerHMM(lyzer=analyzer)
    tagger.load_model(args.m)

    # create a tokenizer instance
    tokenizer = utils.TokenizerHMM(model='model.tok')

    # get the input and prepare the output
    txt = codecs.open(args.i, 'r', 'utf-8').read().strip()
    fd = codecs.open(args.o, 'w', 'utf-8')

    # tokenize and tag
    for sentence in tokenizer.tokenize(txt):
        lower_sentence = map(lambda x: x.lower(), sentence)
        print('', file=fd)
        for i, a in enumerate(tagger.tag_sentence(lower_sentence)):
            # print(f'{str(i+1).rjust(2)}) {sentence[i].ljust(15)}{a}')
            if args.f:
                print(utils.klc2conll(
                        sentence[i], a, i + 1, args.f - 1), file=fd)
            else:
                print(i + 1, sentence[i], a, file=fd, sep='\t')
        print('', file=fd)


if __name__ == '__main__':
    main()
