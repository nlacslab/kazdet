# -*- coding: UTF-8 -*-
from __future__ import print_function
import argparse
import codecs
import utils


def parse_cl():
    cla_parser = argparse.ArgumentParser()
    cla_parser.add_argument('-i', '--input', help='input file name')
    cla_parser.add_argument('-o', '--output', help='output file name')
    cla_parser.add_argument('-m', '--modeldir', action='store',
                            default='model.mor',
                            help='model directory (default: model.mor)')
    cla_parser.add_argument(
            '-f', '--format',
            type=int, action='store', default=0,
            help='output format: 0 - plain (default); \
            1 - conllu; 2 - conllx; [anything else] - plain.')
    args = cla_parser.parse_args()
    if args.format not in range(3):
        args.format = 0
    return args


def main():

    args = parse_cl()

    # create a morphological analyzer instance
    analyzer = utils.AnalyzerDD()
    analyzer.load_model(args.modeldir)

    # create a morphological tagger instance
    tagger = utils.TaggerHMM(lyzer=analyzer)
    tagger.load_model(args.modeldir)

    # create a tokenizer instance
    tokenizer = utils.TokenizerHMM(model='model.tok')

    # get the input and prepare the output
    txt = codecs.open(args.input, 'r', 'utf-8').read().strip()
    fd = codecs.open(args.output, 'w', 'utf-8')

    # tokenize and tag
    for sentence in tokenizer.tokenize(txt):
        lower_sentence = map(lambda x: x.lower(), sentence)
        print('', file=fd)
        for i, a in enumerate(tagger.tag_sentence(lower_sentence)):
            # print(f'{str(i+1).rjust(2)}) {sentence[i].ljust(15)}{a}')
            if args.format:
                print(utils.klc2conll(
                        sentence[i], a, i + 1, args.format - 1), file=fd)
            else:
                print(i + 1, sentence[i], a, file=fd, sep='\t')
        print('', file=fd)


if __name__ == '__main__':
    main()
