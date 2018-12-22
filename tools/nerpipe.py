# -*- coding: UTF-8 -*-

from __future__ import division
import argparse
import codecs
import utils


def parse_cl():
    cla_parser = argparse.ArgumentParser()
    cla_parser.add_argument('-i', '--input', help='input file name')
    cla_parser.add_argument('-o', '--output', help='output file name')
    cla_parser.add_argument('-M', '--modelner', action='store',
                            default='model.ner',
                            help='NER model (default is model.ner)')
    cla_parser.add_argument('-m', '--modelmor', action='store',
                            default='model.mor',
                            help='morphological processing \
                            model directory (default is model.mor)')
    return cla_parser.parse_args()


def main():

    # parse command line arguments
    args = parse_cl()

    # create a naive bayes classifier instance
    model = utils.NB(args.modelner)

    # create a morphological analyzer instance
    analyzer = utils.AnalyzerDD()
    analyzer.load_model(args.modelmor)

    # create a morphological tagger instance
    tagger = utils.TaggerHMM(lyzer=analyzer)
    tagger.load_model(args.modelmor)

    # create a tokenizer instance
    tokenizer = utils.TokenizeRex()

    # get the input and prepare the output
    txt = codecs.open(args.input, 'r', 'utf-8').read().strip()
    fd = codecs.open(args.output, 'w', 'utf-8')

    # tokenize and extract named entities
    for sentence in tokenizer.tokenize(txt):
        lower_sentence = map(lambda x: x.lower(), sentence)
        analyses = tagger.tag_sentence(lower_sentence)
        print('', file=fd)
        for i, nertag in enumerate(model.predict(analyses)):
            print(i + 1, sentence[i], nertag, file=fd, sep='\t')
        print('', file=fd)


if __name__ == '__main__':
    main()
