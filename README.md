# kazdet: NLA-NU Kazakh Dependency Treebank

This repository hosts a [Kazakh](https://en.wikipedia.org/wiki/Kazakh_language) [Dependency Treebank](https://en.wikipedia.org/wiki/Treebank) built at the National Laboratory Astana, Nazarbayev University.


As of December 2018, the treebank contains ~61K sentences and ~934.7K tokens (of those 894.3K alphanumeric).
The treebank is annotated for lemma, part-of-speech, morphology, and dependency relations following the [Universal Dependency 2](http://universaldependencies.org/) guidelines and is stored in the UD-native
[CoNLL-U format](http://universaldependencies.org/format.html). A sample parse in CoNLL-U may look like this:

```
# text = Бір ай болады, Шолпан күнәға белін берік байлаған.
1	Бір	бір	NUM	NUM	_	2	nummod	_	_
2	ай	ай	NOUN	NOUN	_	3	nsubj	_	_
3	болады	бол	VERB	VERB	vbTense=Aor|Person=3	0	root	_	SpaceAfter=No
4	,	,	PUNCT	PUNCT	_	3	punct	_	_
5	Шолпан	Шолпан	PROPN	PROPN	_	9	nsubj	_	_
6	күнәға	күнә	NOUN	NOUN	Case=Dat	9	iobj	_	_
7	белін	бел	NOUN	NOUN	Case=Acc|Poss=3	9	dobj	_	_
8	берік	берік	ADV	ADV	_	9	advmod	_	_
9	байлаған	байла	VERB	VERB	vbTense=Pst	3	parataxis	_	SpaceAfter=No
10	.	.	PUNCT	PUNCT	_	9	punct	_	_
```

In addition to the treebank itself, the project delivers a number of language processing tools, such as a basic parsing pipeline
(tokenization -> tagging -> parsing), a named entity recognizer and a prototype machine translation system.
Some of the tools are implemented as Python 3 modules, others are based on third-party implementations and for them pre-trained 
models for Kazakh are provided, and a translation prototype has been made available as a web-service.

The treebank and our own Python-based implementations of the NLP tools are released under the 
[CC-SA-BY](https://creativecommons.org/licenses/by-sa/4.0/) license that permits any use, 
provided that the source is attributed (cited) and the content is not changed if shared (share alike).

__Important! Pre-trained models for third-party software are _not_ released under CC-SA-BY.
These models assume whatever license corresponding third-party software is distributed under.__

<hr>

If you are using the treebank in your research or elsewhere, please cite the following work:
`
Makazhanov, A., Sultangazina, A., Makhambetov, O. and Yessenbayev, Z., 2015. Syntactic annotation of Kazakh: Following the universal dependencies guidelines. A report. In proceedings of the 3rd International Conference on Turkic Languages Processing (TurkLang 2015), Kazan, Tatarstan (pp. 338-350).
`
Citation info on the tools will be given further along whenever relevant.

<hr>

Contact info: figure out or run the following code (need python 3.6+ or [online interpreter](https://www.python.org/shell/))
```python
frst_name = 'Aibek'
last_name = 'Makazhanov'
sep1, sep2 = '.', '@'
print(f'\n{frst_name.lower()}{sep1}{last_name.lower()}{sep2}nu{sep1}edu{sep1}kz\n')
```

<hr>

### 1 Download

To download the treebank and all the tools and models just clone this repository like so:
```shell
> git clone https://github.com/nlacslab/kazdet.git
```
or download the archived version from https://github.com/nlacslab/kazdet/archive/master.zip.

The treebank is located inside the `data` directory and is compressed with 7zip archivator.

To extract the treebank on a Linux system, `cd` to the data directory and extract the contents of the archive like so:
```shell
~/kazdet/ > cd data
~/kazdet/data > 7z x kdt-NLANU-0.01.connlu.txt.7z
```

<hr>

### 2 Tools

What follows is a chronologically orgonized (according to the project implementation schedule) list of usage examples (when applicable) of the tools developped for the project. To use provided Python scripts make sure that your Python version is 3.6 or higher. Currently all of the usage examples are made for Linux systems, Windows tutorial can be made upon request.

__=== 2016 ===__

#### 2.1 The annotation tool
For this purpose a third-party tool [BRAT](http://brat.nlplab.org/) due to its support of multi-user and online annotation modes. 
The tool comes with its own web-server which was configured and run.
The tool must be configured depending on the annotation task at hand.
The configuration files that were used in our project can be found in the `brat-config.zip` archive in the `misc` directory of the present repo. Please, cite [\[1\]](#ref01), if you use these configuration files.

Unfortunately, the web-server that we used is no longer running, hence we cannot link to it.
Below is a screen shot of the configured BRAT instance in use:
![BRAT annotation tool](https://github.com/nlacslab/kazdet/blob/master/misc/brat_pic.png)

#### 2.2 Parsing pipeline v1

The first version of the basic parsing pipeline consisted of a combination of our implementation of a tokenizer -> tagger pipeline and the state-of-the-art dependency parser at the moment, the [maltparser](http://www.maltparser.org).
This version of the parsing pipeline was used in 2016 and 2017 for a semi-automatic annotation and building of the treebank.

The first step in using this pipeline is to tokenize and tag the input, using the `tagpipe.py` midule which is located in the tools directory, i.e. `kazdet/tools`. The module has the following command line options and usage:
```shell
~/kazdet/tools > python tagpipe.py -h
usage: tagpipe.py [-h] [-i INPUT] [-o OUTPUT] [-m MODEL_DIR] [-f FORMAT]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input file name
  -o OUTPUT, --output OUTPUT
                        output file name
  -m MODELDIR, --modeldir MODELDIR
                        model directory (default: model.mor)
  -f FORMAT, --format FORMAT
                        output format: 0 - plain (default); 1 - conllu; 2 -
                        conllx; [anything else] - plain.
```
Thus, to tokenize and tag an input file `in.txt` (we might need to create it beforehand) using the trigram HMM model (`model.mor` is set by default) and save it to the file `in.toktag.txt` in a conllx format (as it used by the maltparser), we need to run the following commands:
```shell
~/kazdet/tools > echo 'Еңбек етсең ерінбей, тояды қарның тіленбей.' > in.txt
~/kazdet/tools > python tagpipe.py -i in.txt -o in.toktag.txt -f 2
```
... and to check the output:
```
~/kazdet/tools >  cat in.toktag.txt 

1	Еңбек	еңбек	NOUN	NOUN	_	_	_	0	0
2	етсең	ет	VERB	VERB	vbMood=Cond|Person=2	_	_	0	0
3	ерінбей	ерін	VERB	VERB	vbNeg=True|vbType=Cvb	_	_	0	0
4	,	,	PUNCT	PUNCT	_	_	_	0	0
5	тояды	то	VERB	VERB	vbTense=Aor|Person=3	_	_	0	0
6	қарның	қарн	NOUN	NOUN	Poss=2	_	_	0	0
7	тіленбей	тіле	VERB	VERB	vbVcRefx=True|vbNeg=True|vbType=Cvb	_	_	0	0
8	.	.	PUNCT	PUNCT	_	_	_	0	0
```
\* _the morphological processing tools (analyzer and parser) constitute re-implementation of our earlier work, see [1] for example._

The next step in the pipeline is that this tokenized and tagged sentence needs to be parsed.
To this end we need to download maltparser, which is distributed as a java binary code and needs no compilation.

Download the latest release of the maltparser from http://maltparser.org/dist/maltparser-1.9.2.zip and `unzip` (this instruction uses the tools directory of the repo, i.e. `kazdet/tools`, for conveniene):
```shell
~/kazdet/tools > unzip maltparser-1.9.2.zip
```
Enter the maltparser directory and copy the `malt_kdt_001.mco`file from the `models` into this directory from the models directory:
```
~/kazdet/tools > cd maltparser-1.9.2/
~/kazdet/tools > cp ../../models/malt_kdt_001.mco .
```
Now everything is set up for parsing.
To parse our tokenized and tagged file `in.toktag.txt`, save the output to `out.txt`,
and to view the output run the following commands:
```shell
~/kazdet/tools/maltparser-1.9.2 > java -jar maltparser-1.9.2.jar -c malt_kdt_001 -i ../in.toktag.txt -o ../out.tx -m parse
~/kazdet/tools/maltparser-1.9.2 > head ../out.txt
1	Еңбек	еңбек	NOUN	NOUN	_	2	dobj	_	_
2	етсең	ет	VERB	VERB	vbMood=Cond|Person=2	3	ccomp	_	_
3	ерінбей	ерін	VERB	VERB	vbNeg=True|vbType=Cvb	7	advcl	_	_
4	,	,	PUNCT	PUNCT	_	7	punct	_	_
5	тояды	то	VERB	VERB	vbTense=Aor|Person=3	6	acl-relcl	_	_
6	қарның	қарн	NOUN	NOUN	Poss=2	7	nsubj	_	_
7	тіленбей	тіле	VERB	VERB	vbVcRefx=True|vbNeg=True|vbType=Cvb	0	root	_	_
8	.	.	PUNCT	PUNCT	_	7	punct	_	_                                                             
```

__=== 2017 ===__

#### 2.3 Parsing pipeline v2

This pipeline is based on another third-party tool, [udpipe](http://ufal.mff.cuni.cz/udpipe), and covers two objectives of the 2017 implementation schedule, i.e. developing neural-based morphological tagger and parser.

To use this pipeline, first, download precompiled `udpipe` binaries from https://github.com/ufal/udpipe/releases/download/v1.2.0/udpipe-1.2.0-bin.zip and extract its contents to the `tools` directory.

In this case we need to explicitly state the pipeline steps through the arguments to the tool, followed by the indication of the model and input files:
```shell
~/kazdet/tools >  ./udpipe-1.2.0-bin/bin-linux64/udpipe --tokenize --tag --parse ../models/udpipe_kdt_001.mdl in.txt 
Loading UDPipe model: done.
# newdoc id = in.txt
# newpar
# sent_id = 1
# text = Еңбек етсең ерінбей, тояды қарның тіленбей.
1	Еңбек	Еңбек	PROPN	_	_	3	nsubj	_	_
2	етсең	етсең	ADV	_	_	3	advmod	_	_
3	ерінбей	ерін	VERB	_	vbNeg=True|vbType=Adv	7	advcl	_	SpaceAfter=No
4	,	,	PUNCT	_	_	7	punct	_	_
5	тояды	тояды	ADJ	_	_	6	amod	_	_
6	қарның	қар	NOUN	_	Case=Gen	7	nsubj	_	_
7	тіленбей	тілен	VERB	_	vbNeg=True|vbType=Adv	0	root	_	SpaceAfter=No
8	.	.	PUNCT	_	_	7	punct	_	SpacesAfter=\n
```

#### 2.4 Machine translation prototype

We have experimented with both statistical and neural machine translation systems, as well as linguistically motivated factored and rule-based models. This involved considerable amount of work and has resulted in a 
We have built a machine translation system prototype to translate 

The machine t


### References

<a name="ref01"></a> 1. Makazhanov, A., Sultangazina, A., Makhambetov, O. and Yessenbayev, Z., 2015. Syntactic annotation of Kazakh: Following the universal dependencies guidelines. A report. In proceedings of the 3rd International Conference on Turkic Languages Processing (TurkLang 2015), Kazan, Tatarstan (pp. 338-350).





 

