# kazdet: NLA-NU Kazakh Dependency Treebank

This repository hosts a [Kazakh](https://en.wikipedia.org/wiki/Kazakh_language) [Dependency Treebank](https://en.wikipedia.org/wiki/Treebank) built at the National Laboratory Astana, Nazarbayev University.


As of December 2018, the treebank contains ~61K sentences and ~934.7K tokens (of those 772.8K alphanumeric).
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
Here a parse of a sentence is represented by a tab-separated list of values for each token.
There are 10 columns that encode the following values: (i) token id (ordinal number in a sentence); (ii) surface form of a token; (iii) lemma (dictionary form) of a token; (iv) universal part-of-speech tag; (v) alternative part-of-speech tag (identical to universal tag in our case); (vi) vertical bar-separated list of morphological features; (vii) id of the token that governs the current token (0 stands for a conventional root of the sentence); (viii) dependency relation between the dependee and the governor; (ix) enhanced dependencies (multi-headed representation); (x) any other annotation, in our case is used to indicate (for tokenization purposes) whether the next token follows the current one without a space.

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
```shell
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
\* _the morphological processing tools (analyzer and parser) constitute re-implementation of our earlier work, see [\[2\]](#ref02) for example._

The next step in the pipeline is that this tokenized and tagged sentence needs to be parsed.
To this end we need to download maltparser, which is distributed as a java binary code and needs no compilation.

Download the latest release of the maltparser from http://maltparser.org/dist/maltparser-1.9.2.zip and `unzip` (this instruction uses the tools directory of the repo, i.e. `kazdet/tools`, for conveniene):
```shell
~/kazdet/tools > unzip maltparser-1.9.2.zip
```
Enter the maltparser directory and copy the `malt_kdt_001.mco`file from the `models` into this directory from the models directory:
```shell
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
~/kazdet/tools > ./udpipe-1.2.0-bin/bin-linux64/udpipe --tokenize --tag --parse ../models/udpipe_kdt_001.mdl in.txt 
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

We have experimented with both statistical and neural machine translation systems, as well as linguistically motivated factored and rule-based models. This involved considerable amount of work and has resulted in a number of publications, e.g. [\[3\]](#ref02), [\[4\]](#ref02), and [\[5\]](#ref02).

Based on the superior accuracy and speed the prototype of statistical MT was built using [Moses SMT tool](http://www.statmt.org/moses/). It is not released here, due to relative complexity of both usage and documenting, but it is available as a web-server [here](http://kazcorpus.kz/translator/).


__=== 2018 ===__

#### 2.5 Named entity recognizer

This tool identifies named entities (personal names, toponyms, etc.) in a given text and classifies them into one of four categories: (i) PER - personal name; (ii) LOC - location (name of places); (iii) ORG - organization; (iv) OTH - any other named entity, e.g. pet name, book title etc.

Our implementation is based on identification of proper nouns using our tokenizer-tagger pipeline (see sec. 2.2) and then classifying them into NE classes using local context features, such as lemma and POS, in a naive bayes setting.
Here are the command line options for the NER tool:
```shell
~/kazdet/tools > python nerpipe.py -h
usage: nerpipe.py [-h] [-i INPUT] [-o OUTPUT] [-M MODELNER] [-m MODELMOR]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input file name
  -o OUTPUT, --output OUTPUT
                        output file name
  -M MODELNER, --modelner MODELNER
                        NER model (default is model.ner)
  -m MODELMOR, --modelmor MODELMOR
                        morphological processing model directory (default is
                        model.mor)
```

Thus, to extract named entities from file `in.txt` (we may need to create one in advance) and record the output to `out.txt` run the following comands:
```shell
~/kazdet/tools > echo 'Ресей Президенті, Путин, ресми сапармен Қазақстанға келді.' > in.txt
~/kazdet/tools > python nerpipe.py -i in.txt -o out.txt
```
... and to check the output:
```shell
~/kazdet/tools > cat out.txt 

1	Ресей	LOC
2	Президенті	PER
3	,	_
4	Путин	PER
5	,	_
6	ресми	_
7	сапармен	_
8	Қазақстанға	LOC
9	келді	_
10	.	_

```

#### 2.6 Language modeling for Kazakh

This objective was included into 2018 implementation schedule mainly for the benefit of gaining expertise with deep neural language modeling techniques. 
To this end, a two-layer LSTM was implemented in TensorFlow and tested on a range of hyperparameter settings, of which the model with 1500 units unrolled for 35 steps achieved the best performance (compared to the same neural architectures with different parameters and to a statistical ngram language models).

About 887K, 67K, and 79K tokens where used for training, validation, and testing respectively. The performance, measured as perplexity on the test set (the lower the better) for trigram and LSTM models was 123.7 and 115.5 respectively. Later the LSTM model was retrained on a morpheme-like sub-word units to achieve an even lower perplexity of 37.5.

All the data used for the experiments are available for free at https://github.com/Baghdat/LSTM-LM.

<hr>

### 3 Less programming-intensive use cases

Addmittedly this document is oriented mostly on computer scientists / computational linguists who are interested in dependency parsing and would like to work with Kazakh language in this respect.
To compensate, we feel obliged to showcase a couple of less "computational" and more "linguistics" use cases.

#### 3.1 Visualization

The CONNL-U format might be informative and good for machine processing, but it is hard to imagine a syntax tree of a sentence just by looking at those rows and columns of text specifications.

The `UDPipe` project mentioned earlier in this document not only provides an effective parsing pipeline framework, but also a convenient visualization tool, which can be accessed through their website at http://lindat.mff.cuni.cz/services/udpipe/.

Let us try to visualize the sample sentence from the very beginning of this dcument. To do that we need to copy-paste it into the input text area of the UDPipe service landing page:

![UDPipe visualization example: part 1](https://github.com/nlacslab/kazdet/blob/master/misc/udpipe_vis01.png)

We need to make sure that `UD 2.0` is checked as the `Model` and Kazakh UD treebank is chosen from the drop-down list.
`Actions` check boxes must all be deselected and `CoNNL-U` is chosen as an input under the `Advanced options` panel.

If we now hit the `Process input` button and select the `Show trees` tab, we get a nice visualization, where clicking on each node opens up a side panel with additional information:

![UDPipe visualization example: part 2](https://github.com/nlacslab/kazdet/blob/master/misc/udpipe_vis02.png)


#### 3.2 Search and visualization

What if we need to visualize trees that have certain properties, e.g. contain a certain relation, lemma, tag etc.
To this end we can use a query language and search platform developed at Turku university and located at http://bionlp-www.utu.fi/dep_search/.

Let us try to find sentences, where the word _қой_ is governed by any kind of relation.
On a [landing page](http://bionlp-www.utu.fi/dep_search/) choose `Kazakh (UDv2.0)` from the top-left drop-down list and enter the query `қой < _` (any tree, where `қой` is governed by any relation) to the textbox to the right of that drop-down list.
Now upon hitting `search` we should get two trees, with their parses visualized in a "flat-tree" format similar to that used in BRAT annotation tool (see seq. 2.1):

![Query + visualization example: part 1](https://github.com/nlacslab/kazdet/blob/master/misc/turku_vis01.png)
To submit other queries, take a look at the [specifications of the query language](http://bionlp.utu.fi/searchexpressions-new.html) used by the tool.

If we look closer, there are two options at top-left corner of each tree: `context` and `connlu`.
As the names suggest, the first one shows the sentence in context (if any), and the latter - produces already familiar CoNLL-U formatted text:

![Query + visualization example: part 2](https://github.com/nlacslab/kazdet/blob/master/misc/turku_vis02.png)

<hr>

### References

<a name="ref01"></a> \[1\] Makazhanov, A., Sultangazina, A., Makhambetov, O. and Yessenbayev, Z., 2015. Syntactic annotation of Kazakh: Following the universal dependencies guidelines. A report. In proceedings of the 3rd International Conference on Turkic Languages Processing (TurkLang 2015), Kazan, Tatarstan (pp. 338-350).

<a name="ref02"></a> \[2\] O. Makhambetov, A. Makazhanov, I. Sabyrgaliyev, Zh. Yessenbayev. "Data-driven morphological analysis and disambiguation for Kazakh". InInternational Conference on Intelligent Text Processing and Computational Linguistics 2015, pp. 151-163.

<a name="ref03">\[3\] Makazhanov, A., Myrzakhmetov, B., & Kozhirbayev, Z. (2017). On Various Approaches to Machine Translation from Russian to Kazakh.

<a name="ref04">\[4\] Myrzakhmetov, B., Sultangazina, A., & Makazhanov, A. Identification of the parallel documents from multilingual news websites. In Application of Information and Communication Technologies (AICT), 2016 IEEE 10th International Conference on (pp. 1-5). IEEE.

<a name="ref05">\[5\] Assylbekov, Z., Myrzakhmetov, B., & Makazhanov, A. (2016). Experiments with Russian to Kazakh sentence alignment. The 4-th International Conference on Computer Processing of Turkic Languages “TurkLang 2016”.

