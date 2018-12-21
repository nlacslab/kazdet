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

Important! Pre-trained models for third-party software are _not_ released under CC-SA-BY.
These models assume whatever license corresponding third-party software is distributed under.

<hr>

Citation info: if you are using the treebank or the tools in your research or elsewhere, please cite the following work:
`
Makazhanov, A., Sultangazina, A., Makhambetov, O. and Yessenbayev, Z., 2015. Syntactic annotation of Kazakh: Following the universal dependencies guidelines. A report. In proceedings of the 3rd International Conference on Turkic Languages Processing (TurkLang 2015), Kazan, Tatarstan (pp. 338-350).
`

<hr>

Contact info: figure out or run the following code (need python 3.6+ or [online interpreter](https://www.python.org/shell/))
```python
frst_name = 'Aibek'
last_name = 'Makazhanov'
sep1, sep2 = '.', '@'
print(f'\n{frst_name.lower()}{sep1}{last_name.lower()}{sep2}nu{sep1}edu{sep1}kz\n')
```

<hr>


 

http://www.maltparser.org/download.html


```
java -jar maltparser-1.9.2.jar -c test -i examples/data/talbanken05_test.conll -o out.conll -m parse
```

