# kazdet: NLA-NU Kazakh Dependency Treebank

This repository hosts a [Kazakh](https://en.wikipedia.org/wiki/Kazakh_language) [Dependency Treebank](https://en.wikipedia.org/wiki/Treebank) built at the National Laboratory Astana, Nazarbayev University.


As of December 2018, the treebank contains ~61K sentences and ~934.7K tokens overall and 894.3K alphanumeric.
The treebank is unnotated for lemma, part-of-speech, morphology, and dependency relations following (to the extent currently possible) the [Universal Dependency 2](http://universaldependencies.org/) guidelines and is stored in the UD-native
[CoNLL-U format](http://universaldependencies.org/format.html).

The project is implemented by the computer science lab of the National Laboratory Astana, Nazarbayev University.


```
# sent_id = 7 @ ../data_brat/_zhasalash/_zha_0281_87.ann
# text = Ол Жезқазған жылу электр кәсіпорнындағы қазандықта жарақат алған.
1	Ол	ол	PRON	PRON	_	8	nsubj	_	_
2	Жезқазған	Жезқазған	PROPN	PROPN	_	5	nmod-poss	_	_
3	жылу	жылу	NOUN	NOUN	_	4	compound-nn	_	_
4	электр	электр	NOUN	NOUN	_	5	nmod-poss	_	_
5	кәсіпорнындағы	кәсіпорнындағы	ADJ	ADJ	_	6	amod	_	_
6	қазандықта	қазандық	NOUN	NOUN	Case=Loc	8	nmod	_	_
7	жарақат	жарақат	NOUN	NOUN	_	8	dobj	_	_
8	алған	ал	VERB	VERB	vbType=Cvb	0	root	_	SpaceAfter=No
9	.	.	PUNCT	PUNCT	_	8	punct	_	_
```

http://www.maltparser.org/download.html


java -jar maltparser-1.9.2.jar -c test -i examples/data/talbanken05_test.conll -o out.conll -m parse
