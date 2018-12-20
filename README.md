# kazdet
NLA-NU Kazakh Dependency Treebank

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
