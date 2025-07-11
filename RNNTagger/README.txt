
General remarks:

RNNTagger is a part-of-speech tagger and lemmatizer supporting over 50
languages.  RNNTagger is written in Python3 (and some Perl scripts)
and uses the PyTorch machine learning library.


Installation:

In order to use RNNTagger, you need to install Python3, Perl and
PyTorch. 

The directory "cmd" contains a Linux shell script and a Windows batch
script for each language.


After unpacking the zip file, you can call RNNTagger as follows (on Linux systems):

> cd RNNTagger
> echo "This is a test." > test.txt
> cmd/rnn-tagger-english.sh test.txt
This 	DT 	this 
is 	VBZ 	be 
a 	DT 	a 
test 	NN 	test 
. 	. 	. 

Taggers for other languages are called in the same way:
> cmd/rnn-tagger-<language>.sh file


Usage on Windows

Open a command prompt window. Change to the directory RNNTagger and enter
> cmd/rnn-tagger-german.bat text.txt
to annotate the file "text.txt".

The annotation process usually consists of three steps:

The included simple tokenizer first splits the input text into
"tokens" (i.e. words, punctuation, parentheses etc.) Each token is
written on a separate line and each sentence is followed by an empty
line.

The part-of-speech (POS) tagger reads the token sequence and assigns a
POS tag to each token.

The lemmatizer extracts all word-POS tag combinations from the tagged
token sequence, computes the lemma for each pair, and then looks up
the computed lemma for each token-tag pair of the POS-tagged token
sequence.

The Korean tagger uses a special XML-based output format which shows
the eojeols and their components.

The taggers for Old English and Swiss German generate no lemmas.

The Syriac tagger expects tokenized input with one sentence per line.

Information on the part-of-speech tagsets is available via the URLs
listed in the file "Tagset-Information".

The taggers for Old French and Middle French additionally use a
separate lexicon for lemmatization. If the lemma for a given word/tag
combination is not found in this lexicon, then these taggers print the
lemma predicted by the NMT lemmatizer in parentheses. 
