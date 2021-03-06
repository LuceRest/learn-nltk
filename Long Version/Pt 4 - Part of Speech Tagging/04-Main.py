import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw('2005-GWBush.txt')
sample_text = state_union.raw('2006-GWBush.txt')

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print(tagged)
    
    except Exception as e:
        print(str(e))


process_content()



'''
    NB :
        - Tagging
            Tahap untuk mencari bentuk awal dari tiap kata lampau atau hasil dari stemming yang masih memuat beberapa kata lampau yang dikembalikan ke bentuk awalnya.
            
        - Punkt Sentence Tokenizer
            Tokenizer ini membagi teks menjadi daftar kalimat, dengan menggunakan algoritma yang tidak diawasi untuk membangun model kata singkatan, kolokasi, dan kata yang memulai kalimat. Itu harus dilatih pada banyak koleksi plaintext dalam bahasa target sebelum dapat digunakan.

        - PoS tagging → Proses memberikan label kelas kata secara otomatis pada setiap kata yang ada pada suatu teks atau dokumen.

'''



'''

List of Universal POS tags:
•	ADJ : Adjective
•	ADV : Adposition
•	ADP : Adverb
•	AUX : Auxiliary
•	CCONJ : Coordinating Conjuction
•	DET : Determiner
•	INTJ : Interjection
•	NOUN : Noun
•	NUM : Numeral
•	PART : Particle
•	PRON : Pronoun
•	PROPN : Proper Noun
•	PUNCT : Punctuation
•	SCONJ : Subordinating Conjuction
•	SYM : Symbol
•	VERB : Verb
•	X : Other

Another LIST OF TAGS
•	CC : coordinating conjunction
•	CD : cardinal digit
•	DT : determiner
•	EX : existential there (like: “there is” … think of it like “there exists”)
•	FW : foreign word
•	IN : preposition/subordinating conjunction
•	JJ : adjective ‘big’
•	JJR : adjective, comparative ‘bigger’
•	JJS : adjective, superlative ‘biggest’
•	LS : list marker 1)
•	MD : modal could, will
•	NN : noun, singular ‘desk’
•	NNS : noun plural ‘desks’
•	NNP : proper noun, singular ‘Harrison’
•	NNPS : proper noun, plural ‘Americans’
•	PDT : predeterminer ‘all the kids’
•	POS : possessive ending parent‘s
•	PRP : personal pronoun I, he, she
•	PRPS : possessive pronoun my, his, hers
•	RB : adverb very, silently,
•	RBR : adverb, comparative better
•	RBS : adverb, superlative best
•	RP : particle give up
•	TO : to go ‘to‘ the store.
•	UH : interjection errrrrrrrm
•	VB : verb, base form take
•	VBD : verb, past tense took
•	VBG : verb, gerund/present participle taking
•	VBN : verb, past participle taken
•	VBP : verb, sing. present, non-3d take
•	VBZ : verb, 3rd person sing. present takes
•	WDT : wh-determiner which
•	WP : wh-pronoun who, what
•	WPS : possessive wh-pronoun whose
•	WRB : wh-abverb where, when



'''