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
            # print(tagged)
            
            # namedEnt = nltk.ne_chunk(tagged, binary=True)
            namedEnt = nltk.ne_chunk(tagged)
            namedEnt.draw()
    
    except Exception as e:
        print(str(e))


process_content()



'''
    NB :
        - nltk.chunk.ne_chunk() ~> Berfungsi untuk memberi nama pada entity chunker untuk memotong (chunk) daftar token yang diberi tag.
        - nltk.chunk.ne_chunk(<tagged tokens*>,* binary=False/True)
            - binary = False    ~> Berfungsi untuk menampilkan nama entity chunker.
            - binary = True     ~> Berfungsi untuk tidak menampilkan nama entity chunker.

'''

'''
    NE Type Examples

    ORGANIZATION    Georgia-Pacific Corp., WHO
    LOCATION        Murray River, Mount Everest
    DATE            June, 2008-06-29
    PERSON          Eddy Bonte, President Obama  
    TIME            two fifty a m, 1:30 p.m.
    MONEY           175 million Canadian Dollars, GBP 10.40  
    PERCEN          twenty pct, 18.75 %
    FACILITY        Washington Monument, Stonehenge 
    GPE             South East Asia, Midlothian.
    
'''