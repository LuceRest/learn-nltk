from nltk.corpus import wordnet

syns = wordnet.synsets('program')

# Synset
print(syns[0].name() + '\n')

# Just the word
print(syns[0].lemmas()[0].name() + '\n')

# Definition
print(syns[0].definition() + '\n')

# Example
print(f'{syns[0].examples()} \n')


print('------------------------\n')

synonyms = []
antonyms = []

for syn in wordnet.synsets('good'):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(f'{set(synonyms)} \n')
print(f'{set(antonyms)} \n')


print('------------------------\n')

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('boat.n.01')
print(str(w1.wup_similarity(w2)) + '\n')

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('car.n.01')
print(str(w1.wup_similarity(w2)) + '\n')

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('cactus.n.01')
print(str(w1.wup_similarity(w2)) + '\n')



'''
    NB :
        - WordNet → Dabatase bahasa yang digunakan untuk mencari synonym set (synset) pada sebuah kata (lemma) yang nantinya akan berelasi dari satu lema dengan lemma lainnya.

        - <word 1>.wup_similarity(<word 2>) ~> Berfungsi untuk mengcompare similarity (kesamaan) antara kata 1 dengan kata 2.


'''

