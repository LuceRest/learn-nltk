import nltk
from nltk.corpus import wordnet


syn = wordnet.synsets('Computer')
print(f'\nsyn : {syn[0].definition()}\n \n----------------------\n')


# synonyms = []
# for syn in wordnet.synsets('Computer'):
#     for lemma in syn.lemmas():
#         synonyms.append(lemma.name())
        
# print(f'\nsynonyms : {synonyms}\n \n----------------------\n')


# antonyms = []
# for syn in wordnet.synsets('small'):
#     for lemma in syn.lemmas():
#         if lemma.antonyms:
#             antonyms.append(lemma.antonyms()[0].name())
#             # print(lemma.antonyms().lexname())
        
# print(f'\nantonyms : {antonyms}\n \n----------------------\n')



'''
    NB :
        - wordnet.synsets()		~> Berfungsi untuk mendapatkan berbagai sinonim dari suatu kata.
        - wordnet.synsets('<word>')
'''

