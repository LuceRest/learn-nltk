import nltk
import random
from nltk.corpus import movie_reviews


documents = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]
    
random.shuffle(documents)

# print(documents)
print('\n------------------------\n')

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
print(all_words)
print('------------------------\n')
print(all_words.most_common(100))
print('------------------------\n')

print(all_words['stupid'])



'''
    NB :
        - random.shuffle()              ~> Berfungsi untuk mengacak nilai-nilai pada suatu list.
        - random.shuffle(<list>)

        - nltk.FreqDist().most_common() ~> Berfungsi untuk memunculkan nilai yg sering muncul.
        - nltk.FreqDist(<list>).most_common(<banyak nilai yg ingin ditampilkan)

'''

        


