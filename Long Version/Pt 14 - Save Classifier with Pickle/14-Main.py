import nltk
import random
from nltk.corpus import movie_reviews
import pickle

documents = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]
    
random.shuffle(documents)

# print(documents)
# print('\n------------------------\n')

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
# print(all_words.most_common(15))
# print('------------------------\n')

# print(all_words['stupid'])

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features ={}
    for w in word_features:
        features[w] = (w in words)

    return features

# print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
# print('------------------------\n')

featuressets = [(find_features(rev), category) for (rev, category) in documents]

training_set = featuressets[:1900]
testing_set = featuressets[1900:]

# classifier = nltk.NaiveBayesClassifier.train(training_set)

classifier_f = open('Long Version/Pt 14 - Save Classifier with Pickle/naivebayes.pickle', 'rb')
classifier = pickle.load(classifier_f)
classifier_f.close()

print('Naive Bayes Algo accuracy percent :', (nltk.classify.accuracy(classifier, testing_set))*100)
print('------------------------\n')
classifier.show_most_informative_features(15)

# save_classifier = open('Long Version/Pt 14 - Save Classifier with Pickle/naivebayes.pickle', 'wb')
# pickle.dump(classifier, save_classifier)
# save_classifier.close()



'''
    NB :
        - Pickle 			                ~> Modul yang dapat digunakan untuk menyimpan dan membaca data ke dalam / dari sebuah file (menyimpan objek dengan python).

        - open() 			                ~> Berfungsi untuk membuka suatu file.
        - <nama variabel> = open(”<nama file>.<ekstensi file>”, “wb”)
            - wb → Write and binary

        - pickle.dump()			            ~> Berfungsi untuk menyimpan (save) objek menjadi file .pickle.
        - pickle.dump(<objek yg ingin disimpan>, <file .pickle>)

        - pickle.load() 		            ~> Berfungsi untuk load objek yg telah disimpan menjadi file .pickle.
        - <nama variabel pickle> = pickle.load(<file .pickle>)

        - <nama variabel pickle>.close()    ~> Berfungsi untuk menutup file .pickle yg telah digunakan.


'''

