import nltk
import random
from nltk.corpus import movie_reviews


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

classifier = nltk.NaiveBayesClassifier.train(training_set)
print('Naive Bayes Algo accuracy percent :', (nltk.classify.accuracy(classifier, testing_set))*100)
print('------------------------\n')
classifier.show_most_informative_features(15)



'''
    NB :
        - nltk.NaiveBayesClassifier.train() ~> Berfungsi untuk melakukan training pada suatu dataset dengan menggunakan algoritma Naïve Bayes.
        - nltk.NaiveBayesClassifier.train(<training dataset>)

        - nltk.classify.accuracy()          ~> Berfungs untuk mendapatkan akurasi dari suatu klasifier dengan mengetes suatu dataset.
        - nltk.classify.accuracy(<classifier>, <testing dataset>)

        - show_most_informative_features()  ~> Berfungsi untuk menampilkan fitur paling informatif.
        - show_most_informative_features(<jumlah data yg ingin ditampilkan>)

'''

