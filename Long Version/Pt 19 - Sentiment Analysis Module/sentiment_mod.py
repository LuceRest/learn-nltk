import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import sent_tokenize, word_tokenize
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI):
    def __init__(self, *classifier):
        self._classifiers = classifier

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


short_pos = open('Data/positive.txt','r').read()
short_neg = open('Data/negative.txt','r').read()

# move this up here
all_words = []
documents = []


#  j is adject, r is adverb, and v is verb
#allowed_word_types = ["J","R","V"]
allowed_word_types = ["J"]

for p in short_pos.split('\n'):
    documents.append( (p, "pos") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

    
for p in short_neg.split('\n'):
    documents.append( (p, "neg") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

    
# Save Document
save_documents = open("Data/documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()
    
all_words = nltk.FreqDist(all_words)
# print(all_words.most_common(15))

word_features = list(all_words.keys())[:5000]

# Save Word Features
save_word_features = open("Data/word_features5k.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()

def find_features(document):
    words = set(document)
    features ={}
    for w in word_features:
        features[w] = (w in words)

    return features

# print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
# print('------------------------\n')

featuressets = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featuressets)
print(len(featuressets))

# Positive data example
training_set = featuressets[:10000]
testing_set = featuressets[10000:]

# Negative data example
# training_set = featuressets[:100]
# testing_set = featuressets[100:]

classifier_f = open('Data/originalnaivebayes5k.pickle', 'rb')
classifier = pickle.load(classifier_f)
classifier_f.close()

# classifier = nltk.NaiveBayesClassifier.train(training_set)
print('\nOriginal Naive Bayes Algo accuracy percent :', (nltk.classify.accuracy(classifier, testing_set))*100)
print('------------------------\n')
classifier.show_most_informative_features(15)
print('------------------------\n')

# save_classifier = open('Data/originalnaivebayes5k.pickle', 'wb')
# pickle.dump(classifier, save_classifier)
# save_classifier.close()


# MNB_classifier = SklearnClassifier(MultinomialNB())
# MNB_classifier.train(training_set)
# save_classifier = open("Data/MNB_classifier5k.pickle","wb")
# pickle.dump(MNB_classifier, save_classifier)
# save_classifier.close()

classifier_f = open('Data/MNB_classifier5k.pickle', 'rb')
MNB_classifier = pickle.load(classifier_f)
classifier_f.close()
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)


# BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
# BernoulliNB_classifier.train(training_set)
# save_classifier = open("Data/BernoulliNB_classifier5k.pickle","wb")
# pickle.dump(BernoulliNB_classifier, save_classifier)
# save_classifier.close()

classifier_f = open('Data/BernoulliNB_classifier5k.pickle', 'rb')
BernoulliNB_classifier = pickle.load(classifier_f)
classifier_f.close()
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)


LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
save_classifier = open("Data/LogisticRegression_classifier5k.pickle","wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()

classifier_f = open('Data/LogisticRegression_classifier5k.pickle', 'rb')
LogisticRegression_classifier = pickle.load(classifier_f)
classifier_f.close()
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)


# LinearSVC_classifier = SklearnClassifier(LinearSVC())
# LinearSVC_classifier.train(training_set)
# save_classifier = open("Data/LinearSVC_classifier5k.pickle","wb")
# pickle.dump(LinearSVC_classifier, save_classifier)
# save_classifier.close()

classifier_f = open('Data/LinearSVC_classifier5k.pickle', 'rb')
LinearSVC_classifier = pickle.load(classifier_f)
classifier_f.close()
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)


##NuSVC_classifier = SklearnClassifier(NuSVC())
##NuSVC_classifier.train(training_set)
##print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

# SGDC_classifier = SklearnClassifier(SGDClassifier())
# SGDC_classifier.train(training_set)
# save_classifier = open("Data/SGDC_classifier5k.pickle","wb")
# pickle.dump(SGDC_classifier, save_classifier)
# save_classifier.close()

classifier_f = open('Data/SGDC_classifier5k.pickle', 'rb')
SGDC_classifier = pickle.load(classifier_f)
classifier_f.close()
print("SGDClassifier accuracy percent:",nltk.classify.accuracy(SGDC_classifier, testing_set)*100)


print('\n------------------------\n')


voted_classifier = VoteClassifier(classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier
                                )

print('voted_classifier accuracy percent :', (nltk.classify.accuracy(voted_classifier, testing_set))*100)
print('------------------------\n')

print('Classification:', voted_classifier.classify(testing_set[0][0]), 'Confidence %:', voted_classifier.confidence(testing_set[0][0])*100)
print('Classification:', voted_classifier.classify(testing_set[1][0]), 'Confidence %:', voted_classifier.confidence(testing_set[1][0])*100)
print('Classification:', voted_classifier.classify(testing_set[2][0]), 'Confidence %:', voted_classifier.confidence(testing_set[2][0])*100)
print('Classification:', voted_classifier.classify(testing_set[3][0]), 'Confidence %:', voted_classifier.confidence(testing_set[3][0])*100)
print('Classification:', voted_classifier.classify(testing_set[4][0]), 'Confidence %:', voted_classifier.confidence(testing_set[4][0])*100)
print('Classification:', voted_classifier.classify(testing_set[5][0]), 'Confidence %:', voted_classifier.confidence(testing_set[5][0])*100)
print('------------------------\n')


def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)


