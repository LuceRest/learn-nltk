import nltk

nltk.download('wordnet')
nltk.download('vader_lexicon')

from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

text = '''Welcome to the hell. Lets start with our first tutorial on NLTK. We shall leaern the basics of NLTK here.add()'''
demoWords = ['playing', 'hapiness', 'going', 'doing', 'yes', 'no', 'I', 'having', 'had', 'haved', 'coding', 'programming', 'code', 'program']

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
for word in demoWords:
    # Word | Stem | Lemmatize
    print(f'{word} | {stemmer.stem(word)} | {lemmatizer.lemmatize(word, "v")}')

sia = SentimentIntensityAnalyzer()
print(f'\nsia : {sia.polarity_scores("This is not good at all")}\n \n----------------------\n')



'''
    NB :
        - PorterStemmer().stem()	                        ~> Berfungsi untuk melakukan stemming pada suatu kata.
        - PorterStemmer().stem(<word>)

        - WordNetLemmatizer().lemmatize()	                ~> Berfungsu untuk melakukan lemmatizer pada suatu kata.
        - WordNetLemmatizer().lemmatize(<word>, "<pos>")

        - SentimentIntensityAnalyzer().polarity_scores()	~> Berfungsi untuk menampilkan hasil dari sentiment analisis dari suatu kalimat.
        - SentimentIntensityAnalyzer().polarity_scores(<words>)

'''

