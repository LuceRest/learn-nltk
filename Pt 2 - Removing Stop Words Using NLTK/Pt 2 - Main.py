import nltk
# nltk.download('stopwords')

text = '''Welcome to the hell. Lets start with our first tutorial on NLTK. We shall leaern the basics of NLTK here.add()'''
demoWords = ['playing', 'hapiness', 'going', 'doing', 'yes', 'no', 'I', 'having', 'had', 'haved']

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
print(f'\nstop_words : {stop_words}\n \n----------------------\n')

from nltk.tokenize import word_tokenize, sent_tokenize
tokenize_words = word_tokenize(text)
print(f'word_tokenize : {tokenize_words}\n \n----------------------\n')

tokenize_words_without_stop_words = []
for word in tokenize_words:
    if word not in stop_words:
        tokenize_words_without_stop_words.append(word)
        
print(set(tokenize_words_without_stop_words))

print(f'stop words which got removed : {set(tokenize_words) - set(tokenize_words_without_stop_words)}\n \n----------------------\n')
print(f'tokenize words which included all the words including stop words : {tokenize_words}\n \n----------------------\n')
print(f'this is without stop words : {tokenize_words_without_stop_words}\n \n----------------------\n')



'''
    NB :
        - stopwords.words() ~> Berfungsi untuk melakukan stopwords dengan bahasa tertentu.
        - stopwords.words(<bahasa stopword>)





'''





