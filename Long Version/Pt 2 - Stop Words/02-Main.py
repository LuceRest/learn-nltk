from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_setence = 'This is an example showing off stop word filtration'
stop_words = set(stopwords.words('english'))
print(f'stop_words : {stop_words}\n \n----------------------\n')

words = word_tokenize(example_setence)
print(f'words : {words}\n \n----------------------\n')


filtered_setence = []

for w in words:
    if w not in stop_words:
        filtered_setence.append(w)

filtered_setence = [w for w in words if not w in stop_words]

print(f'filtered_setence : {filtered_setence}\n \n----------------------\n')



'''
    NB :
        - Stop Words disebut juga Filtering
        - Filtering → Tahap pemilihan kata-kata penting dari hasil token, yaitu katakata apa saja yang akan digunakan untuk mewakili dokumen

        - stopwords.words() ~> Berfungsi untuk mengambil data stopwords dengan bahasa tertentu.
        - stopwords.words(<bahasa stopword>)
        

'''

