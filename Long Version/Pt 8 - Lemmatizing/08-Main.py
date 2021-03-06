from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize('better', pos='a'))

print(lemmatizer.lemmatize('best', pos='a'))

print(lemmatizer.lemmatize('run'))

print(lemmatizer.lemmatize('run', pos='v'))



'''
    NB :
        - Lemmatization → Proses yang bertujuan untuk melakukan normalisasi pada teks/kata dengan berdasarkan pada bentuk dasar yang merupakan bentuk lemma-nya.
        - Lemma         → Bentuk dasar dari sebuah kata yang memiliki arti tertentu berdasar pada kamus.

        - WordNetLemmatizer().lemmatize()   ~> Berfungsi untuk melakukan lemmatizer pada suatu kata.
        - WordNetLemmatizer().lemmatize(<word>, "<pos>")
            - Nilai default untuk pos       → pos=”n” (noun)

'''

