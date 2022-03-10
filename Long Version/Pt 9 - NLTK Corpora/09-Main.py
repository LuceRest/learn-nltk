from random import sample
from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize

sample = gutenberg.raw('bible-kjv.txt')

tok = sent_tokenize(sample)

print(tok[5:15])



'''
    NB :
        - Korpus                → Kumpulan teks autentik, baik tulis maupun transkrip percakapan dalam jumlah besar yang disimpan secara elektronik.
        - Kelebihan korpus adalah mudah untuk diakses dan analisis berbasis korpus bisa dibuat generalisasi secara kuantitatif.
        - Korpus data           → Data yang dipakai sebagai sumber bahan penelitian.
        - Korpus (linguistik)   → Kumpulan ujaran yang tertulis atau lisan yang digunakan untuk menyokong atau menguji hipotesis tentang struktur bahasa.

'''

