import nltk
import matplotlib.pyplot as plt

# nltk.download()

text = '''Welcome to the hell. Lets start with our first tutorial on NLTK. We shall leaern the basics of NLTK here.add()'''

from nltk.tokenize import word_tokenize

word_tokenized = word_tokenize(text)
print(f'\nword_tokenized : {word_tokenized} \n----------------------\n')

from nltk.tokenize import sent_tokenize
print(f'\nsent_tokenize : {sent_tokenize(text)} \n----------------------\n')

from nltk.probability import FreqDist
fd = FreqDist(word_tokenized)
print(f'\nfd : {fd.most_common(3)} \n----------------------\n')
fd.plot(30, cumulative=False)
plt.show()



'''
    NB :
        - word_tokenize()	        ~> Berfungsi untuk memisah suatu kalimat menjadi kata-kata (dipisahkan berdasarkan spasi).
        - word_tokenize(<kalimat yg ingin dipisah>)

        - FreqDist()		        ~> Berfungsi untuk mendapatkan frekuensi distribusi dari berbagai kata atau untuk menghitung berapa banyak kata yg muncul.
        - FreqDist(<list yg berisi berbagai kata>)

        - FreqDist().most_common() 	~> Berfungsi untuk mengambil frekuensi kata yg sering muncul yg paling banyak.
        - FreqDist(<list yg berisi berbagai kata>).most_common(<jumlah yg ingin dimunculkan>)


'''

