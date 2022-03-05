from lib2to3.pgen2 import token
from nltk.tokenize import sent_tokenize, word_tokenize

# tokenize  -> Word tokenizer... setence tokenizers
# Lexicon and corporas
# Corpora   -> Body of the text. Ex : Medical journals, presidential speeches, English language
# Lexicon   -> Words and their means

# Investor - Speak.... regular english - Speak

# Investor speak 'bull' -> Someone who is positive about the market
# English speak 'bull'  -> Scary animal you dont want running @ you

example_text = "Hello Mr. Rest, how are you doing today? The weather is great and Python is awesome. The sky is pinkish-blue. You should not eat cardboard."

# print(sent_tokenize(example_text))
# print('\n-------------------------\n')
print(f'sent_tokenize : {sent_tokenize(example_text)}\n \n----------------------\n')


# print(word_tokenize(example_text))
# print('\n-------------------------\n')
print(f'word_tokenize : {word_tokenize(example_text)}\n \n----------------------\n')


for i in word_tokenize(example_text):
    print(i)

