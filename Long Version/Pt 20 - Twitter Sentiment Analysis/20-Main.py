from os import stat
from tweepy import Stream
from tweepy import OAuthHandler
import json
import sentiment_mod as s

# consumer key, consumer secret, access token, access secret
ckey='8oZdG4Lj3UUp7AWv1JZgU151n'
csecret='g0icqdCv9FJC21xLbSUdN15GCUS6PUY9mz5QYYz8ZDymAeYEe5'

atoken='1493952640577044486-VinZNIs2RLZUgryvstYqjVgMuh66GJ'
asecret='eOGp2QC6P8wB1EoKOFENQKjpxDT7YpBO1fdXDGx8loUEK'

class listener(Stream):
    def on_data(self, data):
        try:
            all_data = json.loads(data)
            tweet = all_data['text']
            sentiment_value, confidence = s.sentiment(tweet)
            print(tweet, sentiment_value, confidence)

            if confidence*100 >= 0:
                output = open('Data/twitter-out.txt', 'a')
                output.write(sentiment_value)
                output.write('\n')
                output.close()
            
            # time.sleep(0.2)
            return True
        except:
            return True

    def on_error(self, status):
        print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken,asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["happy"])

'''
Traceback (most recent call last):                        io
  File "d:\Coding\Python\Learn Python\Learn NLTK\Long Vers iion\Pt 20 - Twitter Sentiment Analysis\20-Main.py", line 39, in <module>                                            ke
    twitterStream = Stream(auth, listener(ckey,csecret,atoken,asecret))                                             nt
TypeError: __init__() missing 2 required positional arguments: 'access_token' and 'access_token_secret'
'''


        
        