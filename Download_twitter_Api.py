from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream


# Keys for log in to Twitter API
consumer_key = ''
consumer_secret = ''
access_token = ''
access_secret = ''


# Will write the tweets to the file in append mode
class StdOutListener(StreamListener):
    def on_data(self, data):
        # Opening data file in append mode
        with open('data/tweetdata.txt','a') as tf:
            tf.write(data)
        print(data)
        return True

    def on_error(self, status):
        print(status)


if __name__ == '__main__':
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    stream = Stream(auth, l)

    stream.filter(track=['depression', 'anxiety', 'mental health', 'suicide', 'stress', 'sad','sadness'])