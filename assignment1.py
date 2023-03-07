#import packages
import requests  
import re  
import pandas as pd  

import numpy as np   
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction.text import TfidfVectorizer
#install and import tweepy package, An easy-to-use Python library for accessing the Twitter API.
import tweepy as tw
#Authentication is handled by the tweepy.AuthHandler class
from tweepy import OAuthHandler

from nltk.corpus import stopwords
from nltk import trigrams
import string
import nltk
from nltk.stem.porter import *
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
                                             #--------------#Twitter API#---------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#

#storing consumer_key and consumer_secret in a variable and authenticating it
consumer_key = 'dX4d4JmzkSzcziaPuc0WsYbVo'
consumer_secret = 'KxVRPhrBJ7lpHRafVnmrUpZiXBbiu1ao9OlyrBUp3y1XNsUtMz'
#An Access Token and Secret are user-specific credentials used to authenticate OAuth 1.0a API requests.
#They specify the Twitter account the request is made on behalf of.
access_token = '1564695880518602752-Ma6X6YIP3VIcYmfb8IhWocUte4UPJV'
access_token_secret = 'In6zBnt6H973fNO4uatU2YMi4mVsB4XhCgQDzccxhZk5y'
#authenticating consumer keys and consumer secret
auth = tw.OAuthHandler(consumer_key, consumer_secret)
#set and get access
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth,wait_on_rate_limit=True)
#defining a dataframe
topics1=["Bitcoin", "Tether", "Dogecoin","Polkadot","Ethereum"]
df = pd.DataFrame()
for topic in topics1:

    try:
#In order to perform pagination we must supply a page/cursor parameter with each of our requests.
#api.search_tweets allows us to search for tweets with respect to the following parameters
        tweets=tw.Cursor(api.search_tweets,q=topic +" -filter:retweets",lang='en',count=10 ).items(100)
#storing the required objects from twitter API into the dataframe
        for tweet in tweets:
            df = df.append({'LABEL': topic,'Text': tweet._json['text']}, ignore_index=True)
    except BaseException as e:
        print('failed on_status,',str(e))
df.shape
print(df)



df.to_csv("/Users/rika/Documents/TM/Crypto_twitter.csv")


#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
                                             #--------------#News API#---------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#

#decide on the topics and list it
topics=["Bitcoin", "Tether","Dogecoin", "polkadot"]

#creating a .csv file to store the articles 
filename="/Users/rika/Documents/TM/Crypto.csv"
MyFILE2=open(filename,"w")
column_names="LABEL,Headline\n"
MyFILE2.write(column_names)
MyFILE2.close()
#itertate through for each topic
#endpoint of the API
endpoint="https://newsapi.org/v2/everything"
for topic in topics:
    #give in the api key and needed parameters
    URLPost = {'apiKey':'6209a3fea75e4b818c79ff9501155c00',
               'q':topic, 'searchln': "description"
    }
    #request for json
    response=requests.get(endpoint, URLPost)
    print(response)
    jsontxt = response.json()
    print(jsontxt)
    MyFILE2=open(filename, "a")
    LABEL=topic
    for items in jsontxt["articles"]:
        print(items, "\n\n\n")
        Headline=items["description"]
        Headline=str(Headline)
        Headline=Headline.replace(',', '')
        Headline=Headline.replace('.', '')
        #write in the column names for teh csv file
        WriteThis=str(LABEL)+"," + str(Headline) + "\n"
        print(WriteThis)
    
        MyFILE2.write(WriteThis)
    
## CLOSE THE FILE
MyFILE2.close()
news_df = pd.read_csv("/Users/rika/Documents/TM/Crypto.csv")
tem_df = news_df.copy()
news_df = news_df.loc[news_df['LABEL'].isin(["Bitcoin","Tether","Dogecoin", "polkadot"])]
print(news_df.isnull().sum())
news_df['LABEL'] = news_df['LABEL'].apply(lambda x: x.lower())
print(news_df)
#news_df = news_df.drop("Unnamed: 0",axis=1)
news_df['Headline'].replace('', np.nan, inplace=True)
print(news_df.info())


df_TW = pd.read_csv("/Users/rika/Documents/TM/Crypto_twitter.csv", index_col=0)
temp_df1 = df_TW.copy()
df_TW = df_TW.drop("Unnamed: 0",axis=1)
df_TW.columns = ['LABEL', 'Headline']
df_TW['LABEL'] = df_TW['LABEL'].apply(lambda x: x.lower())
print(df_TW)

df_TW = df_TW.reset_index(drop=True)
print(df_TW)

df_final = news_df.append(df_TW)
print(df_final)


df_final.to_csv("/Users/rika/Documents/TM/crypto_final.csv")
df_final = pd.read_csv("/Users/rika/Documents/TM/crypto_final_final.csv", index_col=False) 

#L = [x for x in tem_df['Headline'] for x in trigrams(x.split())]

#print(L)
#-------------#wordcloud#---------------#
text = " ".join(i for i in df_final.Headline)
wordcloud = WordCloud(background_color="white").generate(text)
plt.figure( figsize=(10,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Wordcloud before cleaning')
plt.show()


#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
                                             #--------------#Data Cleaning#---------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#

#removing patterns

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, ' ', input_txt)
    return input_txt   

# removing twitter handles (@user)
df_final['Headline'] = np.vectorize(remove_pattern)(df_final['Headline'], "@[\w]*#")
#removing urls    
df_final['Headline'] = np.vectorize(remove_pattern)(df_final['Headline'], "https?://[A-Za-z./]'*")

#removing special characters, numbers, punctuations
df_final['Headline'] = df_final['Headline'].str.replace("[^a-zA-Z]", " ")

df_final['Headline'] = df_final['Headline'].str.replace("https", "")
    #converting all the words to lowercase 
df_final['Headline'] = df_final['Headline'].str.lower()


#finally removing stopper words from the text column
stop = stopwords.words('english')+['ethereum','bitcoin', 'doge', 'crypto','cryptocurrency','icerket','bitcoins', 'bitcoinist', 'bitcoiners','dogecoin','polkadot','tether','the', 'go', 'get', 'see', 'take', 'thing', 'like', 'one', 'say','via','san']
#extra=["said","bitcoin", 'dogecoin','polkadot','tether','the', 'go', 'get', 'see', 'take', 'thing', 'like', 'one', 'say','via','san']
#stop.append(extra)
#print(stop)
#print(type(stop))
df_final['Headline'] = df_final['Headline'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]) )
#removing words of length lesser than 3 and greater  than 10 
df_final['Headline'] = df_final['Headline'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3 and len(w)<15])) 

#changing all the words to lower case to make it easily readable and avoid messy representation of data

#removing meaningless words in the text column
#words = set(nltk.corpus.words.words())


#def clean_sent(sent):
 #   return " ".join(w for w in nltk.wordpunct_tokenize(sent)
  #   if w.lower() in words or not w.isalpha())

#df_final['Headline'] = df_final['Headline'].apply(clean_sent)

df_final["Headline"] = df_final["Headline"].astype(str).apply(lambda x: ''.join(x))


#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
                                          #------------------#Stemmer#---------------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#


Stem_df = df_final.copy(deep=True)
#removing all rows in the Stem_df DataFrame where the value in the Headline column is an empty string.
Stem_df = Stem_df[Stem_df.Headline!= '']
#creating a PorterStemmer object, which is a tool for stemming words.
A_STEMMER=PorterStemmer()
#performing stemming on each word in the Headline column of the Stem_df DataFrame.
Stem_df['Headline'] = Stem_df['Headline'].apply(lambda x: ' '.join([A_STEMMER.stem(word) for word in x.split()]))

print(Stem_df.head(10))
df_final = df_final[df_final.Headline!= '']


#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
                                          #------------------#Lemmatization#---------------------#
#--------------#############--------------#--------------############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#


lem_df = df_final.copy(deep=True)
lemmatizer = WordNetLemmatizer()    # Instantiate our Stemmer object
lem_df['Headline'] = lem_df['Headline'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
lem_df = lem_df[lem_df.Headline!= '']
print(lem_df.head(10))
df_final["Headline"].values.tolist()
df_final.columns.values.tolist()
print(df_final)

lem_df.to_csv(r'/Users/rika/Documents/TM/dense_csv.csv',index=False)

#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
                                          #-------------#countvectorizer#--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#

#countvectorize to create words frequency array
mycv1 = CountVectorizer(input = 'content', stop_words = 'english', max_features=3000)
mymat = mycv1.fit_transform(lem_df['Headline'])
mycols = mycv1.get_feature_names()
mymat = mymat.toarray()
mydf=pd.DataFrame(mymat, columns = mycols)
print(mydf)
mydf.columns.values.tolist()

mydf.to_csv(r'/Users/rika/Documents/TM/clust_cv.csv',index=False)


#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
                                          #-------------#TFIDF#--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#tfidf to create words frequency array
vect = TfidfVectorizer(stop_words='english', max_features=3000)

X = vect.fit_transform(lem_df.pop('Headline')).toarray()

mydf1 = pd.DataFrame(X, columns=vect.get_feature_names())
mydf1.columns.values.tolist()

mydf1.to_csv(r'/Users/rika/Documents/TM/clust_tfidf.csv',index=False)


#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
                                         #--------------#DVisualization after cleaning#---------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#

text = " ".join(i for i in df_final.Headline)
wordcloud = WordCloud(background_color="white").generate(text)
plt.figure( figsize=(10,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Wordcloud after cleaning')
plt.show()

#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
                                         #--------------#Visualization after cleaning#---------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#

df_final['transaction'] = df_final['Headline'].str.strip('()').str.split(' ')
#exploding the dataframe to create transaction based data

label = df_final['LABEL']


df_transaction= pd.DataFrame(df_final['transaction'].tolist())

#df_transaction.insert (0, label)
#some values had just '#' so removing it and replacing with 'None'
df_transaction.replace({'#': None},inplace =True, regex= True)
#removing index and column names from the data and storing it in a csv file
df_transaction.to_csv(r'/Users/rika/Documents/TM/crypto_transaction.csv',
header=None,index=False)
df_transaction.to_csv(r'/Users/rika/Documents/TM/crypto_transaction_nolabel.csv',
header=None,index=False)


#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
                                         #--------------#Visualization after cleaning#---------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#











