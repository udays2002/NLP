import pandas as pd
m = pd.read_csv('/Users/siddharthsharma/.spyder-py3/SMSSpamCollection',sep='\t',names=['label','message'])
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wordnet=WordNetLemmatizer()

corpus = []
for i in range(len(m)):
    review = re.sub('[^a-zA-Z]', ' ', m['message'][i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X=cv.fit_transform(corpus).toarray()
y=pd.get_dummies(m['label'])
y=y.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_test,X_train,Y_test,Y_train=train_test_split(X,y,test_size=0.20,random_state=0)

from sklearn.naive_bayes import MultinomialNB
spam_detect=MultinomialNB().fit(X_train,Y_train)
y_predict=spam_detect.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_m=confusion_matrix(Y_test, y_predict)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(Y_test, y_predict)
