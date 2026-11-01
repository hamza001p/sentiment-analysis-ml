from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC



texts = []
labels = []

with open ("dataset.txt", "r") as f:
    for line in f:
        text,label = line.strip().split("\t")
        texts.append(text)
        labels.append(int(label))

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.33)


nb = MultinomialNB()
nb.fit(x_train, y_train)
print("Naive Bayes:", nb.predict(x_test))

lr = LogisticRegression()
lr.fit(x_train, y_train)
print("Logistic Regression:", lr.predict(x_test))

svm = SVC()
svm.fit(x_train, y_train)   
print("SVM:", svm.predict(x_test))

sample = vectorizer.transform(["The movie was bad"])
print("NB:", nb.predict(sample))
print("LR:", lr.predict(sample))
print("SVM:", svm.predict(sample))