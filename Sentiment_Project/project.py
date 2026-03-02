# STEP 1 — Import Libraries

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# this only needs to be run once to download the necessary NLTK resources
# nltk.download('stopwords')
# nltk.download('wordnet')


# STEP 2 — Load Dataset

data = pd.read_csv(
r"C:\Users\devan\OneDrive\Desktop\Sentiment_Project\sentimentdataset.csv",
encoding="latin-1"
)

print("Dataset Loaded Successfully")

print(data.head())


# STEP 3 — Clean Text

stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()


def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(r'http\S+','',text)

    text = re.sub(r'[^a-zA-Z ]','',text)

    words = text.split()

    words = [w for w in words if w not in stop_words]

    words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)



data['Cleaned'] = data['text'].apply(clean_text)


print("\nCleaned Text:")

print(data[['text','Cleaned']].head())


# STEP 4 — Remove Empty Rows

data = data.dropna()

data = data[data['Cleaned']!=""]


# STEP 5 — TF-IDF

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(data['Cleaned'])

y = data['sentiment']


print("\nTF-IDF Shape:")

print(X.shape)


# STEP 6 — Train Test Split

X_train,X_test,y_train,y_test = train_test_split(

X,y,

test_size=0.2,

random_state=42

)


print("\nTraining Size:",X_train.shape)

print("Testing Size:",X_test.shape)



# STEP 7 — MODELS

print("\nTraining Models...")


# Logistic Regression

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train,y_train)

lr_pred = lr.predict(X_test)

print("\nLogistic Regression Accuracy:")

print(lr.score(X_test,y_test))



# KNN

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train,y_train)

knn_pred = knn.predict(X_test)

print("\nKNN Accuracy:")

print(knn.score(X_test,y_test))



# SVM

svm = SVC(probability=True)

svm.fit(X_train,y_train)

svm_pred = svm.predict(X_test)

print("\nSVM Accuracy:")

print(svm.score(X_test,y_test))



# STEP 8 — CONFUSION MATRIX

print("\nConfusion Matrix (SVM):")

cm = confusion_matrix(y_test,svm_pred)

print(cm)


print("\nClassification Report:")

print(classification_report(y_test,svm_pred))



# STEP 9 — ROC CURVE

classes = list(set(y))

y_test_bin = label_binarize(y_test,classes=classes)

svm_prob = svm.predict_proba(X_test)


plt.figure()

for i in range(len(classes)):

    fpr,tpr,_ = roc_curve(

    y_test_bin[:,i],

    svm_prob[:,i]

    )

    roc_auc = auc(fpr,tpr)

    plt.plot(

    fpr,

    tpr,

    label="Class "+str(i)+" area="+str(round(roc_auc,2))

    )


plt.plot([0,1],[0,1])


plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()



# STEP 10 — Accuracy Graph

lr_acc = lr.score(X_test,y_test)

knn_acc = knn.score(X_test,y_test)

svm_acc = svm.score(X_test,y_test)


models = [

"Logistic Regression",

"KNN",

"SVM"

]


accuracies = [

lr_acc,

knn_acc,

svm_acc

]


plt.figure()

plt.bar(models,accuracies)

plt.xlabel("Models")

plt.ylabel("Accuracy")

plt.title("Model Accuracy Comparison")

plt.show()



# STEP 11 — FINAL PREDICTION
# STEP 11 — FINAL PREDICTION

print("\nSentiment Prediction System Ready")
print("Type 'exit' to stop")

while True:

    review = input("\nEnter Review: ")

    # Stop condition
    if review.lower() == "exit":
        print("Program Stopped")
        break

    review_clean = clean_text(review)

    review_vector = vectorizer.transform([review_clean])

    prediction = svm.predict(review_vector)

    print("Predicted Sentiment:",prediction[0])