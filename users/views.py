from django.shortcuts import render, HttpResponse
from django.contrib import messages
from django.http import JsonResponse
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from django.http import JsonResponse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import subprocess
import os
from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

# Create your views here.


def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})

def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(
                loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})

def DatasetView(request):
    path = settings.MEDIA_ROOT + "//" + 'balanced_language_dataset.csv'
    df = pd.read_csv(path)
    df = df.to_html
    return render(request, 'users/viewdataset.html', {'data': df})

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
path = settings.MEDIA_ROOT + "//" + 'balanced_language_dataset.csv'
df = pd.read_csv(path)
value_counts = df['Language'].value_counts()
X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Language'], test_size=0.2, random_state=42)
# Vectorize the tweets using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
# Train a Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

def training(request):  
    # Adding labels and title
    plt.xlabel('Categories')
    plt.ylabel('Counts')
    plt.title('Value Counts of Your Column')
    # Displaying the plot
    plt.show()    
    predictions = nb_classifier.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.2f}")      
    nb = classification_report(y_test,predictions,output_dict=True)
    nb = pd.DataFrame(nb).transpose()    
    nb = pd.DataFrame(nb)
    return render(request,"users/training.html",{'nb':nb.to_html,'acc':accuracy})

'''
a classification label, with possible values including 
Arabic (0), Kannada (1), Tamil (2), Sweedish (3), Spanish (4). Russian(5), Portugeese(6), Malayalam(7), Italian(8), Danish(9), Hindi(10), Greek(11), German(12), French(13), English(14), Dutch(15), Turkish(16).
'''

def prediction(request):
    if request.method == 'POST':
        single_tweet = request.POST.get('tweets') 
        print(single_tweet)      
        single_tweet_tfidf = tfidf_vectorizer.transform([single_tweet])
        print('manohar',single_tweet_tfidf)
        # Make prediction
        single_prediction = nb_classifier.predict(single_tweet_tfidf)
        print(single_prediction)
        # Print prediction
        print(f'Tweet: {single_tweet} - Predicted Emotion: {single_prediction[0]}')
        if single_prediction[0] == 0:
            single_prediction='Arabic'
        elif single_prediction[0] == 1:
            single_prediction='Kannada'
        elif single_prediction[0] == 2:
            single_prediction='Tamil'
        elif single_prediction[0] == 3:
            single_prediction='Sweedish'
        elif single_prediction[0] == 4:
            single_prediction='Spanish'
        elif single_prediction[0] == 5:
            single_prediction='Russian'
        elif single_prediction[0] == 6:
            single_prediction='Portugeese'
        elif single_prediction[0] == 7:
            single_prediction='Malayalam'
        elif single_prediction[0] == 8:
            single_prediction='Italian'
        elif single_prediction[0] == 9:
            single_prediction='Danish'
        elif single_prediction[0] == 10:
            single_prediction='Hindi'
        elif single_prediction[0] == 11:
            single_prediction='Greek'
        elif single_prediction[0] == 12:
            single_prediction='German'
        elif single_prediction[0] == 13:
            single_prediction='French'
        elif single_prediction[0] == 14:
            single_prediction='English'
        elif single_prediction[0] == 15:
            single_prediction='Dutch'
        elif single_prediction[0] == 16:
            single_prediction='Turkish'
        return render(request, 'users/predictForm.html', {'output':single_prediction})
    return render(request, 'users/predictForm.html', {})