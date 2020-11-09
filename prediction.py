import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import os

tweets_data = []
x = []
y = []
vectorizer = CountVectorizer(stop_words='english')

# Will retrieve the tweets data from the JSON File
def retrieve_tweet(data_url):
    tweets_data_path = data_url
    tweets_file = open(tweets_data_path, "r")
    for line in tweets_file:
        try:
            tweet = json.loads(line)
            tweets_data.append(tweet)
        except:
            continue

# Will retrieve the processed score which contains the polarity as well
def retrieve_processed(url):
    sentence = pd.read_excel(url)
    for i in range(len(tweets_data)):
        if tweets_data[i]['id'] == sentence['id'][i]:
            x.append(tweets_data[i]['text'])
            y.append(sentence['sentiment'][i])

def Naive_Bayes():
    from sklearn.naive_bayes import MultinomialNB
    X_train_Naive = vectorizer.fit_transform(x)
    actual_naive = y
    model = MultinomialNB()
    model.fit(X_train_Naive, [int(r) for r in y])
    test_features_Naive = vectorizer.transform(x)
    predictions_naive = model.predict(test_features_Naive)

    # ROC Curve will return us the FPR, TPR and Threshold
    fpr_naive, tpr_naive, threshold_naive = metrics.roc_curve(actual_naive, predictions_naive, pos_label=1)
    accuracy_naive = format(metrics.auc(fpr_naive, tpr_naive))
    accuracy_naive = float(accuracy_naive) * 100
    os.system("tput setaf 7")
    print("Accuracy of Naive Bayes Model: ", accuracy_naive, "%")

def Decision_Tree():
    from sklearn import tree
    X_train_Tree = vectorizer.fit_transform(x)
    actual_Tree = y
    model = tree.DecisionTreeClassifier()
    model.fit(X_train_Tree, [int(r) for r in y])
    test_features_Tree = vectorizer.transform(x)
    predictions_Tree = model.predict(test_features_Tree)
    fpr_tree, tpr_tree, threshold_tree = metrics.roc_curve(actual_Tree, predictions_Tree, pos_label=1)
    accuracy_tree = format(metrics.auc(fpr_tree, tpr_tree))
    accuracy_tree = float(accuracy_tree) * 100
    os.system("tput setaf 7")
    print("Accuracy of Decision Tree Model: ", accuracy_tree, "%")

def Random_Forest():
    from sklearn.ensemble import RandomForestClassifier
    X_train_RF = vectorizer.fit_transform(x)
    actual_RF = y
    test_features_rf = vectorizer.transform(x)
    model = RandomForestClassifier(max_depth=2, random_state=0)
    model = model.fit(X_train_RF, [int(i) for i in y])
    prediction_rf = model.predict(test_features_rf)
    fpr_rf, tpr_rf, thresholds_rf = metrics.roc_curve(actual_RF, prediction_rf, pos_label=1)
    accuracy_rf = format(metrics.auc(fpr_rf, tpr_rf))
    accuracy_rf = float(accuracy_rf) * 100
    print("Accuracy of Random Forest Model: ", accuracy_rf, "%")

def main():
    retrieve_tweet('data/tweetdata.txt')
    retrieve_processed('processed_data/output.xlsx')
    Naive_Bayes()
    Decision_Tree()
    Random_Forest()

def best_model_for_input(input_tweet):
    from sklearn import tree
    X_train_Tree = vectorizer.fit_transform(x)
    model = tree.DecisionTreeClassifier()
    model.fit(X_train_Tree, [int(r) for r in y])
    tweet = vectorizer.transform([input_tweet])
    predict_sentiment = model.predict(tweet)
    if predict_sentiment == 1:
        predict_sentiment = "You are Happy! Great!!"
    elif predict_sentiment == 0:
        predict_sentiment = "You are Neutral!!"
    elif predict_sentiment == -1:
        predict_sentiment = "You are sad or depressed!!"
    else:
        print("Nothing")

    print(predict_sentiment)

main()

os.system("tput setaf 9")
input_tweet = input("Enter the tweet if the person is happy or not: ")
best_model_for_input(input_tweet)
