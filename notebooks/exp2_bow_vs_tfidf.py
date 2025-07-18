# imports
import re
import string

import mlflow
import numpy as np
import pandas as pd
import mlflow.sklearn

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# dagshub integrations
import dagshub
dagshub.init(repo_owner='datta-abhi', repo_name='mlops-mini-project', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/datta-abhi/mlops-mini-project.mlflow")
mlflow.set_experiment("BOW vs TFIDF models")

# load data
df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv').drop(columns=['tweet_id'])
print(df.head(3))

# define preprocessing functions
def lemmatization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(df):
    """Normalize the text data."""
    try:
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(removing_punctuations)
        df['content'] = df['content'].apply(removing_urls)
        df['content'] = df['content'].apply(lemmatization)
        return df
    except Exception as e:
        print(f'Error during text normalization: {e}')
        raise

# Normalize the test data
df = normalize_text(df)

# taking only sadness and happiness
df = df[df['sentiment'].isin(['happiness','sadness'])]
df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x=='happiness' else 0)
print(df['sentiment'].mean())

# define vectorizers
vectorizers = {"bow":CountVectorizer(),
               "tfidf": TfidfVectorizer()}

# define algorithms
algos = {"log_reg":LogisticRegression(),
         "naive_bayes": MultinomialNB(),
         "rf": RandomForestClassifier(),
         'xgb':XGBClassifier(),
         "grad_boost":GradientBoostingClassifier()}

# mlflow runs start here
with mlflow.start_run(run_name="All Experiments") as parent:
    # loop over vectorizer
    for vec_name,vectorizer in vectorizers.items():
        # loop over algorithm
        for algo, algorithm in algos.items():
            with mlflow.start_run(run_name= f"{vec_name}_{algo}_exp", nested=True) as child:
              
              # training code
              X = vectorizer.fit_transform(df['content'])
              y = df['sentiment']
              X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
              
              model = algorithm
              model.fit(X_train,y_train)
              
              # evaluation code
              y_pred = model.predict(X_test)
              accuracy = accuracy_score(y_test, y_pred)
              precision = precision_score(y_test, y_pred)
              recall = recall_score(y_test, y_pred)
              f1 = f1_score(y_test, y_pred)
              
              # mlflow logging  
              
              # log params
              mlflow.log_params({"vectorizer":vec_name,
                       "algorithm": algo,
                       "test_size":0.2,
                       })
              
              if algo == "log_reg":
                  mlflow.log_param("C",model.C)
              if algo == "naive_bayes":
                  mlflow.log_param("alpha",model.alpha)    
              if algo == "rf":
                  mlflow.log_params({'n_estimators': model.n_estimators,
                                     'max_depth': model.max_depth})
              if algo == "xgb":
                  mlflow.log_params({'n_estimators': model.n_estimators,
                                     'learning_rate': model.learning_rate,
                                     'max_depth': model.max_depth})
              if algo == "grad_boost":
                  mlflow.log_params({'n_estimators': model.n_estimators,
                                     'learning_rate': model.learning_rate,
                                     'max_depth': model.max_depth})
                      
              # log metrics
              mlflow.log_metrics({"accuracy":accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1})
              
              # log code
              mlflow.log_artifact(__file__)
              
              # log model
              mlflow.sklearn.log_model(model,"model")
    
              # log tags
              mlflow.set_tags({"author":"Abhigyan"})
              
              # print for verification
              print(f"{vec_name}_{algo}_exp")
              print(f"Accuracy: {accuracy}")
              print(f"Precision: {precision}")
              print(f"Recall: {recall}")
              print(f"F1 Score: {f1}")
              print('--'*50)
              
              




