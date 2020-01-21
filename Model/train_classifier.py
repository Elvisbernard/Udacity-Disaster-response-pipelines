import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import warnings
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
warnings.filterwarnings('ignore')

def load_data(database_filepath):
    '''     
    load the data from a sqlite database  
    
    input:
    database_filepath:  string of filepath of the database
    
    output:
    X: messages to be classified
    Y:  categories
    category_name: label for the different categories
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    engine.table_names()
    df = pd.read_sql_table('Messages', con=engine)
    X = df['message']
    Y = df.drop(['id','message', 'original', 'genre'], axis=1)
    names = Y.columns.tolist()
    return X, Y, names

def tokenize(text):
    '''
    a tokenizer function that return a tokenized version of a text
    
    input:
    text:   a string conytaining the text to tokenize
    
    output:
    tokens: a list of token representing the input
    
    '''
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    '''
    build a pipeline model
    
    output: a gridsearch object with a pipeline containing the tokenization, tdfidf transformation and a random forest classifier
    '''
    clf = MultiOutputClassifier(RandomForestClassifier())

    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', clf)
                        ])
    parameters = {'tfidf__norm': ['l2', 'l1'],
             'vect__ngram_range': [(1, 1),(1,2)],
             'clf__estimator__max_depth': [None,2,4],
             'clf__estimator__max_features': ['sqrt', 'log2']}


    cv = GridSearchCV(pipeline, parameters, scoring='f1_weighted')
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate the performance of the model with F1, precision and recall
    
    
    input:
    model: the model to evaluate
    X_test: message from test set
    Y_test: category value from test set
    category_names: list of labels for the categories

    '''
    new_pred=model.predict(X_test)
    print(classification_report(Y_test, new_pred, target_names=category_names))
    


def save_model(model, model_filepath):
    '''
    save the model 
    
    input:
    model:  model to save
    model_filepath name of the file where to save the model
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()