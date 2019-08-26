# update libs
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import re
import warnings
from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
import sys
import sklearn
import os
os.system("pip install scikit-learn -U")
os.system("pip install sklearn -U")
os.system("pip install nltk -U")

print(sklearn.__version__)
# import libraries

nltk.download('punkt')
nltk.download('stopwords')


warnings.filterwarnings(action='ignore', category=FutureWarning)


def load_data(database_filepath):
    database_filepath = 'sqlite:///' + database_filepath
    engine = create_engine(database_filepath)
    df = pd.read_sql_table("InsertTableName", engine)
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    """Normalize, tokenize and stem text string.
    
    Input:
    text: string. String containing the message for processing.
       
    Output:
    A list that contains normalized and stemmed word tokens.
    """
    # Convert text to lowercase and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize words
    tokens = word_tokenize(text)
    # Stem word tokens and remove stop words
    stemmer = PorterStemmer()
    stop_words = stopwords.words("english")
    result = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return result


def build_model():
    parameters = {
        "clf__estimator__n_estimators": [10, 20, 30],
        "clf__estimator__max_depth": [5, 8, 12],
        "clf__estimator__min_samples_leaf": [3, 5, 8]
    }

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    cv = GridSearchCV(pipeline, parameters, cv=5)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    y_test = np.array(Y_test)
    y_test_pred = model.predict(X_test)

    columns = ["Feature", "Accuracy", "F1-Score", "Precision", "Recall"]
    results = []

    for i in range(len(category_names)):
        report = pd.DataFrame(classification_report(
            y_test[:, i], y_test_pred[:, i], output_dict=True))
        result_temp = [category_names[i], report["accuracy"]
                       [0], report["0"][0], report["0"][1], report["0"][2]]
        results.append(result_temp)

    report = pd.DataFrame(results, columns=columns)
    return report


def save_model(model, model_filepath):
    import pickle
    pickle.dump(model, open(model_filepath, 'wb'))
    return True


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

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
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
