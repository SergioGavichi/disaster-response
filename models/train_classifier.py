import sys
import pandas as pd
from sqlalchemy import create_engine
import pickle
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

def load_data(database_filepath):
    '''
    Load data from database file and split the data into X and Y
    Args:
        database_filepath - path to database file
    Returns:
        X - Messages text
        Y - Messages categories
        category_names - list categories
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Messages', engine)
    X = df.message.values
    Y = df.iloc[:,4:].values
    category_names = list(df.columns)[4:]
    return X, Y, category_names

def tokenize(text):
    '''
    Tokenization function to process text data
    Args:
        text - Text to tokenize
    Returns:
        clean_tokens - Tokenized, lemmatized, normalized and striped text
    '''
     # tokenize text
    tokens = word_tokenize(text)
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    '''
    Build pipeline and grid search parameters
    Args:
        None
    Returns:
        cv - a GridSearchCV object
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])

    parameters = {
        #'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75),
        'vect__max_features': (None, 5000),
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4],
        'clf__estimator__max_depth': [5,10],
        # for multioutput The “balanced” mode uses the values of y to automatically adjust weights
        # associated with classes inversely proportional to class frequencies in the input data
        'clf__estimator__class_weight':['balanced']
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Test the model and report the f1 score, precision and recall for each output category
    Args:
        model - trained model to test
        X_test - test messages
        Y_test - test categories of test messages
        category_names - list of messages categories
    Returns:
        Print classification report for each category
    '''
    Y_pred = model.predict(X_test)
    # iterate through the columns (36 cataegories) and call sklearn's classification_report on each.
    for cat in range(Y_pred.shape[1]):
        print(category_names[cat])
        print(classification_report(Y_test[:, cat], Y_pred[:, cat]))

def save_model(model, model_filepath):
    '''
    Export model as pickle filepath
    Args:
        model - trained model to export
        model_filepath - path to target pickle file
    Returns:
        None
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
