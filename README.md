# Disaster Response Pipeline Project
This is the repository for the Disaster Response Pipeline project of the Udacity Data Scientist NanoDegree.
1. We first build an **ETL pipeline** `process_data.py` that loads, transforms and saves the data into a database file
2. We then build an **ML pipeline** `train_classifier.py` that uses different transformers and a classifier optimized by a grid search
3. We build a Flask web application (backend and frontend) to display graphs on training data and propose the user to enter text message to be classified.

Libraries used Scikit-Learn and nltk for NLP, sqlalchemy for SQL engine, Pandas for data manipulation and pickle.

### Project files structure
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process
|- disaster_messages.csv  # data to process
|- process_data.py
|- DisasterResponse.db   # database to save clean data to

- models
|- train_classifier.py
|- disasterClassifier.pkl  # saved model

- README.md

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves it into a pickle file.
      *Careful here, it might takes some time to train. You might want to review GridSearchCV parameters or just reuse the trained classifer pkl file in this repo*
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:3001/ or http://localhost:3001/
