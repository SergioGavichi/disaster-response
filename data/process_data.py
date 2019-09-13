import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load, merge and return a Dataframe based on new messages and categories datasets specified by the user
    Args:
        message_filepath - path to messages csv
        categories_filepath - path to categories csv
    Returns:
        df - merged Pandas Dataframe of messages and categories data
    '''
    messages = pd.read_csv(messages_filepath, encoding='latin-1')
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    '''
    Transform and clean up the loaded data set by load_data by creating new category columns and converting their values to 0 or 1
    Args:
        df - Dataframe with merged messages and categories data sets
    Returns:
        df - cleaned Dataframe
    '''
    # create a dataframe of the individual category columns
    categories = df.categories.str.split(';', expand = True)
    # select the first row of the categories dataframe...
    row = categories.iloc[0,:]
    #...and use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1:])
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # Replace categories column in df with the new category columns
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df,categories], axis=1)

    # Remove duplicates
    df.drop_duplicates(inplace = True)
    return df

def save_data(df, database_filename):
    '''
    Save the clean dataset into an sqlite database
    Args:
        df - clean dataframe to be saved
        database_filename - database file to save the data
    Returns:
        None
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('Messages', engine, index=False)


def main():
    '''
    Execute the ETL pipeline
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
