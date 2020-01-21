
# Disaster Response Pipeline Project
This project is a part of the Udacity Data Scientist nanodegree programme. The data were provided by Figure Eight.

The aim of this project is to create an app that takes disaster-related live message and classify them between 36 categories. the messages belonging to each of these categories could then be transfer to the  organisation taking take of that kind of information. The use of such app could save precious time during disaster and save a lot of lives.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


