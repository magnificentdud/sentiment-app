# Sentiment Analysis – Movie Review Classifier

A machine learning project that classifies movie reviews as positive or negative using NLP techniques.

## Results
- Accuracy: 89.55% on 10,000 held-out reviews
- Trained on 50,000 IMDB movie reviews

## Tech Stack
- Python
- pandas – data loading and cleaning
- scikit-learn – TF-IDF vectorization and Logistic Regression
- Streamlit – web app UI

## How It Works
1. Raw reviews are cleaned (lowercased, HTML tags and punctuation removed)
2. Text is converted to numbers using TF-IDF vectorization (top 10,000 words)
3. A Logistic Regression model is trained on 80% of the data
4. Predictions are made on the remaining 20%, achieving 89.55% accuracy

## Project Structure
- data.csv: IMDB 50K movie reviews dataset (not included, too large)
- analysis.ipynb: Jupyter notebook with data exploration and model training
- app.py: Streamlit web app
- requirements.txt: Python dependencies

## Run Locally
pip3 install pandas scikit-learn streamlit
streamlit run app.py

Note: Requires data.csv from the IMDB dataset on Kaggle:
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
