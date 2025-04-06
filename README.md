# Sentiment-Analysis

### Install Required Libraries
Before starting, you need to install the necessary Python libraries:
pandas: For handling and processing structured data (like Excel files).
scikit-learn: A machine learning library used for data preprocessing, model training, and evaluation.
openpyxl: Helps read and write Excel files.

### Import Necessary Libraries
After installation, you import libraries that help in:
Managing data (pandas).
Cleaning text using regular expressions (re).
Removing common words like "the" or "is" (stopwords from sklearn).
Splitting data into training and testing sets.
Converting text into numerical format using TF-IDF (a technique that assigns importance to words).
Training a Naive Bayes classifier, which is a machine learning model well-suited for text classification.
Evaluating model performance using accuracy and classification reports.

### Load IMDB Dataset
The dataset (IMDB movie reviews) is stored in an Excel file.
If using Google Colab, you must upload the file manually.
Once uploaded, the dataset is loaded into a pandas DataFrame for processing.

### Convert Sentiment to Numeric Labels
Since machine learning models work with numbers, you need to convert the sentiment labels into numerical values:
"positive" → 1
"negative" → 0
This makes it easier for the model to learn patterns in the data.

### Preprocess the Text Data
Text data needs cleaning to remove unnecessary elements and make it more useful for analysis.
Convert text to lowercase to maintain consistency.
Remove HTML tags (if present).
Remove punctuation and special characters, keeping only words.
Split text into words (tokenization).
Remove common stopwords like "the," "is," and "and" to focus on meaningful words.
After this step, the dataset will have a new column with cleaned reviews.

### Split Data into Training & Testing Sets
The dataset is split into two parts:
Training set (80%) – Used to train the model.
Testing set (20%) – Used to evaluate model performance.
Splitting ensures the model learns from one portion of the data and is tested on unseen data.

### Convert Text to Numeric Format (TF-IDF)
Since machine learning models work with numbers, text needs to be transformed into a numerical representation.
TF-IDF (Term Frequency-Inverse Document Frequency) assigns importance to words based on how often they appear in reviews.
It converts the text into a matrix of numbers where each word gets a numerical value based on its relevance.
This step helps the model understand the textual data mathematically.

### Train the Machine Learning Model (Naive Bayes)
A Naive Bayes classifier is chosen for training because it's efficient for text-based classification.
The model learns from the training data by identifying patterns in word usage.
Once trained, it can classify new reviews as either positive or negative based on the learned patterns.

### Make Predictions & Evaluate Performance
The trained model is tested on the unseen test data.
It predicts whether each review is positive or negative.
The model's accuracy is measured by comparing predictions with the actual sentiments.
A classification report is generated, showing precision, recall, and F1-score, which help analyze how well the model performed.

# Output

![Image](https://github.com/user-attachments/assets/cb465f19-1799-4055-95cb-3dfcc7db79e3)
