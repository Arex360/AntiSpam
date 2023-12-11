import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from datasets import load_dataset
dataset = load_dataset("FredZhang7/all-scam-spam")
messages = dataset['train']['text'][:10000] 
labels = dataset['train']['is_spam'][:10000]  
X_train, X_test, y_train, y_test = train_test_split(messages, labels, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
def LoadModel():
    svm_model = joblib.load("model.joblib")
    return svm_model
def Predict(svm_model,text):
    input_text = [text]
    transformed_text = vectorizer.transform(input_text)
    prediction = svm_model.predict(transformed_text)
    print(prediction)
    prediction = svm_model.predict(transformed_text)
    return prediction
