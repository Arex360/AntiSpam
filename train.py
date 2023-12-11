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
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train)
joblib.dump(svm_model, 'model.joblib')
y_pred = svm_model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
input_text = ["Are you looking for clothes that are cheap,inbox me. Give me your credit card info"]
transformed_text = vectorizer.transform(input_text)
prediction = svm_model.predict(transformed_text)
print(prediction)
fpr, tpr, thresholds = roc_curve(y_test, svm_model.decision_function(X_test_tfidf))
roc_auc = roc_auc_score(y_test, svm_model.decision_function(X_test_tfidf))
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
#plt.show()