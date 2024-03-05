import pandas as pd
from xgboost import XGBClassifier
import joblib

# Load datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
validation_data = pd.read_csv('validation.csv')

# Assuming your dataset has 'text' as feature and 'spam' as the target column
X_train, y_train = train_data['text'], train_data['spam']
X_test, y_test = test_data['text'], test_data['spam']
X_validation, y_validation = validation_data['text'], validation_data['spam']

# Convert text data to numerical format using suitable encoding (e.g., TF-IDF, CountVectorizer, etc.)

# Train XGBoost model
model = XGBClassifier()
model.fit(X_train, y_train)

# Evaluate model on validation set
accuracy = model.score(X_validation, y_validation)
print("Validation Accuracy:", accuracy)

# Save the trained model
joblib.dump(model, 'something.pkl')
