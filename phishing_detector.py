# Phishing Email Detection Model
# Machine Learning using Scikit-learn

import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -----------------------------------
# 1. Create a Sample Dataset
# -----------------------------------

data = {
    "email": [
        "Congratulations! You won a prize. Click here to claim now.",
        "Your bank account is blocked. Verify your password immediately.",
        "URGENT! Update your account details using this suspicious link.",
        "You have won a free gift card. Claim your reward today.",
        "Your account will be suspended. Confirm your banking details.",
        "Click this link to receive your free money now.",
        "Your password expires today. Login to verify your account.",
        "You are selected for a lottery prize. Send your details.",
        "We detected unusual activity. Verify your identity immediately.",
        "Claim your reward by clicking this link now.",

        "Hello, please find the meeting schedule attached.",
        "Your assignment submission was successful.",
        "The project meeting is scheduled for tomorrow.",
        "Thank you for your purchase. Your order is confirmed.",
        "Please review the report before the meeting.",
        "Your appointment is confirmed for Monday.",
        "Here are the notes from today's class.",
        "The monthly newsletter is available on our website.",
        "Your payment receipt is attached to this email.",
        "Please send the project report by Friday."
    ],

    "label": [
        "Phishing", "Phishing", "Phishing", "Phishing",
        "Phishing", "Phishing", "Phishing", "Phishing",
        "Phishing", "Phishing",

        "Safe", "Safe", "Safe", "Safe", "Safe",
        "Safe", "Safe", "Safe", "Safe", "Safe"
    ]
}

df = pd.DataFrame(data)

# -----------------------------------
# 2. Feature Extraction
# -----------------------------------

def extract_features(email):
    email = str(email)

    url_count = len(
        re.findall(r"https?://|www\.", email.lower())
    )

    suspicious_keywords = [
        "urgent",
        "password",
        "verify",
        "account blocked",
        "claim",
        "prize",
        "free money",
        "suspended",
        "banking details",
        "click here"
    ]

    keyword_count = sum(
        keyword in email.lower()
        for keyword in suspicious_keywords
    )

    return {
        "email_length": len(email),
        "url_count": url_count,
        "suspicious_keyword_count": keyword_count
    }


# Display extracted features
print("=" * 50)
print("PHISHING EMAIL DETECTION SYSTEM")
print("=" * 50)

print("\nExtracted Email Features:")

for i, email in enumerate(df["email"].head(5), start=1):
    print(f"\nEmail {i}: {email}")
    print("Features:", extract_features(email))


# -----------------------------------
# 3. Prepare the Dataset
# -----------------------------------

X = df["email"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# -----------------------------------
# 4. Build the ML Model
# -----------------------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

# -----------------------------------
# 5. Train the Model
# -----------------------------------

print("\nTraining the machine learning model...")

model.fit(X_train, y_train)

print("Model training completed successfully!")


# -----------------------------------
# 6. Evaluate the Model
# -----------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

# -----------------------------------
# 7. Display Confusion Matrix
# -----------------------------------

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Phishing", "Safe"]
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Phishing", "Safe"],
    yticklabels=["Phishing", "Safe"]
)

plt.title("Phishing Email Detection - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.show()


# -----------------------------------
# 8. Predict New Emails
# -----------------------------------

def predict_email(email):
    prediction = model.predict([email])[0]

    probabilities = model.predict_proba([email])[0]

    class_names = model.named_steps["classifier"].classes_

    probability_map = dict(zip(class_names, probabilities))

    confidence = probability_map[prediction] * 100

    print("\n" + "-" * 50)
    print("EMAIL ANALYSIS RESULT")
    print("-" * 50)

    print("Email:", email)

    print("Extracted Features:", extract_features(email))

    print("Prediction:", prediction)

    print(f"Model Confidence: {confidence:.2f}%")

    if prediction == "Phishing":
        print("Warning: This email may be a phishing attempt!")
    else:
        print("This email appears safe according to the model.")

    print("-" * 50)


# -----------------------------------
# 9. Test with User Input
# -----------------------------------

while True:
    print("\nEnter an email message to analyze.")
    print("Type 'exit' to stop.")

    user_email = input("\nEmail: ").strip()

    if user_email.lower() == "exit":
        print("Thank you for using the detector!")
        break

    if not user_email:
        print("Please enter a valid email message.")
        continue

    predict_email(user_email)

# Load dataset from CSV file
df = pd.read_csv("phishing_emails.csv")

# Check dataset
print(df.head())

# Check for missing values
print(df.isnull().sum())