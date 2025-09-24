# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('Customer-Churn.csv')

# Display the first few rows
print("Dataset Sample:\n", df.head())

# Drop missing values (optional: you could also fill them instead)
df = df.dropna()

# Drop identifier columns if present (example: customer_id, account number, etc.)
if "customer_id" in df.columns:
    df = df.drop(columns=["customer_id"])

# Automatically detect categorical columns (object or string types)
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

# Exclude target column 'churn' from encoding
categorical_cols = [col for col in categorical_cols if col != "churn"]

# One-hot encode categorical columns
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Define features (X) and target (y)
X = df_encoded.drop("churn", axis=1)
y = df_encoded["churn"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
model = LogisticRegression(max_iter=1000)  # increase iterations for convergence
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=1)

print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:\n", report)
