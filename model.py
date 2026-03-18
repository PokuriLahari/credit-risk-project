import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("data/train_u6lujux_CVtuZ9i.csv")

# -------------------------
# HANDLE MISSING VALUES
# -------------------------
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean(), inplace=True)

for col in ['Gender','Married','Dependents','Self_Employed','Credit_History']:
    df[col].fillna(df[col].mode()[0], inplace=True)

# -------------------------
# ENCODE CATEGORICAL DATA
# -------------------------
le = LabelEncoder()

cols = ['Gender','Married','Education','Self_Employed',
        'Property_Area','Loan_Status','Dependents']

for col in cols:
    df[col] = le.fit_transform(df[col].astype(str))

# -------------------------
# FEATURES & TARGET
# -------------------------
X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = df['Loan_Status']

# -------------------------
# TRAIN TEST SPLIT
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# -------------------------
# MODEL 1: DECISION TREE
# -------------------------
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

print("\nDecision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

# -------------------------
# MODEL 2: RANDOM FOREST
# -------------------------
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

print("\nRandom Forest Accuracy:", accuracy_score(y_test, rf.predict(X_test)))

# -------------------------
# MODEL 3: LOGISTIC REGRESSION
# -------------------------
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

print("\nLogistic Regression Accuracy:", accuracy_score(y_test, lr.predict(X_test)))