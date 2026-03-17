import pandas as pd

# Load dataset
df = pd.read_csv("data/train_u6lujux_CVtuZ9i.csv")

# 1. Loan Approval vs Rejection
print("Loan Approval vs Rejection:")
print(df['Loan_Status'].value_counts())

# 2. Average Income
print("\nAverage Applicant Income:")
print(df['ApplicantIncome'].mean())

# Total income (important insight)
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
print("\nAverage Total Income:")
print(df['TotalIncome'].mean())

# 3. Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# 4. Handle Missing Values
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)

print("\nMissing Values After Handling:")
print(df.isnull().sum())