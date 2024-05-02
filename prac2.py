#1
import pandas as pd

# 2/3 Load the dataset (assuming 'exams.csv' is in the same directory)
df = pd.read_csv('exams.csv')

# 4 Data Preprocessing

# Check for missing values
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)

# Get initial statistics
description = df.describe()
print("Summary Statistics:\n", description)

# Variable types
variable_types = df.dtypes
print("Variable Types:\n", variable_types)

df.info()

# Data Formatting and Normalization (Example)

# Convert "writing score" to integer (assuming it's numerical)
df["writing score"] = df["writing score"].astype(int)

# Handle missing values (after type conversion)
df = df.dropna()

# Describe the updated DataFrame
df.describe()


