import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Read the dataset into a DataFrame
df = pd.read_csv("Social_Network_Ads.csv")

# Replace 'Gender' column values with numeric codes (0 for 'Male', 1 for 'Female')
df["Gender"].replace({"Male": 0, "Female": 1}, inplace=True)
print(df)

# Display column names of the DataFrame
print(df.columns)

# Define features (x) and target (y)
x = df[['User ID', 'Gender', 'Age', 'EstimatedSalary']]  # Features
y = df['Purchased']  # Target variable

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=29)

# Initialize and train a Logistic Regression model
model = LogisticRegression()
model.fit(x_train, y_train)

# Use the trained model to make predictions on the test data
y_predict = model.predict(x_test)
print(y_predict)

# Calculate and print the accuracy score on the training set
print(model.score(x_train, y_train))

# Calculate and print the accuracy score on the entire dataset
print(model.score(x, y))

# Compute and print the confusion matrix
cm = confusion_matrix(y_test, y_predict)
print(cm)

# Extract values from the confusion matrix
tp, fp, fn, tn = confusion_matrix(y_test, y_predict).ravel()
print(tp, fp, fn, tn)

# Calculate and print the accuracy score
a = accuracy_score(y_test, y_predict)
print(a)

# Print the error rate
print(1 - a)

# Calculate and print the precision score
print(precision_score(y_test, y_predict))

# Calculate and print the recall score
print(recall_score(y_test, y_predict))
