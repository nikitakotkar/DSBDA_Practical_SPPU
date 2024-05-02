import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset
tdata = pd.read_csv('titanic.csv')

# Handle missing values
tdata['Age'] = tdata['Age'].fillna(tdata['Age'].mean())
tdata['Cabin'] = tdata['Cabin'].fillna(tdata['Cabin'].mode()[0])
tdata['Embarked'] = tdata['Embarked'].fillna(tdata['Embarked'].mode()[0])

# Check for remaining missing values
tdata.isnull().sum()

# Create boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Sex', y='Age', hue='Survived', data=tdata, palette='Set2')
plt.title('Distribution of Age by Gender with Survival Information')
plt.show()
