import seaborn as sns
import matplotlib.pyplot as plt

# Load iris dataset
dataset = sns.load_dataset('iris')

dataset.head()
print(dataset.info())

# Plotting Create a histogram for each feature
fig, axes = plt.subplots(2, 2, figsize=(16, 9))
sns.histplot(dataset['sepal_length'].dropna(), ax=axes[0, 0])
sns.histplot(dataset['sepal_width'].dropna(), ax=axes[0, 1])
sns.histplot(dataset['petal_length'].dropna(), ax=axes[1, 0])
sns.histplot(dataset['petal_width'].dropna(), ax=axes[1, 1])
plt.show()

# Plotting Create a boxplot for each feature
fig, axes = plt.subplots(2, 2, figsize=(16, 9))
sns.boxplot(y='petal_length',x='species',data=dataset, ax=axes[0,0])
sns.boxplot(y='petal_width',x='species',data=dataset, ax=axes[0,1])
sns.boxplot(y='sepal_length',x='species',data=dataset, ax=axes[1,0])
sns.boxplot(y='sepal_width',x='species',data=dataset, ax=axes[1,1])
plt.show()

#Compare distributions and identify outliers
