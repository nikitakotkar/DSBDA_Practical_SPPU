import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Load the Titanic dataset
df = sns.load_dataset("titanic")

# Handle potential inf values (optional)
# df["fare"] = pd.to_numeric(df["fare"], errors='coerce')  # Convert to numeric, coerce inf to NaN

# Create a histogram of the fare prices
sns.histplot(data=df, x="fare", kde=True)

# Add a title and labels
plt.title("Distribution of Fare Prices for Titanic Passengers")
plt.xlabel("Fare (USD)")
plt.ylabel("Density")

# Show the plot
plt.show()
