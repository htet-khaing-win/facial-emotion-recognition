import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_PATH = os.path.join("data", "fer2013.csv")

#Loading Dataset

df = pd.read_csv(DATA_PATH)
print("Dataset shape: ", df.shape)
print(df.head())  # first 5 rows

#Check emotion distribution

emotion_counts = df["emotion"].value_counts()
print("Class distribution: \n", emotion_counts)

#Plot Distribution

plt.figure(figsize=(8,6))
sns.barplot(x=emotion_counts.index, y=emotion_counts.values)
plt.title("Emotion Distribution in FER")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.show()