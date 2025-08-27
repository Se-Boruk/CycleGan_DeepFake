import pandas as pd
import matplotlib.pyplot as plt

#Data
df = pd.read_csv("training_log.csv")

#Plot
df.iloc[:, 1:].plot()  

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Losses Over Epochs")
plt.legend(df.columns[1:]) 


plt.show()