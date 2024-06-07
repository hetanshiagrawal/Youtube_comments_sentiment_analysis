import pandas as pd
import numpy as np
pd.set_option('display.max_colwidth', 200)
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import warnings
warnings.filterwarnings("ignore")

TextBlob("The movie is good").sentiment

comm = pd.read_csv('../../Desktop/SGP/SGP PROJECT - Sheet5.csv')

# Calculating the Sentiment Polarity
pol = []  # list which will contain the polarity of the comments
for i in comm.Comment.values:
    try:
        analysis = TextBlob(i)
        pol.append(analysis.sentiment.polarity)

    except:
        pol.append(0)

comm['pol']=pol
NEUTRAL = 0
POSITIVE = 1
NEGATIVE = -1
#Converting the polarity values from continuous to categorical
comm['pol'][comm.pol==0]= NEUTRAL
comm['pol'][comm.pol > 0]= POSITIVE
comm['pol'][comm.pol < 0]= NEGATIVE

#Displaying the POSITIVE comments
df_positive = comm[comm.pol==POSITIVE]
print("POSITIVE COMMENTS")
print(df_positive.head(10))

#Displaying the NEGATIVE comments
df_positive = comm[comm.pol==NEGATIVE]
print("NEGATIVE COMMENTS")
print(df_positive.head(10))

#Displaying the NEUTRAL comments
df_positive = comm[comm.pol==NEUTRAL]
print("NEUTRAL COMMENTS")
print(df_positive.head(10))

mylabels = ["NEUTRAL", "POSITIVE", "NEGATIVE"]
mycolors = ["y", "g", "r"]
myexplode = [0, 0, 0.2]
y=comm.pol.value_counts()
plt.pie(y, labels = mylabels, colors = mycolors,explode = myexplode,autopct='%.1f%%')
plt.show()