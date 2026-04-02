# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 20:54:14 2025

@author: joshc
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df=pd.read_excel('\\Users\\joshc\\OneDrive\\Documents\\Project\\TestData.xlsx')
df.drop(['FTR','HTR','HF','AF','HTAG','HTHG','SeasonIndex','H2H_MatchCount','Not a Derby','HC','AC','AY','HY','HR','AR','H2H_HomeWins','B365D','B365H','B365A','B365>2.5','B365<2.5','H2H_AwayWins','H2H_Draws','Date','HomeTeam','AwayTeam'],axis=1,inplace=True)

corr=df.corr()
sns.heatmap(corr,annot=False,cmap='coolwarm')

plt.show()