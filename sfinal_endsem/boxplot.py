import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import numpy as np


data  = pd.read_csv("iris.csv")
new_data = data.drop("variety", axis=1)

q1 = new_data.quantile(0.25)
q3 = new_data.quantile(0.75)
iqr = q3 - q1
lowerbound = q1 - 1.5*iqr
upperbound = q3 + 1.5*iqr
outliers = [i for i in data if i < lowerbound or i > upperbound]

if outliers:
    print(outliers)
else: 
    print("No Outliers")

sns.boxplot(data = data["variety"])
plt.show()

# result = data.describe()
# print(result)

# data_new = data.drop(columns={"variety"})
# corr_matrix = data_new.corr()
# print(corr_matrix)

# cov_matrix = data_new.cov()
# print(cov_matrix)

# # sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# sns.heatmap(corr_matrix, annot = True, cmap = "coolwarm", fmt = ".2f")
# plt.title('Correlation between Features in Iris Dataset')
# plt.show()

# sns.heatmap(cov_matrix, annot = True, cmap = "coolwarm", fmt = ".2f")
# plt.title('Covariance between Features in Iris Dataset')
# plt.show()  

