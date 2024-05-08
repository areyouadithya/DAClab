import pandas as pd
import numpy as np
import seaborn as sn
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler,PolynomialFeatures
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


irisdf=pd.read_csv("iris.csv")
print(irisdf.describe())
print(irisdf.shape[0])#to print no of rows


diabdf=diabdf.dropna()
diabdf.rename(columns={'Outcome':'Yes/No'}, inplace=True)
print(diabdf.columns)
print(irisdf.columns)


irisdf=irisdf.drop(columns='Id')
sn.pairplot(irisdf, hue="Species", markers=["o", "s", "D"])
sn.pairplot(diabdf,hue='Yes/No', markers=["o", "s"])

iris_mean=irisdf.drop(columns='Species')
iris_mean=iris_mean.mean()
print(round(iris_mean,2))


scaler=MinMaxScaler()
diabdf['DiabetesPedigreeFunction']=scaler.fit_transform(diabdf[['DiabetesPedigreeFunction']])
print(diabdf)

def remove_outliers_boxplot(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_no_outliers = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_no_outliers

for column in columns_to_remove_outliers:
    diabdf = remove_outliers_boxplot(diabdf, column)
plt.figure(figsize=(12, 6))
sn.boxplot(data=diabdf[columns_to_remove_outliers])
plt.title('Boxplots after removing outliers')
plt.show()
print(diabdf)

def scale_dataframe(df):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(data=scaled_data, columns=df.columns)
    return scaled_df

scaled_diabetes_df = scale_dataframe(diabdf.drop(columns='Yes/No'))
scaled_iris_df = scale_dataframe(irisdf.drop(columns='Species'))

iris_corr = irisdf.drop(columns='Species').corr()
plt.figure(figsize=(10, 8))
sn.heatmap(iris_corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap - Iris Dataset')
plt.show()

def estimate_coef(x, y):
    n = np.size(x)
    m_x = np.mean(x)
    m_y = np.mean(y)

    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x

    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
    return (b_0, b_1)
df['total'] = df['TV'] + df['radio'] + df['newspaper']
b,a=estimate_coef(df['total'],df['sales'])
print(b,a)


plt.scatter(df['total'], df['sales'], color='m', marker='o', s=30, label='Original data')
y_pred = b + a * df['total']
plt.plot(df['total'], y_pred, color='g', label=f'Regression Line: y = {a:.4f}x + {b:.4f}')

plt.xlabel('Total Advertising Budget (TV + radio + newspaper)')
plt.ylabel('Sales')
plt.legend()
plt.show()

x=df[['total']]
y=df['sales']
X_train, X_test, y_train, y_test = train_test_split(x ,y, test_size=0.3, random_state=45)
degrees = range(3, 11)
min_mse = float('inf')
best_degree = None
best_coefficients = None
all_coeff = []
all_rmse = []
all_r2 = []
all_mse=[]

for degree in degrees:
    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred = model.predict(X_test_poly)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    all_coeff.append(model.coef_)
    all_rmse.append(rmse)
    all_r2.append(r2)
    all_mse.append(mse)

    if mse < min_mse:
        min_mse = mse
        best_degree = degree
        best_coefficients = model.coef_

print(f"Best Polynomial Degree: {best_degree}")
print(f"Coefficients for Best Polynomial (Degree {best_degree}):\n", best_coefficients)
print("All Coefficients:", all_coeff)
print("All RMSE:", all_rmse)
print("All R2:", all_r2)
print("All MSE:", all_mse)




x_range = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1, 1)
plt.figure(figsize=(12, 8))
plt.scatter(X_test, y_test, color='m', marker='o', s=30, label='Test data')
for degree in degrees:
    poly_features = PolynomialFeatures(degree=degree)
    X_range_poly = poly_features.fit_transform(x_range)
    y_range_pred = np.dot(X_range_poly, all_coeff[degree - 3])
    plt.plot(x_range, y_range_pred, label=f'Degree {degree}')

plt.xlabel('Total Advertising Budget (TV + radio + newspaper)')
plt.ylabel('Sales')
plt.legend()
plt.show()




iris = datasets.load_iris()
X = iris.data[:, 2:3]  # Using petal length as the feature
y = iris.data[:, 3]    # Using petal width as the target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.scatter(X_test, y_test, label='Actual')
plt.plot(X_test, y_pred, 'r-', label='Predictions', linewidth=2)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend()
plt.show()
y_pred = model.predict(X_test)
print(r2_score(y_test,y_pred))

# sn.boxplot(data=diabdf)
# sn.heatmap(df_corr,annot=True,cmap='coolwarm',linewidths=0.5)
# sn.pairplot(irisdf, hue="Species", markers=["o", "s", "D"])
# plt.plot(x,y,color='',lable='')
# plt.scatter(x,y,color='',marker='',label='')

#dec tree

model = DecisionTreeClassifier(criterion='entropy')
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print(cm)
print(classification_report(y_test,y_pred))
plt.figure(figsize=(12,8))
plot_tree(model)

#knn
k = 3
knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(X, y)
new_data_point = np.array([[9, 20]]) 
predicted_class = knn_model.predict(new_data_point)
print("Predicted class:", predicted_class[0])


#pca

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.title('PCA of Wine Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Class')
plt.grid(True)
plt.show()

#random forest

random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_classifier.fit(X_train, y_train)
y_pred = random_forest_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

#svm

svm_classifier = SVC(kernel='linear', C=1.0)
svm_classifier.fit(X_train, y_train)
y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)



data = {'category': ['A', 'B', 'C', 'A', 'B']}
df = pd.DataFrame(data)
one_hot_encoded = pd.get_dummies(df['category'], prefix='category')