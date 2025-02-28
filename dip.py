import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
data = pd.read_csv('D:/TYCS-515/DS/DATASETS/Salary_Data.csv') 
x, y = data['YearsExperience'], data['Salary'] 
# Calculate coefficients 
B1 = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean())**2).sum() 
B0 = y.mean() - B1 * x.mean() 
# Predict and display results 
Predict = lambda new_x: B0 + B1 * new_x 
print(f'Regression line: y = {round(B0, 3)} + {round(B1, 3)}X') 
print(f'Correlation coefficient: {np.corrcoef(x, y)[0, 1]:.4f}') 
print(f'Goodness of fit (R^2): {np.corrcoef(x, y)[0, 1]**2:.4f}') 
print('Predicted salary:', Predict(70)) 
# Plot 
plt.figure(figsize=(8, 5)) 
plt.scatter(x, y, color='blue', label='Data points') 
plt.plot(x, B0 + B1 * x, color='red', label='Regression line') 
plt.title('How Experience Affects Salary') 
plt.xlabel('Years of Experience') 
plt.ylabel('Salary') 
plt.legend() 
plt.show() 
Practical 2: Logistic Regression 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, confusion_matrix 
data = pd.read_csv('D:/TYCS-515/DS/DATASETS/Admission_Data.csv') 
x = data[['gmat', 'gpa', 'work_experience']] 
y = data['admitted'] 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0) 
model = LogisticRegression().fit(x_train, y_train) 
y_pred = model.predict(x_test) 
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred)) 
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%") 
# Practical 3: Line Plot 
import seaborn as sn 
data = pd.read_csv('D:/TYCS-515/DS/DATASETS/AirPassengers.csv', parse_dates=['Month'], 
index_col='Month') 
plt.figure(figsize=(10, 5)) 
sn.lineplot(data=data, x=data.index, y='#Passengers') 
plt.title('Monthly Air Passengers') 
plt.xlabel('Month') 
plt.ylabel('Number of Passengers') 
plt.show() 
# Practical 4: Label Encoding 
from sklearn.preprocessing import LabelEncoder 
weather = ['sunny', 'sunny', 'overcast', 'rainy', 'rainy'] 
temp = ['hot', 'mild', 'cool', 'cool', 'hot'] 
play = ['no', 'yes', 'yes', 'no', 'yes'] 
le = LabelEncoder() 
weather_encoded = le.fit_transform(weather) 
temp_encoded = le.fit_transform(temp) 
label = le.fit_transform(play) 
print('Weather:', weather_encoded) 
print('Temp:', temp_encoded) 
print('Play:', label) 
from sklearn.cluster import KMeans 
# Practical 5: Clustering 
data = pd.read_csv('D:/TYCS-515/DS/DATASETS/Countryclusters.csv') 
kmeans = KMeans(3).fit(data[['Longitude', 'Latitude']]) 
data['Clusters'] = kmeans.labels_ 
plt.scatter(data['Longitude'], data['Latitude'], c=data['Clusters'], cmap='rainbow') 
plt.title('Clustering by Location') 
plt.xlabel('Longitude') 
plt.ylabel('Latitude') 
plt.show() 
from sklearn.decomposition import PCA 
# Practical 6: PCA 
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]) 
pca = PCA(n_components=2).fit(X) 
print("Explained Variance Ratio:", pca.explained_variance_ratio_) 
print("Singular Values:", pca.singular_values_) 
# Practical 7: Decision Tree 
from sklearn.tree import DecisionTreeClassifier, plot_tree 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
iris = pd.read_csv('D:/TYCS-515/DS/DATASETS/Iris.csv') 
x = iris.iloc[:, 1:5] 
y = iris['Species'] 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0) 
model = DecisionTreeClassifier(criterion='entropy', random_state=0).fit(x_train, y_train) 
print(f"Accuracy: {accuracy_score(y_test, model.predict(x_test)) * 100:.2f}%") 
plt.figure(figsize=(12, 8)) 
plot_tree(model, feature_names=x.columns, class_names=model.classes_, filled=True) 
plt.show() 
# Practical 8: Decision Tree for Diabetes Classification 
from sklearn.preprocessing import StandardScaler 
diabetes = pd.read_csv('D:/TYCS-515/DS/DATASETS/diabetes.csv') 
x = diabetes.iloc[:, :-1] 
y = diabetes.iloc[:, -1] 
x_train, x_test, y_train, y_test = train_test_split(StandardScaler().fit_transform(x), y, test_size=0.2, 
random_state=0) 
model = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=0).fit(x_train, y_train) 
print(f"Test Accuracy: {accuracy_score(y_test, model.predict(x_test)) * 100:.2f}%") 
plt.figure(figsize=(20, 10)) 
plot_tree(model, feature_names=diabetes.columns[:-1], class_names=['No Diabetes', 'Diabetes'], 
f
 illed=True) 
plt.show() 
from apyori import apriori 
# Practical 9: Apriori Algorithm 
data = pd.read_csv('D:/TYCS-515/DS/DATASETS/store_data.csv', header=None) 
records = data.applymap(str).values.tolist() 
rules = list(apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)) 
print(f"Generated {len(rules)} rules.") 
for rule in rules[:5]: 
print(rule)
