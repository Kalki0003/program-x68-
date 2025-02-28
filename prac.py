PRACTICAL 3A 
class Node: 
    def __init__(self, n, v): 
        self.n, self.v, self.children = n, v, [] 
 
def hill_climb(node): 
    while True: 
        best = max(node.children, key=lambda n: n.v, default=None) 
        if not best or best.v <= node.v: 
            return node 
        node = best 
 
A, B, C, D, E = Node("A", 3), Node("B", 5), Node("C", 8), Node("D", 6), Node("E", 2) 
A.children = [B, C] 
B.children = [D, E] 
 
result = hill_climb(A) 
print(f"Hill Climbing Result: {result.n} with value: {result.v}")

Practical 3B 
def alpha_beta(depth, idx, max_play, values, alpha=float('-inf'), beta=float('inf')): 
    if depth == 0: return values[idx] 
     
    val = float('-inf') if max_play else float('inf') 
    for i in range(2): 
        child_val = alpha_beta(depth - 1, idx * 2 + i, not max_play, values, alpha, beta) 
        if max_play: 
            val = max(val, child_val) 
            alpha = max(alpha, val) 
        else: 
            val = min(val, child_val) 
            beta = min(beta, val) 
        if beta <= alpha: break 
    return val 
 
values = [3, 5, 6, 9, 1, 2, 0, -1] 
print("Result:", alpha_beta(3, 0, True, values)) 
 
Practical 3C 
v1 = ["A", "B", "C", "D", "E", "F", "G"] 
const=[("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), 
    ("B", "E"), ("C", "E"), ("C", "F"), ("D", "E"), 
    ("E", "F"), ("E", "G"), ("F", "G")] 
 
def backtrack(assign): 
    if len(assign)==len(v1): return assign 
    for var in (v for v in v1 if v not in assign): 
        for values in ['Monday','Tuesday','Wednesday']: 
            new_assign={**assign,var:values} 
            if all(new_assign.get(x)!=new_assign.get(y) for x,y in const if x in new_assign and y in 
new_assign): 
                result=backtrack(new_assign) 
                if result: return result 
 
print("Backtracking CSP Solution:", backtrack({})) 
 
 
Practical 4 
import pandas as pd,seaborn as sns,matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from keras.models import Sequential 
from keras.layers import Dense 
from keras.optimizers import Adam 
df = pd.read_csv('D:\TYCS-515\AI\Iris.csv') 
print(df.columns, df.head()) 
sns.pairplot(df, hue='Species') 
plt.show() 
x = df.iloc[:, 1:5].values 
y = pd.get_dummies(df['Species']).values 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0) 
model = Sequential([ 
Dense(10, input_shape=(4,), activation='relu'), 
Dense(3, activation='softmax') 
]) 
model.compile(optimizer=Adam(0.01), loss='categorical_crossentropy', metrics=['accuracy']) 
model.fit(x_train, y_train, epochs=50, verbose=0) 
accuracy = model.evaluate(x_test, y_test, verbose=0)[1] * 100 
print(f"Accuracy: {accuracy:.2f}%") 
Practical 4B 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.tree import DecisionTreeClassifier, plot_tree 
from sklearn.metrics import accuracy_score 
data = pd.read_csv('D:\\TYCS-515\\AI\\diabetes.csv') 
x = data.iloc[:, :-1] 
y = data.iloc[:, -1] 
X = StandardScaler().fit_transform(x) 
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 
t = DecisionTreeClassifier(max_depth=3, random_state=0) 
t.fit(x_train, y_train) 
accuracy = accuracy_score(y_test, t.predict(x_test)) * 100 
print(f"Accuracy: {accuracy:.2f}%") 
plt.figure() 
plot_tree(t, feature_names=data.columns[:-1]) 
plt.show() 
Practical 5 
import pandas as pd 
from pgmpy.models import BayesianNetwork 
from pgmpy.estimators import MaximumLikelihoodEstimator 
from pgmpy.inference import VariableElimination 
data = pd.read_csv('D:\TYCS-515\AI\HeartDeasis.csv') 
model = BayesianNetwork([ 
('age', 'Lifestyle'),  
('Gender', 'Lifestyle'),  
('Family', 'heartdisease'), 
('diet', 'cholestrol'),  
('Lifestyle', 'diet'),  
('cholestrol', 'heartdisease') 
]) 
model.fit(data, estimator=MaximumLikelihoodEstimator) 
evidence = {} 
for var in ['age', 'Gender', 'Family', 'diet', 'Lifestyle', 'cholestrol']: 
evidence[var] = int(input(f'Enter {var}: ')) 
inference = VariableElimination(model) 
res = inference.query(variables=['heartdisease'], evidence=evidence) 
print(res) 
Practical 6B 
from sklearn.preprocessing import LabelEncoder 
weather = ['sunnny','sunny','overcast','rainy','rainy','rainy', 
'overcast','sunny','sunny','rainy','sunny','overcast','overcast','rainy'] 
temp = ['hot','hot','hot','mild','cool','cool','cool','mild','cool','mild','mild','mild','hot','mild'] 
play = ['no','no','yes','yes','yes','no','yes','no','yes','yes','yes','yes','yes','no'] 
le = LabelEncoder() 
weather_encode = le.fit_transform(weather) 
temp_encode = le.fit_transform(temp)   
label = le.fit_transform(play) 
print(f"Weather: {weather_encode}\nTemp: {temp_encode}\nPlay: {label}") 
Practical 7A 
import numpy as np 
percep = lambda x, w, b: int(np.dot(w, x) + b >= 0) 
for x in [[0, 0], [0, 1], [1, 0], [1, 1]]: 
print(f"OR{x}={percep(x, [1, 1], -0.5)} AND{x}={percep(x, [1, 1], -1.5)}") 
Practical 7B 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from keras.models import Sequential 
from keras.layers import Dense 
data = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians
diabetes.data.csv', names=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 
'DiabetesPedigreeFunction', 'Age', 'Outcome']) 
X = StandardScaler().fit_transform(data.iloc[:, :-1]) 
y = data.iloc[:, -1] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
model = Sequential([ 
Dense(8, input_dim=8, activation='relu'), 
Dense(1, activation='sigmoid') 
]) 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=0) 
accuracy = model.evaluate(X_test, y_test, verbose=0)[1] * 100 
print(f"Accuracy: {accuracy:.2f}%")
