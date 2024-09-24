#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pandas openpyxl numpy matplotlib seaborn scikit-learn


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[16]:


import pandas as pd
data = pd.read_csv('heart_health.csv')
data.head()


# In[17]:


data.shape


# In[14]:



categorical_columns = ['HeartDiseaseorAttack', 'HighBP', 'HighChol', 'CholCheck', 'Smoker', 
                       'Stroke', 'Diabetes', 'PhysActivity', 'Fruits', 'Veggies', 
                       'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 
                       'Sex', 'GenHlth', 'Age', 'Education', 'Income']


data[categorical_columns] = data[categorical_columns].astype('category')

print(data.isnull().sum())


# In[18]:



data.describe()

plt.figure(figsize=(10, 6))
sns.histplot(data['BMI'], kde=True, bins=30)
plt.title('BMI Distribution')
plt.show()
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# In[19]:



plt.figure(figsize=(10, 6))
sns.boxplot(x='HeartDiseaseorAttack', y='BMI', data=data)
plt.title('Heart Disease vs BMI')
plt.show()

sns.countplot(x='PhysActivity', hue='HeartDiseaseorAttack', data=data)
plt.title('Physical Activity vs Heart Disease')
plt.show()


# In[20]:



data['BMI_Category'] = pd.cut(data['BMI'], bins=[0, 18.5, 24.9, 29.9, np.inf], 
                              labels=['Underweight', 'Normal weight', 'Overweight', 'Obesity'])


# In[21]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


X = data.drop(['HeartDiseaseorAttack'], axis=1)
y = data['HeartDiseaseorAttack']


X = pd.get_dummies(X, drop_first=True)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[25]:


# Save cleaned data
data.to_csv('cleaned_heart_disease_data.csv', index=False)


# UNIVARIATE ANALYSIS
# 

# In[29]:


import matplotlib.pyplot as plt
import seaborn as sns

variables = ['Age', 'BMI', 'MentHlth', 'PhysHlth']

plt.figure(figsize=(20, 10))
for i, var in enumerate(variables):
    plt.subplot(2, 2, i + 1)
    sns.histplot(data[var], bins=30, kde=True)  
    plt.title(f'Distribution of {var}')
    plt.xlabel(var)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[30]:



conditions = ['HighBP', 'HighChol', 'Stroke', 'Diabetes', 'HeartDiseaseorAttack']

plt.figure(figsize=(15, 10))
for i, cond in enumerate(conditions):
    plt.subplot(2, 3, i + 1)
    sns.countplot(x=data[cond], palette='Set2')
    plt.title(f'Prevalence of {cond}')
    plt.xlabel(cond)
    plt.ylabel('Count')

plt.tight_layout()
plt.show()


# In[31]:



high_bp_counts = data['HighBP'].value_counts()
labels = ['No High BP', 'High BP']

plt.figure(figsize=(6, 6))
plt.pie(high_bp_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
plt.title('Prevalence of High Blood Pressure')
plt.show()


# In[32]:


plt.figure(figsize=(6, 6))
sns.countplot(x=data['HeartDiseaseorAttack'], palette='Set1')
plt.title('Distribution of Heart Disease Cases')
plt.xlabel('Heart Disease (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()


# In[33]:



heart_disease_counts = data['HeartDiseaseorAttack'].value_counts()
labels = ['No Heart Disease', 'Heart Disease']

plt.figure(figsize=(6, 6))
plt.pie(heart_disease_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'orange'])
plt.title('Distribution of Heart Disease Cases')
plt.show()


# BIVARIATE ANALYSIS

# In[40]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
sns.boxplot(x='HeartDiseaseorAttack', y='HighBP', data=data)
plt.title('Distribution of High Blood Pressure by Heart Disease Status')
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x='HeartDiseaseorAttack', y='HighChol',data=data)
plt.title('Distribution of High Cholesterol by Heart Disease Status')
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x='HeartDiseaseorAttack', y='BMI',data=data)
plt.title('Distribution of BMI by Heart Disease Status')
plt.show()


# In[42]:



corr_matrix = data[['BMI', 'MentHlth', 'PhysHlth', 'Age']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Continuous Variables')
plt.show()


# In[ ]:




