# Kaggge-Titanic<br/>	
Kaggle_Titanic解答<br/>	

导入需要的库
```
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
```
导入数据:[数据下载点击](https://www.kaggle.com/c/titanic/data)
```
x = pd.read_csv("train.csv") 
```
数据预处理
```
print (x.describe())#数据描述性统计分析

x["Age"]=x["Age"].fillna(x["Age"].median())#将Age空值用平均数填满

#print (x["Sex"].unique())#查看Sex属性分类，结果为0/1分类
x.loc[x["Sex"]=="male",'Sex']=0#将字符串转化为0/1分类
x.loc[x["Sex"]=="female",'Sex']=1#将字符串转化为0/1分类

#print (x["Embarked"].unique())#查看Embarked属性分类，结果为3种分类
print (pd.Series.value_counts(x['Embarked']))#查看Embarked属性分类哪种占比最高
x["Embarked"]=x["Embarked"].fillna('S')#用占比最高的分类将空值填满
x.loc[x["Embarked"]=="S",'Embarked']=0#将字符串转化为数字分类
x.loc[x["Embarked"]=="C",'Embarked']=1#将字符串转化为数字分类
x.loc[x["Embarked"]=="Q",'Embarked']=2#将字符串转化为数字分类
```

**解答1.线性回归**<br/>	
```
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

predictors = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']#自变量
alg=LinearRegression() #estimator选取模型

kf=KFold(n_splits=3,shuffle=False,random_state=1)#训练集分为3份，轮流2份用来train，1份用来测试
predictions=[]

for train,test in kf.split(x):
    #用每一次的2份train data来做模型参数：
    train_predictors=(x[predictors].iloc[train,:])
    train_targets=(x['Survived'].iloc[train])
    alg.fit(train_predictors,train_targets)
    #用上述模型参数，去拟合每一次的1份test data自变量，输出预测结果
    test_predictions=alg.predict(x[predictors].iloc[test,:])
    predictions.append(test_predictions)

predictions=np.concatenate(predictions, axis=0)#将三次预测结果粘合成一维向量

predictions[predictions>.5]=1#将预测结果优化为0/1值
predictions[predictions<=.5]=0#将预测结果优化为0/1值
AUC=sum(predictions==x['Survived'])/len(predictions)#将预测结果和原有分类作比对
print(AUC)#输出精确度
```
**解答2.逻辑回归**<br/>	
```
from sklearn.linear_model import LogisticRegression

alg=LogisticRegression(random_state=1) #estimator选取模型
scores=sklearn.model_selection.cross_val_score(alg,X=x[predictors],y=x['Survived'],cv=3) #用sklearn自带工具直接输出每轮精确度
print(scores.mean()) #将三轮精确度取平均值
```
