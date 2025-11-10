import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv("BostonHousing.csv")




outlier_cols=['crim', 'zn', 'rm', 'b', 'medv']

def remove_outliers_iqr(data,column):
    q1,q2,q3=np.percentile(data[column],[25,50,75])
    print("q1,q2,q3 is :",q1,q2,q3)
    IQR=q3-q1
    print("IQR is:", IQR)

    lower_limit=q1-(1.5*IQR)
    upper_limit=q3+(1.5*IQR)
    data[column]=np.where(data[column]>upper_limit,upper_limit,data[column]) #capping the upper limit
    data[column]=np.where(data[column]<lower_limit,lower_limit,data[column]) #flooring the lower limit


for column in outlier_cols:
    remove_outliers_iqr(df,column)


x = df.drop('medv', axis=1)  
y = df['medv']    


normalisation=MinMaxScaler()
x=normalisation.fit_transform(x)
x=pd.DataFrame(x)
with open('scaling.pkl', 'wb') as f:
    pickle.dump(normalisation, f)

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.33)



model=GradientBoostingRegressor(n_estimators=100)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy=r2_score(y_test, y_pred)
print(accuracy)


# save the model 
pickle.dump(model,open('model.pkl','wb'))