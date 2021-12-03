import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scipy import stats
from scipy.stats import boxcox

# Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

#ML
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso,LinearRegression,Ridge,ElasticNetCV,RidgeCV,LassoCV
from sklearn.metrics import r2_score



# read dataset and simple information
train=pd.read_csv('/Users/ganxin/Downloads/house-prices-advanced-regression-techniques/train.csv')
test=pd.read_csv('/Users/ganxin/Downloads/house-prices-advanced-regression-techniques/test.csv')
print(train.head())
print(train.shape)
print(train.info())

#data cleaning
#let's deal with missing value(NaNs in the dataset)
#plt.figure(figsize=(15,8))
#sns.heatmap(train.isnull(),yticklabels=False, cbar = False, cmap="viridis")     #missing value can be shown as yellow
#plt.show()

#I will directly drop the colomns that the missing value consist of more than 20%
missing_val=pd.DataFrame(train.isnull().sum()[train.isnull().sum()!=0].sort_values(ascending=False)).rename(columns={0:'num_miss'})
missing_val['missing_perc'] = (missing_val/train.shape[0]).round(1)
missing_val = missing_val.query('missing_perc>0.2')
#print(missing_val)
train.drop(columns=['Id','PoolQC','MiscFeature','Alley','Fence','FireplaceQu'],axis=1,inplace=True)
#print(train.shape)
#plt.figure(figsize=(15,8))
#sns.heatmap(train.isnull(),yticklabels=False, cbar = False, cmap="viridis")     #missing value can be shown as yellow
#plt.show()


#now we need to divide the dataset into two groups(numerical features and categorical features)
num_cols=train.select_dtypes(include=['number'])
cat_cols=train.select_dtypes(include=['object'])

# convert year cols to number of years since
mask=num_cols.columns.str.contains('^Year|Yr')
year_cols=num_cols.loc[:,mask].copy()
num_cols[year_cols.columns]=year_cols.apply(lambda x:2021-x)
train[year_cols.columns]=year_cols.apply(lambda x:2021-x)


#next we are going to fill in the NAs
#print(num_cols.isna().sum())
num_cols.MasVnrArea.hist(bins = 50)
num_cols.GarageYrBlt.hist(bins = 50)
a = num_cols.GarageYrBlt.mean()
num_cols.LotFrontage.hist(bins = 50)
b = num_cols.LotFrontage.median()
num_cols.MasVnrArea.fillna(0, inplace = True)
num_cols.GarageYrBlt.fillna(a, inplace = True)
num_cols.LotFrontage.fillna(b, inplace = True)
train.MasVnrArea.fillna(0, inplace = True)
train.GarageYrBlt.fillna(a, inplace = True)
train.LotFrontage.fillna(b, inplace = True)

#category variables NAs
cat_cols_missing=cat_cols.columns[cat_cols.isnull().any()]                #return index
#print(cat_cols_missing)
imputer=SimpleImputer(missing_values=np.NaN,strategy='most_frequent')
for feature in cat_cols_missing:
     cat_cols[feature] = imputer.fit_transform(cat_cols[feature].values.reshape(-1,1))
     train[feature] = imputer.fit_transform(train[feature].values.reshape(-1,1))
print(cat_cols.nunique().sort_values(ascending = False))
le=LabelEncoder()                   #label encoding(one hot encoding/count encoding/target encoding?)any difference?
for feature in cat_cols.columns:
    cat_cols[feature]=le.fit_transform(cat_cols[feature])
    train[feature]=le.fit_transform(train[feature])

#now deal with the target value
#plt.figure(figsize=(12,4))
#plt.suptitle("Testing the skewness of the target variable")
# Distribution Plot
#plt.subplot(1,2,1)
#sns.histplot(train["SalePrice"],stat="density",kde=True)
#plt.title('Distribution Plot')
# Probability Plot
#plt.subplot(1,2,2)
#stats.probplot(train['SalePrice'],plot=plt)
#plt.tight_layout()
#plt.show()
#plt.clf()

#we can see that there is skewness,a box-cox transform is need
target=train['SalePrice']
train['SalePrice'],lambda0=boxcox(target,lmbda=None,alpha=None)
#plt.figure(figsize=(12,4))
#plt.suptitle("After box-cox")

# Distribution Plot
#plt.subplot(1,2,1)
#sns.histplot(train["SalePrice"],stat="density",kde=True)
#plt.title('Distribution Plot')

# Probability Plot
#plt.subplot(1,2,2)
#stats.probplot(train['SalePrice'],plot=plt)
#plt.tight_layout()
#plt.show()          #now better
#plt.clf()

train_1 = train

#finally stantardlize the training data
train_std1 = (train_1-train_1.mean(axis=0)/train_1.std(axis=0))
#After data preprocessing showed above, we try to do regression now.
Y = train_std1['SalePrice']
X = train_std1.iloc[:,:-1]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
linreg = LinearRegression()
linreg.fit(X_train,Y_train)
print("Training set score:{:.2f}".format(linreg.score(X_train,Y_train)))
print("Test set score:{:.2f}".format(linreg.score(X_test,Y_test)))
print("the intercept:",linreg.intercept_)
coeff_df=pd.DataFrame(linreg.coef_,X.columns,columns=['Coefficient'])
print("the coefficient:\n",coeff_df)
pred=linreg.predict(X_test)        #bad

#let's try LASSO and ridge
#try different parameters
pd.set_option('display.max_columns', None)
result=pd.DataFrame(columns=["parameter","lasso training set scores","lasso testing set scores","ridge training set scores","ridge testing set scores"])
for i in range(1,100):
    alpha=i/10
    ridge=Ridge(alpha=alpha)
    lasso=Lasso(alpha=alpha,max_iter=10000)
    ridge.fit(X_train,Y_train)
    lasso.fit(X_train,Y_train)
    result=result.append([{"parameter":alpha,"lasso training set scores":r2_score(Y_train,lasso.predict(X_train)),"lasso testing set scores":r2_score(Y_test,lasso.predict(X_test)),"ridge training set scores":r2_score(Y_train,ridge.predict(X_train)),"ridge testing set scores":r2_score(Y_test,ridge.predict(X_test))}])
print(result)
#use cv to find the best parameter
ridge_model = RidgeCV()
ridge = ridge_model.fit(X_train, Y_train)
print("ridge training set scores："+str(r2_score(Y_train,ridge_model.predict(X_train))))
print("ridge testing set scores："+str(r2_score(Y_test,ridge_model.predict(X_test))))
lasso_model=LassoCV(eps=1e-8, max_iter=100)
lasso = lasso_model.fit(X_train,Y_train)
print("Lasso training set scores："+str(r2_score(Y_train,lasso_model.predict(X_train))))
print("Lasso testing set scores："+str(r2_score(Y_test,lasso_model.predict(X_test))))
coeff_lasso=pd.DataFrame(lasso.coef_,X.columns,columns=['Coefficient'])
print("the coef:\n",coeff_lasso)


#this means we do need more data cleaning
#In our model, I will select only those variables whose correlation to the target is more than 0.4
num_corr_price=num_cols.corr()['SalePrice'][:-1]
#for i in range(0, len(num_cols.columns), 5):
#    sns.pairplot(data=num_cols, x_vars=num_cols.columns[i:i+5], y_vars=['SalePrice'])
#plt.show()
best_features=num_corr_price[abs(num_corr_price)>0.4].sort_values(ascending=False)
for feature in best_features.index:
    num_corr_price.drop(feature,inplace=True)
for feature in num_corr_price.index:
    train.drop(feature,axis=1,inplace=True)
    num_cols.drop(feature,axis=1,inplace=True)
#print(train.shape)

#Besides, we should notice that some variables are highly related. I will delete one of a pair whose correlation is more than 0.7
num_corr=num_cols.corr()
corr_triu=num_corr.where(np.triu(np.ones(num_corr.shape),k=1).astype(bool))
#plt.figure(figsize=(10,10))
#sns.heatmap(num_corr,annot=True,square=True,fmt='.2f',annot_kws={'size':9},mask=np.triu(corr_triu),cmap="coolwarm")
#plt.show()
corr_triu_collinear = corr_triu.iloc[:-1,:-1]
collinear_features = [column for column in corr_triu_collinear.columns if any(corr_triu_collinear[column]>0.7)]
train.drop(columns=collinear_features,inplace=True)
num_cols.drop(columns=collinear_features,inplace=True)
#print(train.shape)


#then delete outliers
#for i in range(0,len(num_cols.columns),1):
#    plt.figure(figsize=(5,5))
#    sns.boxplot(data=num_cols.iloc[:,i])
#plt.show()
train=train.drop(train.MasVnrArea.sort_values(ascending = False)[:1].index)
train=train.drop(train.TotalBsmtSF.sort_values(ascending = False)[:1].index)
train=train.drop(train.GrLivArea.sort_values(ascending = False)[:2].index)
train.reset_index(drop=True,inplace=True)
#print(train.shape)

#now let's deal with category variables
#plt.figure(figsize=(15,15))
#sns.heatmap(cat_cols.corr(),square=True,mask=np.triu(cat_cols.corr()),cmap="coolwarm")
#plt.show()

#correlated variaties
cat_corr=cat_cols.corr()
cat_corr_triu=cat_corr.where(np.triu(np.ones(cat_corr.shape),k=1).astype(bool))
cat_collinear_features=[column for column in cat_corr_triu.columns if any(cat_corr_triu[column]>0.75)]
train.drop(columns=cat_collinear_features,inplace=True)
cat_cols.drop(columns=cat_collinear_features,inplace=True)
#print(train.head())
#print(train.isna().sum().sort_values(ascending=False))
#let's deal with MasVnrArea
#print(train.MasVnrArea.describe())
train.MasVnrArea.fillna(0,inplace=True)     #use median
#print(train.shape)





#After data preprocessing showed above, we try to do regression now.
train_std = (train-train.mean(axis=0)/train.std(axis=0))
Y = train_std['SalePrice']
X = train_std.iloc[:,:-1]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
linreg = LinearRegression()
linreg.fit(X_train,Y_train)
print("Training set score:{:.2f}".format(linreg.score(X_train,Y_train)))
print("Test set score:{:.2f}".format(linreg.score(X_test,Y_test)))
print("the intercept:",linreg.intercept_)
coeff_df=pd.DataFrame(linreg.coef_,X.columns,columns=['Coefficient'])
print("the coefficient:\n",coeff_df)
pred=linreg.predict(X_test)


#visualize
#plt.scatter(Y_test, pred)
#plt.ylabel('Y')
#plt.xlabel('X')
#plt.ylim([min(pred), max(pred)])
#plt.xlim([min(Y), max(Y)])
#plt.plot(np.arange(-100, 100, 0.1), np.arange(-100, 100, 0.1), color='red')
#plt.grid()
#plt.show()

#LASSO and ridge
#use cv to find the best parameter
ridge_model = RidgeCV()
ridge = ridge_model.fit(X_train, Y_train)
print("ridge training set scores："+str(r2_score(Y_train,ridge_model.predict(X_train))))
print("ridge testing set scores："+str(r2_score(Y_test,ridge_model.predict(X_test))))
lasso_model=LassoCV(eps=1e-9, max_iter=100)
lasso = lasso_model.fit(X_train,Y_train)
print("Lasso training set scores："+str(r2_score(Y_train,lasso_model.predict(X_train))))
print("Lasso testing set scores："+str(r2_score(Y_test,lasso_model.predict(X_test))))
coeff_lasso=pd.DataFrame(lasso.coef_,X.columns,columns=['Coefficient'])
print("the coef:\n",coeff_lasso)


#ElasticNet Regression
elasticnet_model = ElasticNetCV()
elasticnet_model.fit(X_train, Y_train)
print("ElasticNet training set scores："+str(r2_score(Y_train,elasticnet_model.predict(X_train))))
print("ElasticNet testing set scores："+str(r2_score(Y_test,elasticnet_model.predict(X_test))))


#randomforestregression
rfor=RandomForestRegressor()
rfor.fit(X_train,Y_train)
print("Training Score:%f"%rfor.score(X_train,Y_train))
print("Testing Score:%f"%rfor.score(X_test,Y_test))

#adaboost
adab=AdaBoostRegressor()
adab.fit(X_train,Y_train)
print("Training Score:%f"%adab.score(X_train,Y_train))
print("Testing Score:%f"%adab.score(X_test,Y_test))

#BaggingRegressor
bag = BaggingRegressor()
bag.fit(X_train, Y_train)
print("Training Score:%f"%bag.score(X_train,Y_train))
print("Testing Score:%f"%bag.score(X_test,Y_test))

#GradientBoostingRegressor
gra = GradientBoostingRegressor()
gra.fit(X_train, Y_train)
print("Training Score:%f"%gra.score(X_train,Y_train))
print("Testing Score:%f"%gra.score(X_test,Y_test))