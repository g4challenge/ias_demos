#%% Regression
import pandas as pd
import matplotlib.pyplot as plt

data = {
    "x": [5,12,18,24,36,42,50,55,67,72],
    "y": [5,19,25,35,45,46,47,46,47,46]
}
df = pd.DataFrame(data)
df.plot(
    x="x",
    y="y",  
    kind="scatter",  
    figsize=(10,10),  
    title="Datenset"
)
# %% Lineare Regression
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()  
regressor.fit(df[['x']], df[['y']])
prediction = regressor.predict(df[['x']])

plt.figure(figsize=(10,10))  
plt.scatter(
    x=df[['x']],
    y=df[['y']],
    label="actual"
)
plt.plot(
    df[['x']],
    prediction,
    label="predicted",  
    color="orange"
)

# %% Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures  
from sklearn.linear_model import LinearRegression

poly_transformer = PolynomialFeatures(degree=5)  
poly_x = poly_transformer.fit_transform(df[['x']])  
regressor.fit(poly_x, df[['y']])
prediction = regressor.predict(poly_x)

plt.figure(figsize=(10,10))  
plt.scatter(
    x=df[['x']],  
    y=df[['y']],
    label="actual"
)
plt.plot(  
    df[['x']],
    prediction,  
    label="predicted", 
    color="orange"
)

# %% Regression Tree
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor()
regressor.fit(df[['x']], df[['y']])
prediction = regressor.predict(df[['x']])

plt.figure(figsize=(10,10))  
plt.scatter(
    x=df[['x']],  
    y=df[['y']],
    label="actual"
)
plt.plot(  
    df[['x']],
    prediction,  
    label="predicted",  
    color="orange"
)

# %%
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=100)  
regressor.fit(df[['x']], df[['y']].to_numpy().ravel())
prediction = regressor.predict(df[['x']])

plt.figure(figsize=(10,10))  
plt.scatter(
    x=df[['x']],  
    y=df[['y']],
    label="actual"
)
plt.plot(  
    df[['x']],
    prediction,  
    label="predicted",  
    color="orange"
)

# %%
