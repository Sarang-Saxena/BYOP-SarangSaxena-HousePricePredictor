import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


house_data = pd.read_csv('data.csv')     #With this we can read the housedata from our csv file which i got from Kaggle


house_data = house_data.dropna()
print(f" Loaded {len(house_data)} clean rows from data.csv\n")     #Removing the last row which has incomplete entries 

X = house_data[['area', 'bedrooms', 'bathrooms', 'stories', 'parking']]    #These are 5 main features we will use to compare the houses on 

y_scaled = house_data['price'] / 100000     #using prices in lakhs to make it easy for training model

y_full = house_data['price']     #using original prices for table comparison


X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, train_size=0.7, random_state=42)     #70% data to train and 30% to test

model = LinearRegression()     #Creating a linear regression model

model.fit(X_train, y_train)     #teaching the model using training data   


y_pred_scaled = model.predict(X_test)     #Asking the model to prices.The data is still in lakhs

y_pred_full = y_pred_scaled * 100000     #Converting them to original prices for our comparison table


table = pd.DataFrame({
    'Price': y_test[:100].values * 100000,          
    'Predicted Price': y_pred_full[:100]
})                                       #making a table based on prices(from our dataset csv file and price predicted by the model)

print("\n" + "="*80)
print("PRICE vs PREDICTED PRICE (FIRST 100 ROWS - FULL ORIGINAL AMOUNTS)")
print("="*80)
print(table.round(0).astype(int))   #making the numbers clean and whole

#MAKING A GRAPH TO COMPARE PRICES
plt.figure(figsize=(16, 9))

# Black dots = actual test prices
plt.scatter(range(len(y_test[:150])), y_test[:150], color='black', label='Test data')   

# Orange stars = what the model predicted
plt.scatter(range(len(y_test[:150])), y_pred_scaled[:150], color='orange', marker='*', label='Predicted Data')

plt.xlabel('Houses')
plt.ylabel('House Prices (in Lakhs)')
plt.title('Test data vs Predicted data')
plt.legend()
plt.show()