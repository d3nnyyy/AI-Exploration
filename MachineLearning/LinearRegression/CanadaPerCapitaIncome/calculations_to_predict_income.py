import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

# Load the original data from the CSV file
dataframe = pd.read_csv("canada_per_capita_income.csv")

# Fit the linear regression model as before
rg = linear_model.LinearRegression()
rg.fit(dataframe[["year"]], dataframe[["per capita income (US$)"]])

# Generate a list of years from 2017 to 2027
years = [x for x in range(2017, 2028)]

# Predict per capita income for these years
predicted_income = rg.predict([[x] for x in years]).flatten()  # Flatten the 1D array

# Create a new DataFrame for the predicted values
predicted_df = pd.DataFrame({'year': years, 'per capita income (US$)': predicted_income})

# Concatenate the original DataFrame and the predicted DataFrame
updated_dataframe = pd.concat([dataframe, predicted_df], ignore_index=True)

# Save the updated DataFrame to a new CSV file
updated_dataframe.to_csv("updated_canada_per_capita_income.csv", index=False)

# Display the updated DataFrame
print(updated_dataframe)

# Create and display the scatter plot with the updated data
plt.xlabel("year")
plt.ylabel("per capita income (US$)")
plt.scatter(updated_dataframe["year"], updated_dataframe["per capita income (US$)"], color='red', marker='+')
plt.plot(updated_dataframe["year"], rg.predict(updated_dataframe[["year"]]), color='green')
plt.show()
