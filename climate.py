#importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import statsmodels.api as sm
#Reading dataset
data = pd.read_csv('daily_data.csv', parse_dates=['DATE'], index_col='DATE')
print(data.head())
#Plotting
data["DailyAverageDewPointTemperature"].plot(figsize=(16, 6), fontsize=15)
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.show()
data["DailyAverageDewPointTemperature"].hist()
plt.show()
data["DailyAverageDryBulbTemperature"].plot(figsize=(16, 6), fontsize=15,color='red')
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.show()
data["DailyAverageDryBulbTemperature"].hist(color='red')
plt.show()
data["DailyAverageRelativeHumidity"].plot(figsize=(16, 6), fontsize=15,color='gray')
plt.xlabel("Date")
plt.ylabel("Humidity")
plt.show()
data["DailyAverageRelativeHumidity"].hist(color='gray')
plt.show()
data["DailyAverageSeaLevelPressure"].plot(figsize=(16, 6), fontsize=15,color='green')
plt.xlabel("Date")
plt.ylabel("Pressure")
plt.show()
data["DailyAverageSeaLevelPressure"].hist(color='green')
plt.show()
data = pd.read_csv('daily_data.csv', parse_dates=['DATE'])
plt.figure(figsize=(10, 6))
plt.scatter(data['DailyAverageRelativeHumidity'], data['DailyAverageDryBulbTemperature'], alpha=0.5)
plt.title('Humidity vs Temperature')
plt.xlabel('Humidity (%)')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.show()

data = pd.read_csv('daily_data.csv')

# Calculate the correlation matrix
correlation_matrix = data.corr()
print("Correlation Matrix:")
print(correlation_matrix)
daily_data = pd.read_csv('daily_data.csv', parse_dates=['DATE'])
monthly_data = pd.read_csv('monthly_data.csv', parse_dates=['DATE'])

daily_avg_temp = daily_data.groupby(pd.Grouper(key='DATE', freq='D'))['DailyAverageDryBulbTemperature'].mean()

monthly_avg_temp = monthly_data.groupby(pd.Grouper(key='DATE', freq='M'))['MonthlyMeanTemperature'].mean()

# Plotting daily and monthly average temperatures
plt.figure(figsize=(12, 6))
plt.plot(daily_avg_temp.index, daily_avg_temp.values, label='Daily Average Temperature', color='yellow')
plt.plot(monthly_avg_temp.index, monthly_avg_temp.values, label='Monthly Average Temperature', color='blue')

plt.title('Daily vs Monthly Average Temperature')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()
daily_data = pd.read_csv('daily_data.csv', parse_dates=['DATE'])
monthly_data = pd.read_csv('monthly_data.csv', parse_dates=['DATE'])

# Plotting histograms for daily and monthly average temperatures
plt.figure(figsize=(12, 6))
plt.hist(daily_data['DailyAverageDryBulbTemperature'], bins=30, alpha=0.5, label='Daily Average Temperature', color='blue')
plt.hist(monthly_data['MonthlyMeanTemperature'], bins=30, alpha=0.5, label='Monthly Average Temperature', color='green')

plt.title('Distribution of Daily and Monthly Average Temperature')
plt.xlabel('Temperature (°C)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()
daily_data = pd.read_csv('daily_data.csv', parse_dates=['DATE'])
monthly_data = pd.read_csv('monthly_data.csv', parse_dates=['DATE'])
three_hour_data = pd.read_csv('three_hour_data.csv', parse_dates=['DATE'])
three_hour_data['HourlyDryBulbTemperature'] = pd.to_numeric(three_hour_data['HourlyDryBulbTemperature'], errors='coerce')
daily_avg_temp = daily_data.groupby(pd.Grouper(key='DATE', freq='D'))['DailyAverageDryBulbTemperature'].mean()
monthly_avg_temp = monthly_data.groupby(pd.Grouper(key='DATE', freq='M'))['MonthlyMeanTemperature'].mean()
three_hour_avg_temp = three_hour_data.groupby(pd.Grouper(key='DATE', freq='3H'))['HourlyDryBulbTemperature'].mean()

# Plotting average temperatures
plt.figure(figsize=(12, 6))

# Plotting three-hour average temperature first with lower transparency
plt.plot(three_hour_avg_temp.index, three_hour_avg_temp.values, label='Three-Hour Average Temperature', color='red', alpha=0.5)

# Plotting daily average temperature and monthly average temperature on top
plt.plot(daily_avg_temp.index, daily_avg_temp.values, label='Daily Average Temperature', color='blue')
plt.plot(monthly_avg_temp.index, monthly_avg_temp.values, label='Monthly Average Temperature', color='black')

plt.title('Comparison of Temperature Averages')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()

import statistics

meanVal, varianceVal = [], []
for i in range(10):
    sample = random.sample(list(data.DailyAverageDewPointTemperature), 100)
    meanVal.append(np.mean(sample))
    varianceVal.append(statistics.variance(sample))

plt.plot(np.mean(data.DailyAverageDewPointTemperature), marker="o", markersize=10, label='Global Mean')
plt.plot(meanVal, label='Mean')
plt.plot(statistics.variance(data.DailyAverageDewPointTemperature), marker="+", markersize=10, label='Global Variance')
plt.plot(varianceVal, label='Variance')
plt.legend()
plt.show()
from statsmodels.tsa.stattools import adfuller

dfTest = adfuller(data.DailyAverageDewPointTemperature, autolag='AIC')

print('l. ADF: ' ,dfTest[0])
print('2. P-Value: ', dfTest[1])
print('3. Num Of Lags: ', dfTest[2])
print('4. Num Of Observations used For ADF Regression and Critical values Calculation: ', dfTest[3])
print('5. Critical Values:')
for key, val in dfTest[4].items():
    print('\t', key, ': ', val)

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
pacf = plot_pacf(data.DailyAverageDewPointTemperature, lags=10)
acf = plot_acf(data.DailyAverageDewPointTemperature, lags=50)
