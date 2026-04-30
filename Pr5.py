import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

def forecast_demand_analysis():
    # 1. Генерація/Завантаження даних про попит
    np.random.seed(42)
    days = 100
    time = np.arange(days)
    demand = 20 + 0.5 * time + 10 * np.sin(2 * np.pi * time / 7) + np.random.normal(0, 3, days)
    
    series = pd.Series(demand)
    
    train_size = int(len(series) * 0.8)
    train, test = series[0:train_size], series[train_size:len(series)]
    
    # 2. Побудова моделі ARIMA
    model = ARIMA(train, order=(5, 1, 0))
    model_fit = model.fit()
    
    # 3. Прогнозування
    history = [x for x in train]
    predictions = list()
    
    for t in range(len(test)):
        model = ARIMA(history, order=(5, 1, 0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        history.append(test.iloc[t])
    
    # 4. Оцінка точності
    error = np.sqrt(mean_squared_error(test, predictions))
    
    # 5. Візуалізація
    plt.figure(figsize=(12, 6))
    plt.plot(series.index, series, label='Історичний попит', color='blue', alpha=0.5)
    plt.plot(test.index, predictions, label='Прогноз (ARIMA)', color='red', linewidth=2)
    plt.fill_between(test.index, predictions, test, color='pink', alpha=0.3, label='Похибка')
    
    plt.title(f'Прогнозування попиту на товар (RMSE: {error:.2f})')
    plt.xlabel('Час (дні)')
    plt.ylabel('Кількість замовлень')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    print(f"Середньоквадратична помилка прогнозу (RMSE): {error:.4f}")

if __name__ == "__main__":
    forecast_demand_analysis()