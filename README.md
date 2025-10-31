# Weather Time Series Forecasting

A comprehensive analysis and comparison of multiple time series forecasting models applied to weather data, including temperature, humidity, and visibility predictions.

## Overview

This project implements and compares various time series forecasting models to predict weather parameters. The analysis includes both traditional statistical methods and modern deep learning approaches, providing insights into their effectiveness for weather prediction tasks.

## Dataset

The project uses the `weatherHistory1.csv` dataset containing historical weather data with the following features:

- **Temperature (C)**: Temperature in Celsius
- **Humidity**: Relative humidity levels
- **Visibility (km)**: Visibility distance in kilometers
- **Formatted Date**: Timestamp for each observation
- **Summary**: Weather condition summary
- **Daily Summary**: Daily weather overview
- **Precip Type**: Type of precipitation

## Models Implemented

### Traditional Statistical Models
1. **ARIMA** (AutoRegressive Integrated Moving Average)
2. **Exponential Smoothing**
3. **Holt-Winters** (Triple Exponential Smoothing)
4. **SARIMA** (Seasonal ARIMA)
5. **Naive Forecasting** (Baseline)

### Deep Learning Models
1. **LSTM** (Long Short-Term Memory)
2. **GRU** (Gated Recurrent Unit)

## Project Structure

```
├── ime_Series_Forecasting_for_weather.ipynb    # Main notebook
├── weatherHistory1.csv                          # Dataset
└── README.md                                    # Project documentation
```

## Methodology

### 1. Data Preprocessing
- Loading and exploring the dataset
- Handling missing values using forward fill
- Converting date columns to datetime format
- Encoding categorical variables (summary, daily summary, precip type)
- Data normalization using MinMaxScaler for deep learning models

### 2. Exploratory Data Analysis
- Time series visualization of weather parameters
- Statistical analysis of data distributions
- Identifying trends and patterns

### 3. Model Training & Evaluation

**Training/Test Split**: 80% training, 20% testing

**Evaluation Metrics**:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)

### 4. Model Comparison
Comprehensive comparison across all models for each weather parameter using RMSE as the primary metric.

## Results

### Temperature Forecasting (RMSE)
- **LSTM**: 1.068 ⭐ (Best)
- **GRU**: 1.088
- **Naive**: 1.366
- **ARIMA**: 8.993
- **Holt-Winters**: 35.48
- **Exponential Smoothing**: 7950.93

### Humidity Forecasting (RMSE)
- **LSTM**: 0.051 ⭐ (Best)
- **GRU**: 0.052
- **Naive**: 0.059
- **Holt-Winters**: 0.19
- **ARIMA**: 0.191
- **Exponential Smoothing**: 0.26

### Visibility Forecasting (RMSE)
- **GRU**: 2.146 ⭐ (Best)
- **ARIMA**: 4.884
- **Holt-Winters**: 19.70
- **SARIMA**: Results vary
- **Exponential Smoothing**: 1712.70

## Key Findings

1. **Deep learning models (LSTM and GRU) significantly outperform traditional statistical methods** for all weather parameters
2. **LSTM shows the best overall performance** for temperature and humidity prediction
3. **GRU is most effective** for visibility forecasting
4. Traditional methods struggle with the complexity and non-linearity of weather data
5. The Naive baseline often outperforms more complex traditional models

## Technologies Used

- **Python 3.x**
- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization
- **scikit-learn**: Data preprocessing and evaluation metrics
- **TensorFlow/Keras**: Deep learning model implementation
- **statsmodels**: Statistical time series models (ARIMA, SARIMA, Exponential Smoothing)

## Installation

```bash
# Clone the repository
git clone https://github.com/lojaine001/weather-time-series-forecasting.git
cd weather-time-series-forecasting

# Install required packages
pip install pandas numpy matplotlib scikit-learn tensorflow statsmodels
```

## Usage

1. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook ime_Series_Forecasting_for_weather.ipynb
   ```

2. **Run in Google Colab**: Click the "Open in Colab" badge at the top of this README

3. **Execute cells sequentially** to:
   - Load and preprocess the data
   - Train different models
   - Evaluate and compare results
   - Visualize predictions

## Model Configuration

### LSTM Model
```python
Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(time_steps, 1)),
    LSTM(units=50),
    Dense(units=1)
])
```
- Optimizer: Adam
- Loss: Mean Squared Error
- Epochs: 100
- Batch Size: 32

### GRU Model
```python
Sequential([
    GRU(units=50, return_sequences=True, input_shape=(time_steps, 1)),
    GRU(units=50),
    Dense(units=1)
])
```

## Future Improvements

- [ ] Implement attention mechanisms for improved long-term predictions
- [ ] Add more weather parameters (wind speed, pressure, etc.)
- [ ] Experiment with hybrid models combining statistical and deep learning approaches
- [ ] Implement ensemble methods
- [ ] Add real-time prediction capabilities
- [ ] Optimize hyperparameters using grid search or Bayesian optimization
- [ ] Extend forecast horizon to multi-step predictions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).

## Contact

**Lojaine** - [GitHub Profile](https://github.com/lojaine001)

## Acknowledgments

- Dataset source: Historical weather data
- Inspiration from various time series forecasting research papers
- TensorFlow and statsmodels documentation

---

⭐ If you find this project useful, please consider giving it a star!
