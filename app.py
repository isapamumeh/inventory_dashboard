from flask import Flask, request, jsonify, render_template, make_response
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)

@app.route('/')
def index():
    # Render the upload.html template when accessing the root URL
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    df = pd.read_csv(file)

    # Perform data analysis and calculations
    total_vendors = int(df['Vendor'].nunique())
    total_categories = int(df['Category'].nunique())
    total_products = int(df['Product'].nunique())
    total_sales = float(df['SalesRate'].sum())

    # Prepare data for time series forecasting
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    sales_data = df.groupby('Date')['SalesRate'].sum().resample('D').sum().fillna(0)

    # Build and fit the ARIMA model
    model = ARIMA(sales_data, order=(1, 1, 1))
    model_fit = model.fit()

    # Forecast the next 30 days
    forecast = model_fit.forecast(steps=30)
    forecast_dates = pd.date_range(sales_data.index[-1] + pd.Timedelta(days=1), periods=30)
    forecast_series = pd.Series(forecast, index=forecast_dates)

    # Convert index and values to lists of native Python types
    historical_dates = sales_data.index.strftime('%Y-%m-%d').tolist()
    historical_sales = sales_data.values.astype(float).tolist()

    forecast_dates_str = forecast_series.index.strftime('%Y-%m-%d').tolist()
    forecast_sales = forecast_series.values.astype(float).tolist()

    # Prepare the data to send back to the frontend
    data = {
        "totalVendors": total_vendors,
        "totalCategories": total_categories,
        "totalProducts": total_products,
        "totalSales": total_sales,
        "salesReport": [
            {
                "x": historical_dates,
                "y": historical_sales,
                "name": "Historical Sales"
            },
            {
                "x": forecast_dates_str,
                "y": forecast_sales,
                "name": "Forecasted Sales"
            }
        ]
    }
    return jsonify(data)

# Add this function to allow embedding in an iFrame
@app.after_request
def add_header(response):
    response.headers['X-Frame-Options'] = 'ALLOWALL'
    return response

if __name__ == '__main__':
    # Run the Flask app in debug mode
    app.run()
