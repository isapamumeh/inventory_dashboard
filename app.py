from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import logging
import os

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

# Route to serve the CSV template
@app.route('/download-template')
def download_template():
    try:
        # Construct the absolute path to the 'static/templates' directory
        templates_dir = os.path.join(app.root_path, 'static', 'templates')
        filename = 'sample_inventory_template.csv'

        # Check if the file exists
        file_path = os.path.join(templates_dir, filename)
        if not os.path.exists(file_path):
            app.logger.error(f"File {filename} not found in {templates_dir}")
            return jsonify({'error': 'Template file not found.'}), 404

        # Send the file as an attachment
        return send_from_directory(templates_dir, filename, as_attachment=True)
    except Exception as e:
        app.logger.error(f"Error in download_template: {e}")
        return jsonify({'error': 'Internal Server Error.'}), 500
@app.route('/')
def index():
    # Render the upload.html template when accessing the root URL
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    logging.info("Received upload request")
    file = request.files.get('file')
    if not file:
        logging.error("No file uploaded")
        return jsonify({'error': 'No file uploaded'}), 400

    try:
        # Load CSV file into a DataFrame
        df = pd.read_csv(file)
        logging.info("CSV file loaded into DataFrame")

        # Define required columns and check if they exist in the dataset
        required_columns = {'Date', 'Vendor', 'Category', 'Product', 'SalesRate',
                            'Sales Revenue', 'Profit', 'Stock Level', 'Reorder Point'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            logging.error(f"Missing required columns: {', '.join(missing_columns)}")
            return jsonify({'error': f'Missing required columns: {", ".join(missing_columns)}'}), 400

        # Validate data types
        try:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
            df['SalesRate'] = pd.to_numeric(df['SalesRate'])
            df['Sales Revenue'] = pd.to_numeric(df['Sales Revenue'])
            df['Profit'] = pd.to_numeric(df['Profit'])
            df['Stock Level'] = pd.to_numeric(df['Stock Level'])
            df['Reorder Point'] = pd.to_numeric(df['Reorder Point'])
        except Exception as e:
            logging.error(f"Data type conversion error: {str(e)}")
            return jsonify({'error': f'Data type conversion error: {str(e)}'}), 400

        # Perform calculations
        total_vendors = int(df['Vendor'].nunique())
        total_categories = int(df['Category'].nunique())
        total_products = int(df['Product'].nunique())
        total_sales_rate = float(df['SalesRate'].sum())
        total_sales_revenue = float(df['Sales Revenue'].sum())
        total_profit = float(df['Profit'].sum())

        # Low stock products (Stock Level <= Reorder Point)
        low_stock_products = df[df['Stock Level'] <= df['Reorder Point']].shape[0]

        # Out of stock products (Stock Level == 0)
        out_of_stock_products = df[df['Stock Level'] == 0].shape[0]

        # Prepare data for time series forecasting (using Sales Rate)
        sales_data = df.groupby('Date')['SalesRate'].sum().resample('D').sum().fillna(0)
        logging.info("Sales data prepared for ARIMA")

        # Build and fit the ARIMA model
        model = ARIMA(sales_data, order=(1, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=30)
        forecast_dates = pd.date_range(sales_data.index[-1] + pd.Timedelta(days=1), periods=30)
        forecast_series = pd.Series(forecast, index=forecast_dates)
        logging.info("ARIMA model fitted and forecasted")

        # Convert index and values to lists of native Python types
        historical_dates = sales_data.index.strftime('%Y-%m-%d').tolist()
        historical_sales = sales_data.values.astype(float).tolist()

        forecast_dates_str = forecast_series.index.strftime('%Y-%m-%d').tolist()
        forecast_sales = forecast_series.values.astype(float).tolist()

        # Prepare sales breakdown by category
        sales_by_category = df.groupby('Category')['Sales Revenue'].sum().reset_index()

        # Prepare stock levels vs reorder points
        stock_levels = df.groupby('Product').agg({
            'Stock Level': 'sum',
            'Reorder Point': 'mean'  # Assuming average reorder point per product
        }).reset_index()

        # Prepare profit analysis by category
        profit_by_category = df.groupby('Category')['Profit'].sum().reset_index()

        # Prepare forecast statistics
        forecast_details = {
            "meanForecast": forecast_series.mean(),
            "medianForecast": forecast_series.median(),
            "maxForecast": forecast_series.max(),
            "minForecast": forecast_series.min()
        }

        # Prepare the data to send back to the frontend
        data = {
            "totalVendors": total_vendors,
            "totalCategories": total_categories,
            "totalProducts": total_products,
            "totalSalesRate": total_sales_rate,
            "totalSalesRevenue": total_sales_revenue,
            "totalProfit": total_profit,
            "lowStockProducts": low_stock_products,
            "outOfStockProducts": out_of_stock_products,
            "salesReport": [
                {
                    "x": historical_dates,
                    "y": historical_sales,
                    "name": "Historical Sales",
                    "type": "scatter",
                    "mode": "lines",
                    "line": {"color": "#17BECF"}
                },
                {
                    "x": forecast_dates_str,
                    "y": forecast_sales,
                    "name": "Forecasted Sales",
                    "type": "scatter",
                    "mode": "lines",
                    "line": {"dash": "dash", "color": "#7F7F7F"}
                }
            ],
            "salesByCategory": {
                "labels": sales_by_category['Category'].tolist(),
                "values": sales_by_category['Sales Revenue'].tolist()
            },
            "stockLevels": {
                "products": stock_levels['Product'].tolist(),
                "stockLevels": stock_levels['Stock Level'].tolist(),
                "reorderPoints": stock_levels['Reorder Point'].tolist()
            },
            "profitByCategory": {
                "labels": profit_by_category['Category'].tolist(),
                "values": profit_by_category['Profit'].tolist()
            },
            "forecastDetails": forecast_details
        }
        logging.info("Data prepared for response")
        return jsonify(data)

    except Exception as e:
        logging.exception("Error processing the uploaded file")
        return jsonify({'error': f'An error occurred during processing: {str(e)}'}), 500

# Allow embedding in an iFrame
@app.after_request
def add_header(response):
    response.headers['X-Frame-Options'] = 'ALLOWALL'
    return response

if __name__ == '__main__':
    # Run the Flask app in debug mode
    app.run(debug=True)
