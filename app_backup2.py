from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import logging
import os
from dotenv import load_dotenv
import openai
from prophet import Prophet

load_dotenv()  # Load environment variables from .env

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

# Configure OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize a global variable to store the latest forecast details
latest_forecast = {}

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/download-template')
def download_template():
    try:
        templates_dir = os.path.join(app.root_path, 'static', 'templates')
        filename = 'sample_inventory_template.csv'

        file_path = os.path.join(templates_dir, filename)
        if not os.path.exists(file_path):
            app.logger.error(f"File {filename} not found in {templates_dir}")
            return jsonify({'error': 'Template file not found.'}), 404

        return send_from_directory(templates_dir, filename, as_attachment=True)
    except Exception as e:
        app.logger.error(f"Error in download_template: {e}")
        return jsonify({'error': 'Internal Server Error.'}), 500

@app.route('/upload', methods=['POST'])
def upload():
    logging.info("Received upload request")
    file = request.files.get('file')
    if not file:
        logging.error("No file uploaded")
        return jsonify({'error': 'No file uploaded'}), 400

    try:
        df = pd.read_csv(file)
        logging.info("CSV file loaded into DataFrame")

        required_columns = {'Date', 'Vendor', 'Category', 'Product', 'SalesRate',
                            'Sales Revenue', 'Profit', 'Stock Level', 'Reorder Point'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            logging.error(f"Missing required columns: {', '.join(missing_columns)}")
            return jsonify({'error': f"Missing required columns: {', '.join(missing_columns)}"}), 400

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

        total_vendors = int(df['Vendor'].nunique())
        total_categories = int(df['Category'].nunique())
        total_products = int(df['Product'].nunique())
        total_sales_rate = float(df['SalesRate'].sum())
        total_sales_revenue = float(df['Sales Revenue'].sum())
        total_profit = float(df['Profit'].sum())
        low_stock_products = df[df['Stock Level'] <= df['Reorder Point']].shape[0]
        out_of_stock_products = df[df['Stock Level'] == 0].shape[0]

        sales_data = df.groupby('Date')['SalesRate'].sum().reset_index()
        sales_data.rename(columns={'Date': 'ds', 'SalesRate': 'y'}, inplace=True)

        # Use Prophet for forecasting
        model = Prophet()
        model.fit(sales_data)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30)

        forecast_dates_str = forecast['ds'].dt.strftime('%Y-%m-%d').tolist()
        forecast_sales = forecast['yhat'].round(2).tolist()

        historical_dates = sales_data['ds'].dt.strftime('%Y-%m-%d').tolist()
        historical_sales = sales_data['y'].round(2).tolist()

        salesReport = [
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
        ]

        sales_by_category = df.groupby('Category')['Sales Revenue'].sum().reset_index()
        stock_levels = df.groupby('Product').agg({
            'Stock Level': 'sum',
            'Reorder Point': 'mean'
        }).reset_index()
        profit_by_category = df.groupby('Category')['Profit'].sum().reset_index()

        forecast_details = {
            "meanForecast": forecast['yhat'].mean(),
            "medianForecast": forecast['yhat'].median(),
            "maxForecast": forecast['yhat'].max(),
            "minForecast": forecast['yhat'].min()
        }

        global latest_forecast
        latest_forecast = forecast_details

        data = {
            "totalVendors": total_vendors,
            "totalCategories": total_categories,
            "totalProducts": total_products,
            "totalSalesRate": total_sales_rate,
            "totalSalesRevenue": total_sales_revenue,
            "totalProfit": total_profit,
            "lowStockProducts": low_stock_products,
            "outOfStockProducts": out_of_stock_products,
            "salesReport": salesReport,
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

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        if not user_message:
            return jsonify({'error': 'No message provided.'}), 400

        if "predict" in user_message.lower() or "forecast" in user_message.lower():
            prediction = generate_inventory_prediction()
            ai_reply = f"Here's the inventory forecast:\n"
            ai_reply += f"Mean: {prediction['meanForecast']:.2f}\n"
            ai_reply += f"Median: {prediction['medianForecast']:.2f}\n"
            ai_reply += f"Max: {prediction['maxForecast']:.2f}\n"
            ai_reply += f"Min: {prediction['minForecast']:.2f}"
        else:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an AI assistant helping with inventory management."},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=150,
                temperature=0.7,
            )
            ai_reply = response['choices'][0]['message']['content'].strip()

        return jsonify({'reply': ai_reply})
    except Exception as e:
        logging.error(f"Error in /chat route: {e}")
        return jsonify({'error': 'Internal Server Error.'}), 500

def generate_inventory_prediction():
    global latest_forecast
    return latest_forecast if latest_forecast else {
        "meanForecast": 0,
        "medianForecast": 0,
        "maxForecast": 0,
        "minForecast": 0
    }

if __name__ == '__main__':
    app.run(debug=True)
