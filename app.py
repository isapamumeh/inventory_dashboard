from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import logging
import os
from dotenv import load_dotenv
import openai
from prophet import Prophet

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError("OpenAI API key is missing. Please check your .env file.")

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

# Global storage for uploaded data and forecast details
uploaded_data = None
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
            return jsonify({'error': 'Template file not found.'}), 404
        return send_from_directory(templates_dir, filename, as_attachment=True)
    except Exception as e:
        logging.error(f"Error in download_template: {e}")
        return jsonify({'error': 'Internal Server Error.'}), 500


@app.route('/upload', methods=['POST'])
def upload():
    global uploaded_data

    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    try:
        df = pd.read_csv(file)

        required_columns = {'Date', 'Vendor', 'Category', 'Product', 'SalesRate', 
                            'Sales Revenue', 'Profit', 'Stock Level', 'Reorder Point'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            return jsonify({'error': f"Missing required columns: {', '.join(missing_columns)}"}), 400

        # Convert columns to appropriate data types
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df['SalesRate'] = pd.to_numeric(df['SalesRate'])
        df['Sales Revenue'] = pd.to_numeric(df['Sales Revenue'])
        df['Profit'] = pd.to_numeric(df['Profit'])
        df['Stock Level'] = pd.to_numeric(df['Stock Level'])
        df['Reorder Point'] = pd.to_numeric(df['Reorder Point'])

        # Perform calculations and convert to native Python types
        total_vendors = int(df['Vendor'].nunique())
        total_categories = int(df['Category'].nunique())
        total_products = int(df['Product'].nunique())
        total_sales_rate = float(df['SalesRate'].sum())
        total_sales_revenue = float(df['Sales Revenue'].sum())
        total_profit = float(df['Profit'].sum())
        low_stock_products = int(df[df['Stock Level'] <= df['Reorder Point']].shape[0])
        out_of_stock_products = int(df[df['Stock Level'] == 0].shape[0])

        # Prepare time series data for Prophet
        sales_data = df.groupby('Date')['SalesRate'].sum().reset_index()
        sales_data.rename(columns={'Date': 'ds', 'SalesRate': 'y'}, inplace=True)
        model = Prophet()
        model.fit(sales_data)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        forecast_dates = forecast['ds'].dt.strftime('%Y-%m-%d').tolist()
        forecast_sales = forecast['yhat'].astype(float).tolist()

        # Prepare other analysis data
        sales_by_category = df.groupby('Category')['Sales Revenue'].sum().reset_index()
        stock_levels = df.groupby('Product').agg({'Stock Level': 'sum', 'Reorder Point': 'mean'}).reset_index()
        profit_by_category = df.groupby('Category')['Profit'].sum().reset_index()

        global latest_forecast, uploaded_data
        uploaded_data = df.to_dict(orient='records')
        latest_forecast = {
            "meanForecast": float(forecast['yhat'].mean()),
            "medianForecast": float(forecast['yhat'].median()),
            "maxForecast": float(forecast['yhat'].max()),
            "minForecast": float(forecast['yhat'].min())
        }

        # Prepare response data, ensuring all values are native Python types
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
                {"x": forecast_dates, "y": forecast_sales, "name": "Forecasted Sales"}
            ],
            "salesByCategory": {
                "labels": sales_by_category['Category'].tolist(),
                "values": sales_by_category['Sales Revenue'].astype(float).tolist()
            },
            "stockLevels": {
                "products": stock_levels['Product'].tolist(),
                "stockLevels": stock_levels['Stock Level'].astype(float).tolist(),
                "reorderPoints": stock_levels['Reorder Point'].astype(float).tolist()
            },
            "profitByCategory": {
                "labels": profit_by_category['Category'].tolist(),
                "values": profit_by_category['Profit'].astype(float).tolist()
            },
            "forecastDetails": latest_forecast
        }

        return jsonify(data)

    except Exception as e:
        logging.exception("Error processing upload")
        return jsonify({'error': str(e)}), 500



@app.route('/chat', methods=['POST'])
def chat():
    global uploaded_data

    try:
        data = request.get_json()
        user_message = data.get('message', '')
        if not user_message:
            return jsonify({'error': 'No message provided.'}), 400

        # Provide uploaded data context to the AI
        context = (
            "You are an AI assistant helping with inventory management.\n"
            "Here is the uploaded inventory data in JSON format:\n"
            f"{uploaded_data}\n\n"
            "Respond to the user's query based on this data."
        )

        # Use the OpenAI API with gpt-4o
        response = openai.ChatCompletion.create(
            model="gpt-4o",  # Use the optimized GPT-4 model
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": user_message}
            ],
            max_tokens=150,
            temperature=0.7
        )

        reply = response['choices'][0]['message']['content']
        return jsonify({'reply': reply})

    except Exception as e:
        logging.error(f"Error in chat route: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
