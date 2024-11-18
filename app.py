from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import logging
import os
from dotenv import load_dotenv
from openai import Client, OpenAIError  # Import OpenAI Client and error handling classes
from prophet import Prophet
from markupsafe import escape


# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
# Initialize OpenAI Client
if not openai_api_key:
    logging.error("OpenAI API key is missing. Please check your .env file.")
    raise EnvironmentError("OpenAI API key is missing. Please check your .env file.")

client = Client(api_key=openai_api_key)

# Initialize the Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log"),
    ],
)

@app.route('/')
def index():
    return render_template('upload.html')  # Adjust to your actual template name
    

@app.route('/download-template')
def download_template():
    try:
        templates_dir = os.path.join(app.root_path, 'static', 'templates')
        filename = 'sample_inventory_template.csv'
        file_path = os.path.join(templates_dir, filename)
        if not os.path.exists(file_path):
            logging.warning(f"Template file {filename} not found in {templates_dir}.")
            return jsonify({'error': 'Template file not found.'}), 404
        return send_from_directory(templates_dir, filename, as_attachment=True)
    except Exception as e:
        logging.exception(f"Error in download_template: {e}")
        return jsonify({'error': 'Internal Server Error.'}), 500

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file:
        logging.warning("No file uploaded in /upload route.")
        return jsonify({'error': 'No file uploaded'}), 400

    try:
        df = pd.read_csv(file)

        required_columns = {'Date', 'Vendor', 'Category', 'Product', 'SalesRate', 
                            'Sales Revenue', 'Profit', 'Stock Level', 'Reorder Point'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            logging.warning(f"Missing required columns: {missing_columns}")
            return jsonify({'error': f"Missing required columns: {', '.join(missing_columns)}"}), 400

        # Convert columns to appropriate data types with error handling
        try:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            numeric_columns = ['SalesRate', 'Sales Revenue', 'Profit', 'Stock Level', 'Reorder Point']
            df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
        except Exception as e:
            logging.exception(f"Data type conversion error: {e}")
            return jsonify({'error': 'Data type conversion error. Please check your CSV file.'}), 400

        # Check for NaN values after conversion
        if df[numeric_columns].isnull().any().any():
            logging.warning("CSV file contains NaN values after data type conversion.")
            return jsonify({'error': 'Some numeric fields contain invalid data. Please check your CSV file.'}), 400

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

        # Check if there are enough data points
        if sales_data.shape[0] < 2:
            logging.warning("Not enough data points for forecasting.")
            return jsonify({'error': 'Not enough data points for forecasting.'}), 400

        try:
            model = Prophet()
            model.fit(sales_data)
        except Exception as e:
            logging.exception(f"Prophet model fitting error: {e}")
            return jsonify({'error': 'Error in forecasting model fitting.'}), 500

        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        forecast_dates = forecast['ds'].dt.strftime('%Y-%m-%d').tolist()
        forecast_sales = forecast['yhat'].astype(float).tolist()

        # Prepare other analysis data
        sales_by_category = df.groupby('Category')['Sales Revenue'].sum().reset_index()
        stock_levels = df.groupby('Product').agg({'Stock Level': 'sum', 'Reorder Point': 'mean'}).reset_index()
        profit_by_category = df.groupby('Category')['Profit'].sum().reset_index()

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
            "forecastDetails": {
                "meanForecast": float(forecast['yhat'].mean()),
                "medianForecast": float(forecast['yhat'].median()),
                "maxForecast": float(forecast['yhat'].max()),
                "minForecast": float(forecast['yhat'].min())
            }
        }
 # Convert the DataFrame to CSV string for analysis
        csv_data = df.to_csv(index=False)
        # Call the function to get analysis from ChatGPT
        analysis_result = get_chatgpt_analysis(csv_data)

       # Add the analysis to your data dictionary
        data['chatgpt_analysis'] = analysis_result

        logging.info("Data upload and processing successful, including ChatGPT analysis.")
        return jsonify(data)
    except Exception as e:
        logging.exception(f"Error in /upload route: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route("/chat", methods=["POST"])
def chat():
    """
    Chat endpoint to handle user input and get a response from OpenAI's GPT-4o.
    """
    try:
        user_message = request.json.get("message", "")
        if not user_message:
            logging.warning("No message provided in /chat route.")
            return jsonify({"error": "No message provided"}), 400

        # Use OpenAI Client to generate a response with GPT-4o
        response = client.chat.completions.create(
            model="gpt-4o",  # Specify the GPT-4o model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_message},
            ],
            max_tokens=150,
            temperature=0.7,
        )

        # Extract and return the response
        reply = response.choices[0].message.content
        logging.info(f"Chat response: {reply}")
        return jsonify({"reply": reply})
    except OpenAIError as e:
        logging.error(f"OpenAI API error: {e}")
        return jsonify({"error": f"OpenAI error: {str(e)}"}), 500
    except Exception as e:
        logging.error(f"Unexpected error in /chat route: {e}")
        return jsonify({"error": "An unexpected error occurred."}), 500



@app.route('/analyze', methods=['POST'])
def analyze_data():
    try:
        user_data = request.json.get('data', '')
        if not user_data:
            logging.warning("No data provided in /analyze route.")
            return jsonify({"error": "No data provided"}), 400

        # Prepare the system prompt
        system_prompt = (
            "Analyze the uploaded sales data to provide both inventory predictions and alert on overstock or low stock situations.\n\n"
            "Consider trends, seasonality, outliers, or any identifiable patterns in the uploaded sales data to forecast future inventory requirements. Use the information to estimate not only the overall future demand but also identify which items may have higher or lower sales in upcoming periods. Additionally, provide alerts for overstock and low stock situations.\n\n"
            "# Steps\n\n"
            "1. **Data Analysis**: Begin by analyzing past sales data for trends over time. Identify important patterns, such as peaks in sales or periods of sustained high/low sales.\n"
            "   - Look for notable fluctuations, local outliers, or consistent seasonality (days, weeks, months, specific times of year).\n"
            "2. **Pattern Recognition**: Recognize historical trends that might affect future sales, e.g., weekly increases or holiday peaks. Document how these changes occurred and identify any causative factors.\n"
            "3. **Forecasting Calculation**: Calculate inventory needs based on identified patterns, either by forecasting future demand directly or by comparing similar periods with past data.\n"
            "4. **Overstock/Low Stock Alerts**: Calculate thresholds for each item to determine whether inventory is sufficient, overstocked, or critically low. Use forecasted demand and buffer requirements to derive these thresholds.\n"
            "5. **Special Considerations**: If relevant, mention factors like promotions, supply chain issues, or other conditions that could affect the forecast.\n\n"
            "Use statistical models only as tools for insight, adjusting based on external variables where appropriate.\n\n"
            "# Output Format\n\n"
            "The response should be formatted in three sections: \"Analysis,\" \"Inventory Prediction,\" and \"Stock Alerts.\" Provide the underlying reasoning and key data insights in the Analysis section, followed by an actionable forecast in the Inventory Prediction section, and conclude with alerts that signal overstock or low stock situations.\n\n"
            "1. **Analysis**: A paragraph summarizing trends, observed seasonality, and any other relevant historical indicators.\n"
            "2. **Inventory Prediction**: A clear list outlining the predicted inventory quantities for product categories / key items. Example format:\n"
            "   ```\n"
            "   Item A: Predicted Quantity 500 units\n"
            "   Item B: Predicted Quantity 350 units\n"
            "   ...\n"
            "   ```\n"
            "3. **Stock Alerts**: Provide a list indicating items that are at risk of overstocking or running a low stock. Example format:\n"
            "   ```\n"
            "   Item A: Low Stock Alert\n"
            "   Item B: Overstock Alert\n"
            "   Item C: Stock Level Normal\n"
            "   ...\n"
            "   ```\n\n"
            "# Notes\n\n"
            "- Inventory may need adjustment based on unforeseen factors such as supplier delays, changes in consumer preference, or economic disruptions.\n"
            "- Predict an extra buffer stock for products with a high likelihood of demand surges (e.g., promotions or holidays).\n"
            "- Alerts should factor in potential lead time and restocking policies for proactive adjustments."
        )

        # Create the conversation
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_data}
        ]

        response = client.chat.completions.create(
    model="gpt-4o",  # Ensure this model exists and is accessible
    messages=messages,
    max_tokens=2048,
    temperature=1,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)


        ai_response = response['choices'][0]['message']['content'].strip()
        logging.info("Data analysis successful.")
        return jsonify({"forecast": ai_response})
    except openai.error.OpenAIError as e:
        logging.exception(f"OpenAI API error in /analyze route: {e}")
        return jsonify({"error": f"OpenAI API error: {str(e)}"}), 500
    except Exception as e:
        logging.exception(f"Error in /analyze route: {e}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
def get_chatgpt_analysis(csv_data):
    try:
        # Prepare the messages for ChatGPT
        messages = [
            {"role": "system", "content": "You are a data analyst specializing in inventory management."},
            {"role": "user", "content": f"""
Analyze the following sales data to provide inventory predictions and alerts on overstock or low stock situations.

Sales Data (CSV):
{csv_data}

Please provide your analysis in the following format:

Analysis:
[Your analysis]

Inventory Prediction:
[Predictions]

Stock Alerts:
[Alerts]
"""}
        ]

        # Call the OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-4o",  # Replace with "gpt-3.5-turbo" if you don't have access to GPT-4
            messages=messages,
            max_tokens=1500,
            temperature=0.7,
        )

        # Get the generated text
        analysis = response['choices'][0]['message']['content'].strip()
        return analysis
    except Exception as e:
        logging.exception(f"Error in get_chatgpt_analysis: {e}")
        return "An error occurred while getting analysis from ChatGPT."

if __name__ == '__main__':
    app.run(debug=True)  # Set to False in production
