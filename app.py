# Golden Goose 1.0 - Fully Updated Code with Improvements, No Omissions, No Abbreviations
# Collaboration step removed, pre-validation sanity check remains, and extensive logging added throughout.

import os
import json
import logging
from logging.handlers import RotatingFileHandler
import traceback
import re
from flask import Flask, request, jsonify, render_template, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import pandas as pd
import openai
from dotenv import load_dotenv
from flask_migrate import Migrate
import yfinance as yf  # For fetching company data
import difflib  # For approximate string matching
from newsapi import NewsApiClient  # For fetching news articles
from textblob import TextBlob  # For sentiment analysis
import numpy as np  # For numerical operations

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configure logging
log_formatter = logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
)

log_file = 'app.log'

file_handler = RotatingFileHandler(log_file, maxBytes=1000000, backupCount=3)
file_handler.setLevel(logging.DEBUG)  # Set to DEBUG to capture detailed logs
file_handler.setFormatter(log_formatter)

app.logger.addHandler(file_handler)
app.logger.setLevel(logging.DEBUG)

for handler in app.logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        app.logger.removeHandler(handler)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx'}

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(BASE_DIR, "feedback.db")
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)

openai.api_key = os.getenv('OPENAI_API_KEY')

if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set it in the environment variables.")

newsapi = NewsApiClient(api_key=os.getenv('NEWSAPI_KEY'))

# Database Models
class AssumptionSet(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sector = db.Column(db.String(50), nullable=False)
    industry = db.Column(db.String(50), nullable=False)
    sub_industry = db.Column(db.String(50), nullable=False)
    scenario = db.Column(db.String(50), nullable=False)
    stock_ticker = db.Column(db.String(20), nullable=True)
    revenue_growth_rate = db.Column(db.Float, nullable=False)
    tax_rate = db.Column(db.Float, nullable=False)
    cogs_pct = db.Column(db.Float, nullable=False)
    wacc = db.Column(db.Float, nullable=False)  # Changed to 'wacc'
    terminal_growth_rate = db.Column(db.Float, nullable=False)
    operating_expenses_pct = db.Column(db.Float, nullable=False)
    feedbacks = db.relationship('Feedback', backref='assumption_set', lazy=True)

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sector = db.Column(db.String(50), nullable=False)
    industry = db.Column(db.String(50), nullable=False)
    sub_industry = db.Column(db.String(50), nullable=False)
    scenario = db.Column(db.String(50), nullable=False)
    score = db.Column(db.Integer, nullable=False)
    comments = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    # Fields for assumption feedback
    revenue_growth_feedback = db.Column(db.String(20), nullable=True)
    tax_rate_feedback = db.Column(db.String(20), nullable=True)
    cogs_pct_feedback = db.Column(db.String(20), nullable=True)
    operating_expenses_feedback = db.Column(db.String(20), nullable=True)
    wacc_feedback = db.Column(db.String(20), nullable=True)  # Changed to 'wacc_feedback'
    # Foreign key to AssumptionSet
    assumption_set_id = db.Column(db.Integer, db.ForeignKey('assumption_set.id'), nullable=False)

try:
    with app.app_context():
        db.create_all()
        app.logger.debug("Database and tables created successfully.")
except Exception as e:
    app.logger.exception("Failed to initialize the database.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_json(json_like_str):
    cleaned = re.sub(r'//.*?\n', '\n', json_like_str)
    cleaned = re.sub(r',\s*}', '}', cleaned)
    cleaned = re.sub(r',\s*\]', ']', cleaned)
    return cleaned

def parse_json_from_reply(reply):
    cleaned_reply = clean_json(reply)
    cleaned_reply = re.sub(r'"WACC"|\'WACC\'', '"wacc"', cleaned_reply, flags=re.IGNORECASE)
    json_match = re.search(r'\{.*\}', cleaned_reply, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            app.logger.exception("JSON parsing failed in parse_json_from_reply")
            return {}
    return {}

def summarize_feedback(sector, industry, sub_industry, scenario):
    app.logger.debug(f"Summarizing feedback for {sector}, {industry}, {sub_industry}, {scenario}")
    try:
        feedback_entries = Feedback.query.join(AssumptionSet).filter(
            AssumptionSet.sector == sector,
            AssumptionSet.industry == industry,
            AssumptionSet.sub_industry == sub_industry,
            AssumptionSet.scenario == scenario
        ).all()

        if not feedback_entries:
            return "No relevant feedback available."

        assumption_feedback_counts = {
            'revenue_growth_rate': {'too_low': 0, 'about_right': 0, 'too_high': 0},
            'tax_rate': {'too_low': 0, 'about_right': 0, 'too_high': 0},
            'cogs_pct': {'too_low': 0, 'about_right': 0, 'too_high': 0},
            'operating_expenses_pct': {'too_low': 0, 'about_right': 0, 'too_high': 0},
            'wacc': {'too_low': 0, 'about_right': 0, 'too_high': 0},
        }

        for entry in feedback_entries:
            for assumption in assumption_feedback_counts.keys():
                feedback_value = getattr(entry, f"{assumption}_feedback")
                if feedback_value in assumption_feedback_counts[assumption]:
                    assumption_feedback_counts[assumption][feedback_value] += 1

        summary_lines = []
        for assumption, counts in assumption_feedback_counts.items():
            total = sum(counts.values())
            if total > 0:
                summary = f"{assumption.replace('_', ' ').title()}:"
                for feedback_value, count in counts.items():
                    percentage = (count / total) * 100
                    summary += f" {percentage:.1f}% {feedback_value.replace('_', ' ')};"
                summary_lines.append(summary)

        comments = [f"- {entry.comments}" for entry in feedback_entries if entry.comments]
        if comments:
            summary_lines.append("\nUser Comments:")
            summary_lines.extend(comments)

        return "\n".join(summary_lines)
    except Exception as e:
        app.logger.exception("Error summarizing feedback")
        return "Error retrieving feedback."

def validate_assumptions(adjusted_assumptions):
    app.logger.debug(f"Validating assumptions: {adjusted_assumptions}")
    ranges = {
        'revenue_growth_rate': (0.0, 1.0, 0.05),     
        'tax_rate': (0.01, 0.5, 0.21),
        'cogs_pct': (0.01, 1.0, 0.6),
        'wacc': (0.01, 0.2, 0.10),
        'terminal_growth_rate': (0.0, 0.05, 0.02),
        'operating_expenses_pct': (0.01, 1.0, 0.2)
    }
    validated_assumptions = {}
    for key, (min_val, max_val, default_val) in ranges.items():
        value = adjusted_assumptions.get(key)
        if value is not None:
            try:
                value = float(value)
                if value > 1.0:
                    value = value / 100.0
                value = max(min_val, min(value, max_val))
            except (ValueError, TypeError):
                value = default_val
        else:
            value = default_val
        validated_assumptions[key] = value
    app.logger.debug(f"Validated assumptions: {validated_assumptions}")
    return validated_assumptions

def call_openai_api(prompt):
    app.logger.debug(f"Calling OpenAI API with prompt: {prompt[:1000]}...")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You are an expert financial analyst."},
                {"role": "user", "content": prompt}
            ]
        )
        assistant_reply = response['choices'][0]['message']['content']
        app.logger.debug(f"Agent output: {assistant_reply[:500]}...")
        return assistant_reply
    except Exception as e:
        app.logger.exception("Error in OpenAI API call")
        return ""

def call_openai_api_with_messages(messages):
    app.logger.debug(f"Calling OpenAI API with messages: {messages}")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.2,
            messages=messages
        )
        assistant_reply = response['choices'][0]['message']['content']
        app.logger.debug(f"Assistant output: {assistant_reply[:500]}...")
        return assistant_reply
    except Exception as e:
        app.logger.exception("Error in OpenAI API call with messages")
        return ""

custom_mappings = {
    'Revenue': ['TotalRevenue', 'OperatingRevenue', 'Net Sales', 'Sales Revenue'],
    'COGS': ['CostOfRevenue', 'CostOfGoodsSold', 'Cost of Revenue', 'Cost of Goods Sold', 'Cost of Goods Manufactured'],
    'Operating Expenses': ['OperatingExpenses', 'SG&A', 'Selling General & Administrative Expenses', 'Selling, General and Administrative Expenses'],
    'Depreciation': ['Depreciation & Amortization', 'Depreciation Expense'],
    'Capital Expenditures': ['CapEx', 'Capital Spending', 'Purchase of Fixed Assets', 'Purchases of property and equipment'],
    'Current Assets': ['Total Current Assets', 'Current Assets'],
    'Current Liabilities': ['Total Current Liabilities', 'Current Liabilities'],
}

def normalize_string(s):
    return re.sub(r'[^a-zA-Z0-9]', '', s).lower()

def get_field_mapping_via_openai(field, existing_labels):
    app.logger.debug(f"Getting field mapping for {field} using OpenAI")
    prompt = f"""
You are a financial data expert. We have a required financial field: '{field}'.
Given the following available data labels from a financial dataset:
{', '.join(existing_labels)}
Please map the required field to the most appropriate label from the available labels.
Provide the mapping in valid JSON format, where the key is the required field and the value is the matching label.
Only use labels exactly as they appear in the available labels. Do not include any comments or extra text.
"""
    response_text = call_openai_api(prompt)
    mapping = parse_json_from_reply(response_text)
    if mapping:
        app.logger.debug(f"Mapping for {field}: {mapping}")
        return mapping.get(field)
    else:
        app.logger.error("Failed to extract JSON from the assistant's reply.")
        return None

def get_field_mappings(required_fields, existing_labels):
    app.logger.debug(f"Getting field mappings for: {required_fields}")
    field_mapping = {}
    existing_labels_normalized = {normalize_string(label): label for label in existing_labels}
    for field in required_fields:
        mapped = False
        if field in custom_mappings:
            for alias in custom_mappings[field]:
                alias_normalized = normalize_string(alias)
                if alias_normalized in existing_labels_normalized:
                    field_mapping[field] = existing_labels_normalized[alias_normalized]
                    mapped = True
                    break
        if not mapped:
            field_normalized = normalize_string(field)
            if field_normalized in existing_labels_normalized:
                field_mapping[field] = existing_labels_normalized[field_normalized]
                mapped = True
            else:
                matches = difflib.get_close_matches(field_normalized, existing_labels_normalized.keys(), n=1, cutoff=0.0)
                if matches:
                    field_mapping[field] = existing_labels_normalized[matches[0]]
                    mapped = True
                else:
                    label = get_field_mapping_via_openai(field, existing_labels)
                    if label:
                        field_mapping[field] = label
                        mapped = True
                    else:
                        field_mapping[field] = None
    app.logger.debug(f"Field mappings result: {field_mapping}")
    return field_mapping

def process_uploaded_file(file_path, file_type):
    app.logger.debug(f"Processing uploaded file: {file_path} of type {file_type}")
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.csv':
            data = pd.read_csv(file_path)
        elif file_extension in ['.xls', '.xlsx']:
            data = pd.read_excel(file_path)
        else:
            return None

        data = data.fillna(0)
        if file_type == 'income_statement':
            return process_income_statement(data)
        elif file_type == 'balance_sheet':
            return process_balance_sheet(data)
        elif file_type == 'cash_flow_statement':
            return process_cash_flow_statement(data)
        else:
            return None
    except Exception as e:
        app.logger.exception(f"Error processing {file_type}")
        return None
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

def process_income_statement(data):
    app.logger.debug("Processing income statement")
    required_fields = ['Revenue', 'COGS', 'Operating Expenses']
    return extract_fields(data, required_fields)

def process_balance_sheet(data):
    app.logger.debug("Processing balance sheet")
    required_fields = ['Current Assets', 'Current Liabilities']
    return extract_fields(data, required_fields)

def process_cash_flow_statement(data):
    app.logger.debug("Processing cash flow statement")
    required_fields = ['Depreciation', 'Capital Expenditures']
    return extract_fields(data, required_fields)

def extract_fields(data, required_fields):
    app.logger.debug(f"Extracting fields: {required_fields}")
    labels = data.iloc[:, 0].astype(str).str.strip().tolist()
    app.logger.debug(f"Existing labels: {labels}")
    data_values = data.iloc[:, 1:].applymap(lambda x: float(str(x).replace(',', '').replace('(', '-').replace(')', '')))

    data_dict = dict(zip(labels, data_values.values.tolist()))
    existing_labels = list(data_dict.keys())

    field_mapping = get_field_mappings(required_fields, existing_labels)
    if not field_mapping:
        app.logger.error("Failed to obtain field mappings.")
        return None

    processed_data = {}
    for field, label in field_mapping.items():
        if label and label in data_dict:
            try:
                values = data_dict[label]
                processed_data[field] = values[0]  # Assuming single-period data
            except KeyError:
                app.logger.error(f"Label '{label}' not found in data dictionary.")
                return None
        else:
            app.logger.error(f"Label for required field '{field}' not found in data dictionary.")
            return None

    app.logger.debug(f"Extracted data: {processed_data}")
    return processed_data

def get_risk_free_rate():
    app.logger.debug("Fetching risk-free rate")
    tnx = yf.Ticker("^TNX")
    data = tnx.history(period="1d")
    if not data.empty:
        current_yield = data['Close'].iloc[-1] / 100.0
        app.logger.debug(f"Risk-free rate: {current_yield}")
        return current_yield
    else:
        return 0.02

def calculate_mrp():
    app.logger.debug("Calculating MRP")
    spy = yf.Ticker("SPY")
    market_data = spy.history(period="10y")
    if market_data.empty:
        return 0.05

    initial_price = market_data['Close'].iloc[0]
    final_price = market_data['Close'].iloc[-1]
    num_days = (market_data.index[-1] - market_data.index[0]).days
    years = num_days / 365.25
    annualized_market_return = (final_price / initial_price) ** (1 / years) - 1
    current_risk_free = get_risk_free_rate()
    mrp = annualized_market_return - current_risk_free
    if mrp < 0:
        mrp = 0.05
    app.logger.debug(f"MRP: {mrp}")
    return mrp

def adjust_for_sector(sector):
    app.logger.debug(f"Adjusting for sector: {sector}")
    prompt = f"""
As a financial analyst specializing in the {sector} sector, analyze current macroeconomic trends, global economic conditions, regulatory changes, and technological advancements affecting this sector. Based on this analysis, provide financial assumptions that reflect these macro-level influences.

Please provide reasonable values for the following financial assumptions, ensuring they are within the specified ranges:

- **revenue_growth_rate (0.0 to 1.0):** Consider how macroeconomic factors might impact revenue growth in the sector.
- **tax_rate (0.01 to 0.5):** Account for any sector-wide tax policies or changes.
- **cogs_pct (0.01 to 1.0):** Reflect on how global supply chain dynamics affect the cost of goods sold.
- **wacc (0.01 to 0.2):** Adjust for macro-level risks affecting the sector's cost of capital.
- **terminal_growth_rate (0.0 to 0.05):** Consider long-term growth prospects influenced by macro trends.
- **operating_expenses_pct (0.01 to 1.0):** Evaluate how sector-wide operational changes impact expenses.

**Important:** Return the results in valid JSON format without any additional text or comments. The JSON should directly contain the key-value pairs for each financial assumption, like this:

{{
  "revenue_growth_rate": ...,
  "tax_rate": ...,
  "cogs_pct": ...,
  "wacc": ...,
  "terminal_growth_rate": ...,
  "operating_expenses_pct": ...
}}

Do not include any additional keys, nesting, or comments in the JSON output.
"""
    assistant_reply = call_openai_api(prompt)
    try:
        cleaned_reply = clean_json(assistant_reply)
        cleaned_reply = re.sub(r'"WACC"|\'WACC\'', '"wacc"', cleaned_reply, flags=re.IGNORECASE)
        json_match = re.search(r'\{.*\}', cleaned_reply, re.DOTALL)
        if json_match:
            adjusted_assumptions = json.loads(json_match.group())
            if isinstance(adjusted_assumptions, dict):
                app.logger.debug(f"Sector adjustments: {adjusted_assumptions}")
                return adjusted_assumptions
            else:
                app.logger.error("Adjusted assumptions is not a dictionary.")
                return {}
        else:
            app.logger.error("Failed to extract JSON from the assistant's reply.")
            return {}
    except json.JSONDecodeError as e:
        app.logger.exception("Error parsing JSON in adjust_for_sector")
        return {}

def adjust_for_industry(industry):
    app.logger.debug(f"Adjusting for industry: {industry}")
    prompt = f"""
As a financial analyst specializing in the {industry} industry, analyze industry-specific dynamics such as competitive landscape, market demand, technological innovations, and regulatory factors. Based on this analysis, provide financial assumptions that reflect these industry-level influences.

Please provide reasonable values for the following financial assumptions, ensuring they are within the specified ranges:

- **revenue_growth_rate (0.0 to 1.0):** Consider industry growth rates and competitive pressures.
- **tax_rate (0.01 to 0.5):** Include any industry-specific tax incentives or burdens.
- **cogs_pct (0.01 to 1.0):** Reflect industry norms for production or service delivery costs.
- **wacc (0.01 to 0.2):** Adjust for industry-specific risk factors.
- **terminal_growth_rate (0.0 to 0.05):** Consider long-term industry sustainability.
- **operating_expenses_pct (0.01 to 1.0):** Evaluate typical operating expenses within the industry.

**Important:** Return the results in valid JSON format without any additional text or comments. The JSON should directly contain the key-value pairs for each financial assumption, like this:

{{
  "revenue_growth_rate": ...,
  "tax_rate": ...,
  "cogs_pct": ...,
  "wacc": ...,
  "terminal_growth_rate": ...,
  "operating_expenses_pct": ...
}}

Do not include any additional keys, nesting, or comments in the JSON output.
"""
    assistant_reply = call_openai_api(prompt)
    try:
        cleaned_reply = clean_json(assistant_reply)
        cleaned_reply = re.sub(r'"WACC"|\'WACC\'', '"wacc"', cleaned_reply, flags=re.IGNORECASE)
        json_match = re.search(r'\{.*\}', cleaned_reply, re.DOTALL)
        if json_match:
            adjusted_assumptions = json.loads(json_match.group())
            if isinstance(adjusted_assumptions, dict):
                app.logger.debug(f"Industry adjustments: {adjusted_assumptions}")
                return adjusted_assumptions
            else:
                app.logger.error("Adjusted assumptions is not a dictionary.")
                return {}
        else:
            app.logger.error("Failed to extract JSON from the assistant's reply.")
            return {}
    except json.JSONDecodeError as e:
        app.logger.exception("Error parsing JSON in adjust_for_industry")
        return {}

def adjust_for_sub_industry(sub_industry):
    app.logger.debug(f"Adjusting for sub_industry: {sub_industry}")
    prompt = f"""
As a financial analyst specializing in the {sub_industry} sub-industry, analyze specific trends such as niche market developments, customer behavior, and sub-industry regulatory issues. Based on this detailed analysis, provide financial assumptions that add granularity to our understanding of the sub-industry's performance.

Please provide reasonable values for the following financial assumptions, ensuring they are within the specified ranges:

- **revenue_growth_rate (0.0 to 1.0):** Reflect niche market growth and customer trends.
- **tax_rate (0.01 to 0.5):** Account for sub-industry-specific tax considerations.
- **cogs_pct (0.01 to 1.0):** Consider unique cost structures in the sub-industry.
- **wacc (0.01 to 0.2):** Adjust for risks unique to the sub-industry.
- **terminal_growth_rate (0.0 to 0.05):** Evaluate long-term prospects specific to the sub-industry.
- **operating_expenses_pct (0.01 to 1.0):** Assess typical operating expenses within the sub-industry.

**Important:** Return the results in valid JSON format without any additional text or comments. The JSON should directly contain the key-value pairs for each financial assumption, like this:

{{
  "revenue_growth_rate": ...,
  "tax_rate": ...,
  "cogs_pct": ...,
  "wacc": ...,
  "terminal_growth_rate": ...,
  "operating_expenses_pct": ...
}}

Do not include any additional keys, nesting, or comments in the JSON output.
"""
    assistant_reply = call_openai_api(prompt)
    try:
        cleaned_reply = clean_json(assistant_reply)
        cleaned_reply = re.sub(r'"WACC"|\'WACC\'', '"wacc"', cleaned_reply, flags=re.IGNORECASE)
        json_match = re.search(r'\{.*\}', cleaned_reply, re.DOTALL)
        if json_match:
            adjusted_assumptions = json.loads(json_match.group())
            if isinstance(adjusted_assumptions, dict):
                app.logger.debug(f"Sub-industry adjustments: {adjusted_assumptions}")
                return adjusted_assumptions
            else:
                app.logger.error("Adjusted assumptions is not a dictionary.")
                return {}
        else:
            app.logger.error("Failed to extract JSON from the assistant's reply.")
            return {}
    except json.JSONDecodeError as e:
        app.logger.exception("Error parsing JSON in adjust_for_sub_industry")
        return {}

def adjust_for_scenario(scenario):
    app.logger.debug(f"Adjusting for scenario: {scenario}")
    prompt = f"""
As a financial analyst, provide financial assumptions appropriate for a '{scenario}' scenario.

- **Base Case:** Assumes moderate growth and typical market conditions.
- **Optimistic Scenario:** Assumes favorable conditions leading to higher growth.
- **Pessimistic Scenario:** Assumes challenging conditions leading to lower growth.

Based on the '{scenario}' scenario, adjust the following financial assumptions accordingly, ensuring they are within the specified ranges:

- **revenue_growth_rate (0.0 to 1.0):** Adjust to reflect the scenario's outlook.
- **tax_rate (0.01 to 0.5):** Consider scenario-specific tax implications.
- **cogs_pct (0.01 to 1.0):** Reflect changes in cost structures due to the scenario.
- **wacc (0.01 to 0.2):** Adjust for changes in perceived risk.
- **terminal_growth_rate (0.0 to 0.05):** Consider long-term growth under the scenario.
- **operating_expenses_pct (0.01 to 1.0):** Evaluate how expenses might change.

**Important:** Return the results in valid JSON format without any additional text or comments. The JSON should directly contain the key-value pairs for each financial assumption, like this:

{{
  "revenue_growth_rate": ...,
  "tax_rate": ...,
  "cogs_pct": ...,
  "wacc": ...,
  "terminal_growth_rate": ...,
  "operating_expenses_pct": ...
}}

Do not include scenario names, any additional keys, nesting, or comments in the JSON output.
"""
    assistant_reply = call_openai_api(prompt)
    try:
        cleaned_reply = clean_json(assistant_reply)
        cleaned_reply = re.sub(r'"WACC"|\'WACC\'', '"wacc"', cleaned_reply, flags=re.IGNORECASE)
        json_match = re.search(r'\{.*\}', cleaned_reply, re.DOTALL)
        if json_match:
            adjusted_assumptions = json.loads(json_match.group())
            if isinstance(adjusted_assumptions, dict):
                app.logger.debug(f"Scenario adjustments: {adjusted_assumptions}")
                return adjusted_assumptions
            else:
                app.logger.error("Adjusted assumptions is not a dictionary.")
                return {}
        else:
            app.logger.error("Failed to extract JSON from the assistant's reply.")
            return {}
    except json.JSONDecodeError as e:
        app.logger.exception("Error parsing JSON in adjust_for_scenario")
        return {}

def adjust_for_company(stock_ticker):
    app.logger.debug(f"Adjusting for company: {stock_ticker}")
    company = yf.Ticker(stock_ticker)
    info = company.info
    company_name = info.get('longName', 'the company')
    industry = info.get('industry', 'the industry')
    business_summary = info.get('longBusinessSummary', '')

    prompt = f"""
As a financial analyst, analyze {company_name} ({stock_ticker}), operating in the {industry}. Consider the company's competitive advantages, financial health, strategic initiatives, and market position.

**Company Business Summary:**
{business_summary}

Based on this analysis, provide adjusted financial assumptions appropriate for {company_name}, ensuring they are within the specified ranges:

- **revenue_growth_rate (0.0 to 1.0):** Reflect expected growth based on company initiatives.
- **tax_rate (0.01 to 0.5):** Include company-specific tax rates or benefits.
- **cogs_pct (0.01 to 1.0):** Consider the company's cost management and efficiencies.
- **wacc (0.01 to 0.2):** Adjust for the company's specific risk profile.
- **terminal_growth_rate (0.0 to 0.05):** Evaluate the company's long-term growth prospects.
- **operating_expenses_pct (0.01 to 1.0):** Assess operating expenses relative to revenue.

**Important:** Return the results in valid JSON format without any additional text or comments. The JSON should directly contain the key-value pairs for each financial assumption, like this:

{{
  "revenue_growth_rate": ...,
  "tax_rate": ...,
  "cogs_pct": ...,
  "wacc": ...,
  "terminal_growth_rate": ...,
  "operating_expenses_pct": ...
}}

Do not include any additional keys, nesting, or comments in the JSON output.
"""
    assistant_reply = call_openai_api(prompt)
    try:
        cleaned_reply = clean_json(assistant_reply)
        cleaned_reply = re.sub(r'"WACC"|\'WACC\'', '"wacc"', cleaned_reply, flags=re.IGNORECASE)
        json_match = re.search(r'\{.*\}', cleaned_reply, re.DOTALL)
        if json_match:
            adjusted_assumptions = json.loads(json_match.group())
            if isinstance(adjusted_assumptions, dict):
                app.logger.debug(f"Company adjustments: {adjusted_assumptions}")
                return adjusted_assumptions
            else:
                app.logger.error("Adjusted assumptions is not a dictionary.")
                return {}
        else:
            app.logger.error("Failed to extract JSON from the assistant's reply.")
            return {}
    except json.JSONDecodeError as e:
        app.logger.exception("Error parsing JSON in adjust_for_company")
        return {}
    except Exception as e:
        app.logger.exception(f"Error adjusting for company {stock_ticker}")
        return {}

def adjust_based_on_feedback(sector, industry, sub_industry, scenario):
    app.logger.debug("Adjusting based on feedback")
    feedback_summary = summarize_feedback(sector, industry, sub_industry, scenario)
    prompt = f"""
You have received user feedback for the {sector} sector, {industry} industry, and {sub_industry} sub-industry under a {scenario} scenario:

{feedback_summary}

Analyze this feedback to identify common themes or consensus on the assumptions being too high or too low. Based on your analysis, adjust the financial assumptions accordingly, ensuring they are within the specified ranges:

- **revenue_growth_rate (0.0 to 1.0)**
- **tax_rate (0.01 to 0.5)**
- **cogs_pct (0.01 to 1.0)**
- **wacc (0.01 to 0.2)**
- **terminal_growth_rate (0.0 to 0.05)**
- **operating_expenses_pct (0.01 to 1.0)**

**Important:** Return the results in valid JSON format without any additional text or comments. The JSON should directly contain the key-value pairs for each financial assumption, like this:

{{
  "revenue_growth_rate": ...,
  "tax_rate": ...,
  "cogs_pct": ...,
  "wacc": ...,
  "terminal_growth_rate": ...,
  "operating_expenses_pct": ...
}}

Do not include any additional keys, nesting, or comments in the JSON output.
"""
    assistant_reply = call_openai_api(prompt)
    try:
        cleaned_reply = clean_json(assistant_reply)
        cleaned_reply = re.sub(r'"WACC"|\'WACC\'', '"wacc"', cleaned_reply, flags=re.IGNORECASE)
        json_match = re.search(r'\{.*\}', cleaned_reply, re.DOTALL)
        if json_match:
            adjusted_assumptions = json.loads(json_match.group())
            if isinstance(adjusted_assumptions, dict):
                app.logger.debug(f"Feedback adjustments: {adjusted_assumptions}")
                return adjusted_assumptions
            else:
                app.logger.error("Adjusted assumptions is not a dictionary.")
                return {}
        else:
            app.logger.error("Failed to extract JSON from the assistant's reply.")
            return {}
    except json.JSONDecodeError as e:
        app.logger.exception("Error parsing JSON in adjust_based_on_feedback")
        return {}

def adjust_based_on_sentiment(stock_ticker):
    app.logger.debug(f"Adjusting based on sentiment for {stock_ticker}")
    try:
        # Initialize NewsAPI client
        company = yf.Ticker(stock_ticker)
        info = company.info
        company_name = info.get('longName', '')

        if not company_name:
            app.logger.warning(f"Company name not found for ticker {stock_ticker}")
            return {}

        articles = newsapi.get_everything(q=company_name, language='en', sort_by='relevancy', page_size=100)

        if 'articles' not in articles or len(articles['articles']) == 0:
            app.logger.warning(f"No news articles found for {company_name}")
            return {}

        sentiment_scores = []
        for article in articles['articles']:
            content = article.get('content', '')
            if content:
                blob = TextBlob(content)
                sentiment_scores.append(blob.sentiment.polarity)

        if not sentiment_scores:
            app.logger.warning(f"No sentiment scores calculated for {company_name}")
            return {}

        aggregate_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        adjusted_assumptions = {}

        if aggregate_sentiment > 0.1:
            adjusted_assumptions['revenue_growth_rate'] = 0.07
            adjusted_assumptions['wacc'] = 0.09
        elif aggregate_sentiment < -0.1:
            adjusted_assumptions['revenue_growth_rate'] = 0.03
            adjusted_assumptions['wacc'] = 0.11
        else:
            adjusted_assumptions['revenue_growth_rate'] = 0.05
            adjusted_assumptions['wacc'] = 0.10

        app.logger.debug(f"Sentiment adjustments: {adjusted_assumptions}")
        return adjusted_assumptions

    except Exception as e:
        app.logger.exception(f"Error in adjust_based_on_sentiment for {stock_ticker}")
        return {}

def adjust_based_on_historical_data(stock_ticker):
    app.logger.debug(f"Adjusting based on historical data for {stock_ticker}")
    try:
        company = yf.Ticker(stock_ticker)
        income_statement = company.financials
        if income_statement.empty:
            app.logger.warning(f"No financial data available for {stock_ticker}.")
            return {}

        required_fields = ['Total Revenue', 'Income Before Tax', 'Income Tax Expense', 'Cost Of Revenue', 'Selling General Administrative']
        for field in required_fields:
            if field not in income_statement.index:
                app.logger.warning(f"Field '{field}' not found in financial data for {stock_ticker}.")
                return {}

        revenue = income_statement.loc['Total Revenue'].dropna()
        if len(revenue) < 2:
            app.logger.warning(f"Not enough revenue data to calculate growth rates for {stock_ticker}.")
            return {}

        revenue_growth_rates = revenue.pct_change().dropna()
        average_revenue_growth = revenue_growth_rates.mean()

        income_before_tax = income_statement.loc['Income Before Tax']
        income_tax_expense = income_statement.loc['Income Tax Expense']
        tax_rates = income_tax_expense / income_before_tax
        tax_rates = tax_rates.replace([np.inf, -np.inf], np.nan).dropna()
        average_tax_rate = tax_rates.mean()

        cogs = income_statement.loc['Cost Of Revenue']
        cogs_pct = (cogs / revenue).mean()
        average_cogs_pct = cogs_pct

        opex = income_statement.loc['Selling General Administrative']
        opex_pct = (opex / revenue).mean()
        average_opex_pct = opex_pct

        beta = company.info.get('beta')
        if beta is None:
            beta = 1.0

        # Risk-free rate (e.g., 2%)
        risk_free_rate = 0.02
        # Market risk premium (e.g., 5%)
        market_risk_premium = 0.05

        cost_of_equity = risk_free_rate + beta * market_risk_premium
        average_wacc = cost_of_equity

        adjusted_assumptions = {
            'revenue_growth_rate': average_revenue_growth,
            'tax_rate': average_tax_rate,
            'cogs_pct': average_cogs_pct,
            'operating_expenses_pct': average_opex_pct,
            'wacc': average_wacc,
        }

        adjusted_assumptions = validate_assumptions(adjusted_assumptions)
        app.logger.debug(f"Historical data adjustments: {adjusted_assumptions}")
        return adjusted_assumptions
    except Exception as e:
        app.logger.exception(f"Error adjusting based on historical data for {stock_ticker}")
        return {}

def get_agent_importance_weights():
    app.logger.debug("Getting agent importance weights")
    prompt = """
You are a financial expert. Assign an importance weight between 0 and 1 to each of the following agents based on their role in providing financial assumptions for a DCF model. Consider that historical data is a strong indicator of future performance.

- sector_agent
- industry_agent
- sub_industry_agent
- scenario_agent
- company_agent
- feedback_agent
- historical_data_agent
- sentiment_agent

The sum of all importance weights should equal 1.

Provide the weights in valid JSON format like this:

{
  "sector_agent": ...,
  "industry_agent": ...,
  "sub_industry_agent": ...,
  "scenario_agent": ...,
  "company_agent": ...,
  "feedback_agent": ...,
  "historical_data_agent": ...,
  "sentiment_agent": ...
}

Do not include any comments or extra text.
"""
    assistant_reply = call_openai_api(prompt)
    agent_weights = parse_json_from_reply(assistant_reply)
    if agent_weights:
        total_weight = sum(agent_weights.values())
        if total_weight > 0:
            agent_weights = {k: v / total_weight for k, v in agent_weights.items()}
    else:
        agent_weights = {}
    app.logger.debug(f"Agent importance weights: {agent_weights}")
    return agent_weights

def validation_agent(final_adjustments, sector, industry, sub_industry, scenario, stock_ticker, agent_adjustments):
    app.logger.debug("Running validation agent")
    prompt = f"""
You are a Validation Agent tasked with reviewing and validating the financial assumptions provided by each agent. Your objectives are:

1. **Reasoning**: For each financial assumption, compare the values from all agents. Highlight agreements and discrepancies, especially where they diverge from historical data provided by the historical_data_agent. Consider:

   - Adherence to specified ranges.
   - Consistency with sector, industry, and company specifics.
   - Logical coherence among the assumptions.

2. **Confidence Scoring**: Assign a confidence score between 0 and 1 to each agent's output, reflecting:

   - Accuracy and reliability of the data.
   - Alignment with historical trends and current market conditions.
   - The agent's domain expertise.

3. **Output**: Provide your detailed reasoning and the confidence scores in valid JSON format:

{{
    "reasoning": "Your detailed reasoning here...",
    "confidence_scores": {{
        "sector_agent": ...,
        "industry_agent": ...,
        "sub_industry_agent": ...,
        "scenario_agent": ...,
        "company_agent": ...,
        "feedback_agent": ...,
        "historical_data_agent": ...,
        "sentiment_agent": ...
    }}
}}

**Important:** Ensure your reasoning is clear, and confidence scores accurately reflect your assessment. Do not include any comments or extra text outside the JSON.
"""
    assistant_reply = call_openai_api(prompt)
    response = parse_json_from_reply(assistant_reply)
    if response:
        reasoning = response.get("reasoning", "")
        confidence_scores = response.get("confidence_scores", {})
        app.logger.debug(f"Validation reasoning: {reasoning}")
        app.logger.debug(f"Confidence scores: {confidence_scores}")
        return reasoning, confidence_scores
    else:
        return "", {}

def compute_final_adjustments_with_agents(agent_adjustments, agent_importance_weights, agent_confidence_scores):
    app.logger.debug("Computing final adjustments with agents")
    assumptions = ['revenue_growth_rate', 'tax_rate', 'cogs_pct', 'wacc', 'terminal_growth_rate', 'operating_expenses_pct']
    final_adjustments = {}
    for assumption in assumptions:
        weighted_sum = 0
        total_weight = 0
        for agent, adjustments in agent_adjustments.items():
            agent_value = adjustments.get(assumption)
            if agent_value is not None:
                importance_weight = agent_importance_weights.get(agent, 0)
                confidence_score = agent_confidence_scores.get(agent, 1)
                final_weight = importance_weight * confidence_score
                weighted_sum += final_weight * agent_value
                total_weight += final_weight
        final_adjustments[assumption] = weighted_sum / total_weight if total_weight > 0 else 0
    app.logger.debug(f"Final adjustments: {final_adjustments}")
    return final_adjustments

def pre_validation_sanity_check(adjusted_assumptions, historical_wacc, historical_growth, stock_ticker):
    app.logger.debug("Running pre-validation sanity check")
    prompt = f"""
You are a financial analyst with deep knowledge of {stock_ticker}.
Proposed assumptions:
{json.dumps(adjusted_assumptions, indent=2)}

Historical WACC ~ {historical_wacc}, Historical revenue growth ~ {historical_growth}.

Check if assumptions are reasonable. If not, provide corrected assumptions in valid JSON without extra text:
{{
  "revenue_growth_rate": ...,
  "tax_rate": ...,
  "cogs_pct": ...,
  "wacc": ...,
  "terminal_growth_rate": ...,
  "operating_expenses_pct": ...
}}
"""
    assistant_reply = call_openai_api(prompt)
    corrected = parse_json_from_reply(assistant_reply)
    if corrected:
        corrected = validate_assumptions(corrected)
        app.logger.debug(f"Sanity check corrected assumptions: {corrected}")
        return corrected
    else:
        return adjusted_assumptions

def adjust_financial_variables(sector, industry, sub_industry, scenario, stock_ticker):
    app.logger.debug("Adjusting financial variables from all agents")
    sector_adjustments = validate_assumptions(adjust_for_sector(sector))
    app.logger.debug(f"After sector: {sector_adjustments}")
    industry_adjustments = validate_assumptions(adjust_for_industry(industry))
    app.logger.debug(f"After industry: {industry_adjustments}")
    sub_industry_adjustments = validate_assumptions(adjust_for_sub_industry(sub_industry))
    app.logger.debug(f"After sub-industry: {sub_industry_adjustments}")
    scenario_adjustments = validate_assumptions(adjust_for_scenario(scenario))
    app.logger.debug(f"After scenario: {scenario_adjustments}")
    company_adjustments = validate_assumptions(adjust_for_company(stock_ticker))
    app.logger.debug(f"After company: {company_adjustments}")
    feedback_adjustments = validate_assumptions(adjust_based_on_feedback(sector, industry, sub_industry, scenario))
    app.logger.debug(f"After feedback: {feedback_adjustments}")
    historical_data_adjustments = validate_assumptions(adjust_based_on_historical_data(stock_ticker))
    app.logger.debug(f"After historical data: {historical_data_adjustments}")
    sentiment_adjustments = validate_assumptions(adjust_based_on_sentiment(stock_ticker))
    app.logger.debug(f"After sentiment: {sentiment_adjustments}")

    agent_adjustments = {
        'sector_agent': sector_adjustments,
        'industry_agent': industry_adjustments,
        'sub_industry_agent': sub_industry_adjustments,
        'scenario_agent': scenario_adjustments,
        'company_agent': company_adjustments,
        'feedback_agent': feedback_adjustments,
        'historical_data_agent': historical_data_adjustments,
        'sentiment_agent': sentiment_adjustments
    }
    app.logger.debug(f"All agent adjustments collected: {agent_adjustments}")

    assumptions_keys = ['revenue_growth_rate', 'tax_rate', 'cogs_pct', 'wacc', 'terminal_growth_rate', 'operating_expenses_pct']
    combined_pre_sanity = {}
    for k in assumptions_keys:
        vals = [a[k] for a in agent_adjustments.values() if k in a]
        combined_pre_sanity[k] = sum(vals)/len(vals) if vals else 0.05
    app.logger.debug(f"Combined pre-sanity assumptions: {combined_pre_sanity}")

    historical_wacc = historical_data_adjustments.get('wacc', 0.10)
    historical_growth = historical_data_adjustments.get('revenue_growth_rate', 0.05)
    corrected_assumptions = pre_validation_sanity_check(combined_pre_sanity, historical_wacc, historical_growth, stock_ticker)
    agent_adjustments['sanity_check_agent'] = corrected_assumptions
    app.logger.debug(f"Agent adjustments after sanity check: {agent_adjustments}")

    agent_importance_weights = get_agent_importance_weights()
    reasoning, agent_confidence_scores = validation_agent({}, sector, industry, sub_industry, scenario, stock_ticker, agent_adjustments)
    final_adjustments = compute_final_adjustments_with_agents(agent_adjustments, agent_importance_weights, agent_confidence_scores)
    app.logger.debug(f"Final adjustments after weighting: {final_adjustments}")
    app.logger.debug(f"Validation reasoning: {reasoning}, Confidence scores: {agent_confidence_scores}")
    return final_adjustments, reasoning, agent_confidence_scores

class DCFModel:
    def __init__(self, initial_values, assumptions):
        self.initial_values = initial_values or {}
        self.assumptions = assumptions or {}
        self.years = 5
        self.fcf_values = []
        self.discounted_fcf = []
        self.intrinsic_value = 0
        self.intrinsic_value_per_share = 0
        self.discounted_terminal_value = 0
        self.terminal_value = 0
        self.projections = pd.DataFrame({'Year': range(1, self.years + 1)})

    def project_fcf(self):
        app.logger.debug("Projecting FCF")
        revenue = self.initial_values.get('Revenue', 0)
        revenue_growth_rate = self.assumptions.get('revenue_growth_rate', 0.05)
        tax_rate = self.assumptions.get('tax_rate', 0.21)
        cogs_pct = self.assumptions.get('cogs_pct', 0.6)
        operating_expenses_pct = self.assumptions.get('operating_expenses_pct', 0.2)
        depreciation_pct = self.assumptions.get('depreciation_pct', 0.05)
        capex_pct = self.assumptions.get('capex_pct', 0.05)
        nwc_pct = self.assumptions.get('nwc_pct', 0.2)

        prev_nwc = revenue * nwc_pct

        for year in range(self.years):
            projected_revenue = revenue * ((1 + revenue_growth_rate) ** (year + 1))
            cogs = projected_revenue * cogs_pct
            gross_profit = projected_revenue - cogs
            operating_expenses = projected_revenue * operating_expenses_pct
            depreciation = projected_revenue * depreciation_pct
            ebit = gross_profit - operating_expenses - depreciation
            nopat = ebit * (1 - tax_rate)
            capex = projected_revenue * capex_pct
            nwc = projected_revenue * nwc_pct
            change_in_nwc = nwc - prev_nwc
            prev_nwc = nwc
            fcf = nopat + depreciation - capex - change_in_nwc
            self.fcf_values.append(fcf)
            self.projections.loc[year, 'Revenue'] = projected_revenue
            self.projections.loc[year, 'COGS'] = cogs
            self.projections.loc[year, 'Operating Expenses'] = operating_expenses
            self.projections.loc[year, 'Depreciation'] = depreciation
            self.projections.loc[year, 'EBIT'] = ebit
            self.projections.loc[year, 'NOPAT'] = nopat
            self.projections.loc[year, 'CAPEX'] = capex
            self.projections.loc[year, 'Change in NWC'] = change_in_nwc
            self.projections.loc[year, 'FCF'] = fcf
            app.logger.debug(f"Year {year + 1}: Projected Revenue = {projected_revenue}, FCF = {fcf}")

    def calculate_terminal_value(self):
        terminal_growth_rate = self.assumptions.get('terminal_growth_rate', 0.02)
        wacc = self.assumptions.get('wacc', 0.10)
        last_fcf = self.fcf_values[-1]
        app.logger.debug(f"Calculating terminal value with wacc={wacc}, terminal_growth_rate={terminal_growth_rate}, last_fcf={last_fcf}")
        self.terminal_value = last_fcf * (1 + terminal_growth_rate) / (wacc - terminal_growth_rate)

    def discount_cash_flows(self):
        app.logger.debug("Discounting cash flows")
        wacc = self.assumptions.get('wacc', 0.10)
        for i, fcf in enumerate(self.fcf_values):
            discounted = fcf / ((1 + wacc) ** (i + 1))
            self.discounted_fcf.append(discounted)
        self.discounted_terminal_value = self.terminal_value / ((1 + wacc) ** self.years)

    def calculate_intrinsic_value(self):
        app.logger.debug("Calculating intrinsic value")
        self.intrinsic_value = sum(self.discounted_fcf) + self.discounted_terminal_value
        shares_outstanding = self.assumptions.get('shares_outstanding', 1)
        self.intrinsic_value_per_share = self.intrinsic_value / shares_outstanding

    def run_model(self):
        app.logger.debug("Running DCF model")
        self.project_fcf()
        self.calculate_terminal_value()
        self.discount_cash_flows()
        self.calculate_intrinsic_value()

    def get_results(self):
        app.logger.debug("Fetching DCF model results")
        return {
            "intrinsic_value": self.intrinsic_value,
            "intrinsic_value_per_share": self.intrinsic_value_per_share,
            "discounted_fcf": self.discounted_fcf,
            "discounted_terminal_value": self.discounted_terminal_value,
            "fcf_values": self.fcf_values,
            "projections": self.projections.to_dict(orient='records')
        }

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    app.logger.debug("Received file upload request")
    required_files = ['income_statement', 'balance_sheet', 'cash_flow_statement']
    files = {}
    for file_type in required_files:
        if file_type not in request.files:
            return jsonify({'error': f'No {file_type} part'}), 400
        file = request.files[file_type]
        if file.filename == '':
            return jsonify({'error': f'No selected file for {file_type}'}), 400
        if file and allowed_file(file.filename):
            files[file_type] = file
        else:
            return jsonify({'error': f'Invalid file type for {file_type}'}), 400

    data = {}
    for file_type, file in files.items():
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(file_path)
            file_data = process_uploaded_file(file_path, file_type)
            if file_data is None:
                return jsonify({'error': f'File processing failed or data invalid for {file_type}'}), 400
            data[file_type] = file_data
        except Exception as e:
            app.logger.exception("Error saving or processing file")
            return jsonify({'error': 'An error occurred while processing the files.'}), 500

    app.logger.debug(f"Files uploaded successfully: {data}")
    return jsonify({'message': 'Files uploaded successfully', 'data': data}), 200

def get_shares_outstanding(stock_ticker):
    app.logger.debug(f"Fetching shares outstanding for {stock_ticker}")
    try:
        company = yf.Ticker(stock_ticker)
        shares_outstanding = company.info.get('sharesOutstanding')
        if shares_outstanding:
            app.logger.debug(f"Shares outstanding: {shares_outstanding}")
            return shares_outstanding
        else:
            app.logger.warning(f"Shares outstanding not found for {stock_ticker}.")
            return None
    except Exception as e:
        app.logger.exception(f"Error fetching shares outstanding for {stock_ticker}")
        return None

@app.route('/calculate', methods=['POST'])
def calculate_intrinsic_value():
    app.logger.debug("Received /calculate request")
    try:
        content = request.get_json()
        data = content.get('data')
        user_assumptions = content.get('assumptions', {})
        sector = content.get('sector')
        industry = content.get('industry')
        sub_industry = content.get('sub_industry')
        scenario = content.get('scenario')
        stock_ticker = content.get('stock_ticker')

        app.logger.debug(f"Input content: sector={sector}, industry={industry}, sub_industry={sub_industry}, scenario={scenario}, stock_ticker={stock_ticker}")
        income_data = data.get('income_statement', {})
        balance_sheet_data = data.get('balance_sheet', {})
        cash_flow_data = data.get('cash_flow_statement', {})

        combined_data = {**income_data, **balance_sheet_data, **cash_flow_data}
        app.logger.debug(f"Combined data from uploads: {combined_data}")

        required_fields = ['Revenue', 'Operating Expenses', 'Depreciation', 'Capital Expenditures', 'Current Assets', 'Current Liabilities']
        missing_fields = [field for field in required_fields if field not in combined_data]
        if missing_fields:
            app.logger.error(f"Missing required financial data: {missing_fields}")
            return jsonify({'error': f'Missing required financial data: {missing_fields}'}), 400

        initial_revenue = combined_data['Revenue']
        combined_data['NWC'] = combined_data['Current Assets'] - combined_data['Current Liabilities']

        final_assumptions, validation_reasoning, agent_confidence_scores = adjust_financial_variables(
            sector, industry, sub_industry, scenario, stock_ticker
        )

        required_keys = ['revenue_growth_rate', 'tax_rate', 'cogs_pct', 'wacc', 'terminal_growth_rate', 'operating_expenses_pct']
        missing_keys = [key for key in required_keys if key not in final_assumptions]
        if missing_keys:
            app.logger.error(f"Adjusted assumptions missing keys: {missing_keys}")
            return jsonify({'error': f'Missing adjusted assumptions: {missing_keys}'}), 500

        final_assumptions.update(user_assumptions)

        final_assumptions['operating_expenses_pct'] = combined_data['Operating Expenses'] / initial_revenue
        final_assumptions['depreciation_pct'] = combined_data['Depreciation'] / initial_revenue
        final_assumptions['capex_pct'] = combined_data['Capital Expenditures'] / initial_revenue
        final_assumptions['nwc_pct'] = combined_data['NWC'] / initial_revenue

        shares_outstanding = final_assumptions.get('shares_outstanding')
        if shares_outstanding is None or shares_outstanding <= 0:
            shares_outstanding = get_shares_outstanding(stock_ticker)
            if shares_outstanding is None:
                shares_outstanding = 1
        else:
            shares_outstanding = float(shares_outstanding)
        final_assumptions['shares_outstanding'] = shares_outstanding

        # Log the final assumptions before running the model
        app.logger.debug(f"Final assumptions passed to DCFModel: {final_assumptions}")

        assumption_set = AssumptionSet(
            sector=sector,
            industry=industry,
            sub_industry=sub_industry,
            scenario=scenario,
            stock_ticker=stock_ticker,
            revenue_growth_rate=final_assumptions.get('revenue_growth_rate', 0.05),
            tax_rate=final_assumptions.get('tax_rate', 0.21),
            cogs_pct=final_assumptions.get('cogs_pct', 0.6),
            wacc=final_assumptions.get('wacc', 0.1),
            terminal_growth_rate=final_assumptions.get('terminal_growth_rate', 0.02),
            operating_expenses_pct=final_assumptions.get('operating_expenses_pct', 0.2)
        )
        db.session.add(assumption_set)
        db.session.commit()

        results = {}

        model = DCFModel(combined_data, final_assumptions)
        model.run_model()
        model_results = model.get_results()
        results.update(model_results)

        results['adjusted_assumptions'] = final_assumptions
        results['validation_reasoning'] = validation_reasoning
        results['agent_confidence_scores'] = agent_confidence_scores
        results['assumption_set_id'] = assumption_set.id

        return jsonify(results), 200

    except Exception as e:
        app.logger.exception("Error in /calculate route")
        return jsonify({'error': 'An error occurred during calculation.'}), 500

@app.route('/feedback', methods=['POST'])
def receive_feedback():
    app.logger.debug("Received /feedback request")
    try:
        content = request.get_json()
        sector = content.get('sector')
        industry = content.get('industry')
        sub_industry = content.get('sub_industry')
        scenario = content.get('scenario')
        score = content.get('score')
        comments = content.get('comments')
        assumption_feedback = content.get('assumption_feedback', {})
        assumption_set_id = content.get('assumption_set_id')

        if not assumption_set_id:
            return jsonify({'error': 'Assumption set ID is required.'}), 400

        assumption_set = AssumptionSet.query.get(assumption_set_id)
        if not assumption_set:
            return jsonify({'error': 'Assumption set not found.'}), 404

        feedback = Feedback(
            sector=sector,
            industry=industry,
            sub_industry=sub_industry,
            scenario=scenario,
            score=score,
            comments=comments,
            revenue_growth_feedback=assumption_feedback.get('revenue_growth_rate'),
            tax_rate_feedback=assumption_feedback.get('tax_rate'),
            cogs_pct_feedback=assumption_feedback.get('cogs_pct'),
            operating_expenses_feedback=assumption_feedback.get('operating_expenses_pct'),
            wacc_feedback=assumption_feedback.get('wacc'),
            assumption_set_id=assumption_set_id
        )
        db.session.add(feedback)
        db.session.commit()

        app.logger.debug("Feedback received successfully")
        return jsonify({'message': 'Feedback received'}), 200

    except Exception as e:
        db.session.rollback()
        app.logger.exception("Error saving feedback")
        return jsonify({'error': 'Failed to save feedback.'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
