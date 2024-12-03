import os
import json
import logging
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

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure upload folder and maximum file size
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx'}

# Configure the SQLite database with absolute path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(BASE_DIR, "feedback.db")
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set it in the environment variables.")

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
    discount_rate = db.Column(db.Float, nullable=False)
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

    # New fields for assumption feedback
    revenue_growth_feedback = db.Column(db.String(20), nullable=True)
    tax_rate_feedback = db.Column(db.String(20), nullable=True)
    cogs_pct_feedback = db.Column(db.String(20), nullable=True)
    operating_expenses_feedback = db.Column(db.String(20), nullable=True)
    discount_rate_feedback = db.Column(db.String(20), nullable=True)
    # Foreign key to AssumptionSet
    assumption_set_id = db.Column(db.Integer, db.ForeignKey('assumption_set.id'), nullable=False)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to summarize feedback from the database
def summarize_feedback(sector, industry, sub_industry, scenario):
    try:
        # Filter feedback by sub-industry and scenario, joining AssumptionSet
        feedback_entries = Feedback.query.join(AssumptionSet).filter(
            AssumptionSet.sector == sector,
            AssumptionSet.industry == industry,
            AssumptionSet.sub_industry == sub_industry,
            AssumptionSet.scenario == scenario
        ).all()

        if not feedback_entries:
            return "No relevant feedback available."

        # Collect feedback counts for each assumption
        assumption_feedback_counts = {
            'revenue_growth_rate': {'too_low': 0, 'about_right': 0, 'too_high': 0},
            'tax_rate': {'too_low': 0, 'about_right': 0, 'too_high': 0},
            'cogs_pct': {'too_low': 0, 'about_right': 0, 'too_high': 0},
            'operating_expenses_pct': {'too_low': 0, 'about_right': 0, 'too_high': 0},
            'discount_rate': {'too_low': 0, 'about_right': 0, 'too_high': 0},
        }

        for entry in feedback_entries:
            for assumption in assumption_feedback_counts.keys():
                feedback_value = getattr(entry, f"{assumption}_feedback")
                if feedback_value in assumption_feedback_counts[assumption]:
                    assumption_feedback_counts[assumption][feedback_value] += 1

        # Build summary
        summary_lines = []
        for assumption, counts in assumption_feedback_counts.items():
            total = sum(counts.values())
            if total > 0:
                summary = f"{assumption.replace('_', ' ').title()}:"
                for feedback_value, count in counts.items():
                    percentage = (count / total) * 100
                    summary += f" {percentage:.1f}% {feedback_value.replace('_', ' ')};"
                summary_lines.append(summary)

        # Include general comments
        comments = [f"- {entry.comments}" for entry in feedback_entries if entry.comments]
        if comments:
            summary_lines.append("\nUser Comments:")
            summary_lines.extend(comments)

        return "\n".join(summary_lines)
    except Exception as e:
        logger.error(f"Error summarizing feedback: {e}\n{traceback.format_exc()}")
        return "Error retrieving feedback."

# Helper function to validate assumptions
def validate_assumptions(adjusted_assumptions):
    # Define the valid ranges and default values for each assumption
    ranges = {
        'revenue_growth_rate': (0.0, 1.0, 0.05),     # Default to 5%
        'tax_rate': (0.01, 0.5, 0.21),               # Default to 21%
        'cogs_pct': (0.01, 1.0, 0.6),                # Default to 60%
        'discount_rate': (0.01, 0.2, 0.10),          # Default to 10%
        'terminal_growth_rate': (0.0, 0.05, 0.02),   # Default to 2%
        'operating_expenses_pct': (0.01, 1.0, 0.2)   # Default to 20%
    }
    validated_assumptions = {}
    for key, (min_val, max_val, default_val) in ranges.items():
        value = adjusted_assumptions.get(key)
        if value is not None:
            try:
                # Ensure the value is a float
                value = float(value)
                # If percentage values are over 1, assume they are in percent form and convert
                if value > 1.0:
                    value = value / 100.0
                # Clamp the value within the specified range
                value = max(min_val, min(value, max_val))
            except (ValueError, TypeError):
                value = default_val  # Use default value if invalid
        else:
            value = default_val  # Use default value if missing
        validated_assumptions[key] = value
    return validated_assumptions

# Helper function to call OpenAI API
def call_openai_api(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.2,  # Low temperature for consistency
            messages=[
                {"role": "system", "content": "You are an expert financial analyst."},
                {"role": "user", "content": prompt}
            ]
        )
        assistant_reply = response['choices'][0]['message']['content']
        json_str = re.search(r'\{.*\}', assistant_reply, re.DOTALL)
        if json_str:
            adjusted_assumptions = json.loads(json_str.group())
            logger.info(f"Agent output: {adjusted_assumptions}")
            return adjusted_assumptions
        else:
            logger.error("Failed to extract JSON from the assistant's reply.")
            return {}
    except Exception as e:
        logger.error(f"Error in OpenAI API call: {e}")
        return {}

# Individual agents
def adjust_for_sector(sector):
    prompt = f"""
    As a financial analyst specializing in the {sector} sector, provide typical financial assumptions that reflect current industry trends.

    Please provide reasonable values for the following financial assumptions, ensuring they are within the specified ranges:

    - revenue_growth_rate (as a decimal, e.g., 0.05 for 5%, between 0% and 1.0)
    - tax_rate (between 0.01 and 0.5)
    - cogs_pct (Cost of Goods Sold as a percentage of Revenue, between 0.01 and 1.0)
    - discount_rate (between 0.01 and 0.2)
    - terminal_growth_rate (between 0.0 and 0.05)
    - operating_expenses_pct (Operating Expenses as a percentage of Revenue, between 0.01 and 1.0)

    Return the results in JSON format without any additional text.
    """
    adjusted_assumptions = call_openai_api(prompt)
    return adjusted_assumptions

def adjust_for_industry(industry):
    prompt = f"""
    As a financial analyst specializing in the {industry} industry, provide typical financial assumptions that reflect current industry trends.

    Please provide reasonable values for the following financial assumptions, ensuring they are within the specified ranges:

    - revenue_growth_rate (as a decimal, e.g., 0.05 for 5%, between 0% and 1.0)
    - tax_rate (between 0.01 and 0.5)
    - cogs_pct (Cost of Goods Sold as a percentage of Revenue, between 0.01 and 1.0)
    - discount_rate (between 0.01 and 0.2)
    - terminal_growth_rate (between 0.0 and 0.05)
    - operating_expenses_pct (Operating Expenses as a percentage of Revenue, between 0.01 and 1.0)

    Return the results in JSON format without any additional text.
    """
    adjusted_assumptions = call_openai_api(prompt)
    return adjusted_assumptions

def adjust_for_sub_industry(sub_industry):
    prompt = f"""
    As a financial analyst specializing in the {sub_industry} sub-industry, provide typical financial assumptions that reflect current industry trends.

    Please provide reasonable values for the following financial assumptions, ensuring they are within the specified ranges:

    - revenue_growth_rate (as a decimal, e.g., 0.05 for 5%, between 0% and 1.0)
    - tax_rate (between 0.01 and 0.5)
    - cogs_pct (Cost of Goods Sold as a percentage of Revenue, between 0.01 and 1.0)
    - discount_rate (between 0.01 and 0.2)
    - terminal_growth_rate (between 0.0 and 0.05)
    - operating_expenses_pct (Operating Expenses as a percentage of Revenue, between 0.01 and 1.0)

    Return the results in JSON format without any additional text.
    """
    adjusted_assumptions = call_openai_api(prompt)
    return adjusted_assumptions

def adjust_for_scenario(scenario):
    prompt = f"""
    As a financial analyst, provide financial assumptions appropriate for a {scenario} scenario.

    Please provide reasonable values for the following financial assumptions, ensuring they are within the specified ranges:

    - revenue_growth_rate (as a decimal, e.g., 0.05 for 5%, between 0% and 1.0)
    - tax_rate (between 0.01 and 0.5)
    - cogs_pct (Cost of Goods Sold as a percentage of Revenue, between 0.01 and 1.0)
    - discount_rate (between 0.01 and 0.2)
    - terminal_growth_rate (between 0.0 and 0.05)
    - operating_expenses_pct (Operating Expenses as a percentage of Revenue, between 0.01 and 1.0)

    Return the results in JSON format without any additional text.
    """
    adjusted_assumptions = call_openai_api(prompt)
    return adjusted_assumptions

def adjust_based_on_feedback(sector, industry, sub_industry, scenario):
    feedback_summary = summarize_feedback(sector, industry, sub_industry, scenario)
    prompt = f"""
    Based on the following user feedback for the {sector} sector, {industry} industry, and {sub_industry} sub-industry under a {scenario} scenario:
    {feedback_summary}

    Please provide adjusted financial assumptions, ensuring they are within the specified ranges:

    - revenue_growth_rate (as a decimal, e.g., 0.05 for 5%, between 0% and 1.0)
    - tax_rate (between 0.01 and 0.5)
    - cogs_pct (Cost of Goods Sold as a percentage of Revenue, between 0.01 and 1.0)
    - discount_rate (between 0.01 and 0.2)
    - terminal_growth_rate (between 0.0 and 0.05)
    - operating_expenses_pct (Operating Expenses as a percentage of Revenue, between 0.01 and 1.0)

    Take into account whether users found the assumptions too high or too low, and adjust accordingly.

    Return the results in JSON format without any additional text.
    """
    adjusted_assumptions = call_openai_api(prompt)
    return adjusted_assumptions

# New agent: Adjust for Company
def adjust_for_company(stock_ticker):
    try:
        # Fetch company-specific data using yfinance
        company = yf.Ticker(stock_ticker)
        info = company.info

        # Extract relevant company information
        company_name = info.get('longName', 'the company')
        industry = info.get('industry', 'the industry')
        business_summary = info.get('longBusinessSummary', '')

        prompt = f"""
        As a financial analyst, analyze {company_name} ({stock_ticker}), which operates in the {industry}. Considering its specific business operations and market position:

        {business_summary}

        Provide adjusted financial assumptions that are appropriate for this company, ensuring they are within the specified ranges:

        - revenue_growth_rate (as a decimal, e.g., 0.05 for 5%, between 0% and 1.0)
        - tax_rate (between 0.01 and 0.5)
        - cogs_pct (between 0.01 and 1.0)
        - discount_rate (between 0.01 and 0.2)
        - terminal_growth_rate (between 0.0 and 0.05)
        - operating_expenses_pct (between 0.01 and 1.0)

        Return the results in JSON format without any additional text.
        """
        adjusted_assumptions = call_openai_api(prompt)
        return adjusted_assumptions
    except Exception as e:
        logger.error(f"Error adjusting for company {stock_ticker}: {e}")
        return {}

# Consensus agent with dynamic weighting
def merge_adjustments(adjustments_list, initial_agent_weights):
    # Map each agent to its corresponding adjustments
    agent_names = ['sector', 'industry', 'sub_industry', 'scenario', 'company', 'feedback']
    agent_adjustments = dict(zip(agent_names, adjustments_list))

    # Identify valid agents
    valid_agents = {}
    for agent_name, adjustments in agent_adjustments.items():
        if evaluate_agent_output(adjustments):
            valid_agents[agent_name] = adjustments
        else:
            logger.warning(f"Agent '{agent_name}' provided invalid or incomplete data.")

    if not valid_agents:
        logger.error("No valid agent outputs available.")
        return {}  # Return empty or default values as appropriate

    # Adjust weights dynamically
    adjusted_weights = adjust_weights_dynamically(initial_agent_weights, valid_agents)

    # Initialize final adjustments dictionary
    final_adjustments = {}
    keys = ['revenue_growth_rate', 'tax_rate', 'cogs_pct', 'discount_rate', 'terminal_growth_rate', 'operating_expenses_pct']

    for key in keys:
        weighted_sum = 0.0
        total_weight = 0.0
        for agent_name, adjustments in valid_agents.items():
            value = adjustments.get(key)
            if value is not None and isinstance(value, (int, float)):
                weight = adjusted_weights.get(agent_name, 0)
                weighted_sum += value * weight
                total_weight += weight
        if total_weight > 0:
            final_adjustments[key] = weighted_sum / total_weight
        else:
            final_adjustments[key] = 0.0  # Default value if no valid data
    return final_adjustments

def evaluate_agent_output(output):
    # Define acceptable ranges for each assumption
    ranges = {
        'revenue_growth_rate': (0.0, 1.0),
        'tax_rate': (0.01, 0.5),
        'cogs_pct': (0.01, 1.0),
        'discount_rate': (0.01, 0.2),
        'terminal_growth_rate': (0.0, 0.05),
        'operating_expenses_pct': (0.01, 1.0)
    }
    # Check completeness and validity
    for key, (min_val, max_val) in ranges.items():
        value = output.get(key)
        if value is None or not (min_val <= value <= max_val):
            return False  # Output is invalid or incomplete
    return True  # Output is valid and complete

def adjust_weights_dynamically(initial_weights, valid_agents):
    total_initial_weight = sum(initial_weights.values())
    total_valid_weight = sum(initial_weights[agent] for agent in valid_agents)
    adjusted_weights = {}
    for agent in valid_agents:
        # Proportionally adjust the weight
        weight = initial_weights[agent]
        adjusted_weight = weight / total_valid_weight * total_initial_weight
        adjusted_weights[agent] = adjusted_weight
    return adjusted_weights

# Function to adjust financial variables using multi-agent system
def adjust_financial_variables(sector, industry, sub_industry, scenario, stock_ticker):
    # Run agents to get adjustments
    sector_adjustments = adjust_for_sector(sector)
    industry_adjustments = adjust_for_industry(industry)
    sub_industry_adjustments = adjust_for_sub_industry(sub_industry)
    scenario_adjustments = adjust_for_scenario(scenario)
    company_adjustments = adjust_for_company(stock_ticker)  # New agent
    feedback_adjustments = adjust_based_on_feedback(sector, industry, sub_industry, scenario)

    # Validate each agent's adjustments
    sector_adjustments = validate_assumptions(sector_adjustments)
    industry_adjustments = validate_assumptions(industry_adjustments)
    sub_industry_adjustments = validate_assumptions(sub_industry_adjustments)
    scenario_adjustments = validate_assumptions(scenario_adjustments)
    company_adjustments = validate_assumptions(company_adjustments)
    feedback_adjustments = validate_assumptions(feedback_adjustments)

    # Initial weights as per your weighting system
    initial_agent_weights = {
        'sector': 0.1,
        'industry': 0.1,
        'sub_industry': 0.1,
        'scenario': 0.25,
        'company': 0.3,
        'feedback': 0.15
    }

    # Combine adjustments using the Consensus Agent with dynamic weighting
    final_adjustments = merge_adjustments([
        sector_adjustments,
        industry_adjustments,
        sub_industry_adjustments,
        scenario_adjustments,
        company_adjustments,
        feedback_adjustments
    ], initial_agent_weights)

    return final_adjustments

# Data processing functions
def process_uploaded_file(file_path, file_type):
    try:
        # Read the file with headers
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.csv':
            data = pd.read_csv(file_path)
        elif file_extension in ['.xls', '.xlsx']:
            data = pd.read_excel(file_path)
        else:
            return None

        data = data.fillna(0)
        # Process data based on file type
        if file_type == 'income_statement':
            return process_income_statement(data)
        elif file_type == 'balance_sheet':
            return process_balance_sheet(data)
        elif file_type == 'cash_flow_statement':
            return process_cash_flow_statement(data)
        else:
            return None
    except Exception as e:
        logger.error(f"Error processing {file_type}: {e}\n{traceback.format_exc()}")
        return None
    finally:
        # Delete the file after processing
        if os.path.exists(file_path):
            os.remove(file_path)

def extract_fields(data, required_fields):
    labels = data.iloc[:, 0].astype(str).str.strip().tolist()
    data_values = data.iloc[:, 1:].applymap(lambda x: float(str(x).replace(',', '').replace('(', '-').replace(')', '')))

    data_dict = dict(zip(labels, data_values.values.tolist()))
    existing_labels = list(data_dict.keys())

    field_mapping = get_field_mappings(required_fields, existing_labels)
    if not field_mapping:
        logger.error("Failed to obtain field mappings.")
        return None

    processed_data = {}
    for field, label in field_mapping.items():
        try:
            values = data_dict[label]
            processed_data[field] = values[0]  # Assuming single-period data
        except KeyError:
            logger.error(f"Label '{label}' not found in data dictionary.")
            return None

    return processed_data

def process_income_statement(data):
    required_fields = ['Revenue', 'COGS', 'Operating Expenses']
    return extract_fields(data, required_fields)

def process_balance_sheet(data):
    required_fields = ['Current Assets', 'Current Liabilities']
    return extract_fields(data, required_fields)

def process_cash_flow_statement(data):
    required_fields = ['Depreciation', 'Capital Expenditures']
    return extract_fields(data, required_fields)

# Function to get OpenAI mappings for all required fields
def get_field_mappings(required_fields, existing_labels):
    prompt = f"""
    You are a financial data expert. We have a list of required financial fields: {', '.join(required_fields)}.
    Given the following available data labels from a financial dataset:
    {', '.join(existing_labels)}
    Please map each required field to the closest matching label from the available labels.
    Provide the mappings in JSON format, where each key is the required field and the value is the matching label.
    Only use labels exactly as they appear in the available labels.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        assistant_reply = response['choices'][0]['message']['content']
        # Extract JSON using regex
        json_str = re.search(r'\{.*\}', assistant_reply, re.DOTALL)
        if json_str:
            field_mapping = json.loads(json_str.group())
            return field_mapping
        else:
            logger.error("Failed to extract JSON from the assistant's reply.")
            return None
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return None

# DCF Model
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
            # Save projections
            self.projections.loc[year, 'Revenue'] = projected_revenue
            self.projections.loc[year, 'COGS'] = cogs
            self.projections.loc[year, 'Operating Expenses'] = operating_expenses
            self.projections.loc[year, 'Depreciation'] = depreciation
            self.projections.loc[year, 'EBIT'] = ebit
            self.projections.loc[year, 'NOPAT'] = nopat
            self.projections.loc[year, 'CAPEX'] = capex
            self.projections.loc[year, 'Change in NWC'] = change_in_nwc
            self.projections.loc[year, 'FCF'] = fcf
            logger.info(f"Year {year + 1}: Projected Revenue = {projected_revenue}, FCF = {fcf}")

    def calculate_terminal_value(self):
        terminal_growth_rate = self.assumptions.get('terminal_growth_rate', 0.02)
        discount_rate = self.assumptions.get('discount_rate', 0.10)
        last_fcf = self.fcf_values[-1]
        self.terminal_value = last_fcf * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)

    def discount_cash_flows(self):
        discount_rate = self.assumptions.get('discount_rate', 0.10)
        # Discount FCFs
        for i, fcf in enumerate(self.fcf_values):
            discounted = fcf / ((1 + discount_rate) ** (i + 1))
            self.discounted_fcf.append(discounted)
        # Discount Terminal Value
        self.discounted_terminal_value = self.terminal_value / ((1 + discount_rate) ** self.years)

    def calculate_intrinsic_value(self):
        self.intrinsic_value = sum(self.discounted_fcf) + self.discounted_terminal_value
        shares_outstanding = self.assumptions.get('shares_outstanding', 1)
        self.intrinsic_value_per_share = self.intrinsic_value / shares_outstanding

    def run_model(self):
        self.project_fcf()
        self.calculate_terminal_value()
        self.discount_cash_flows()
        self.calculate_intrinsic_value()

    def get_results(self):
        return {
            "intrinsic_value": self.intrinsic_value,
            "intrinsic_value_per_share": self.intrinsic_value_per_share,
            "discounted_fcf": self.discounted_fcf,
            "discounted_terminal_value": self.discounted_terminal_value,
            "fcf_values": self.fcf_values,
            "projections": self.projections.to_dict(orient='records')
        }

# Flask routes
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
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
            logger.error(f"Error saving or processing file: {e}")
            return jsonify({'error': 'An error occurred while processing the files.'}), 500

    return jsonify({'message': 'Files uploaded successfully', 'data': data}), 200

@app.route('/calculate', methods=['POST'])
def calculate_intrinsic_value():
    try:
        content = request.get_json()
        data = content.get('data')
        user_assumptions = content.get('assumptions', {})
        sector = content.get('sector')
        industry = content.get('industry')
        sub_industry = content.get('sub_industry')
        scenario = content.get('scenario')
        stock_ticker = content.get('stock_ticker')

        # Merge data from different statements
        income_data = data.get('income_statement', {})
        balance_sheet_data = data.get('balance_sheet', {})
        cash_flow_data = data.get('cash_flow_statement', {})

        # Combine all data
        combined_data = {**income_data, **balance_sheet_data, **cash_flow_data}

        # Calculate percentages for assumptions
        initial_revenue = combined_data['Revenue']
        combined_data['NWC'] = combined_data['Current Assets'] - combined_data['Current Liabilities']

        # Adjusted assumptions from AI
        adjusted_assumptions = adjust_financial_variables(sector, industry, sub_industry, scenario, stock_ticker)

        # Merge AI-generated assumptions with user adjustments
        final_assumptions = adjusted_assumptions.copy()
        final_assumptions.update(user_assumptions)

        # Calculate percentages based on actual financial data
        final_assumptions['operating_expenses_pct'] = combined_data['Operating Expenses'] / initial_revenue
        final_assumptions['depreciation_pct'] = combined_data['Depreciation'] / initial_revenue
        final_assumptions['capex_pct'] = combined_data['Capital Expenditures'] / initial_revenue
        final_assumptions['nwc_pct'] = combined_data['NWC'] / initial_revenue

        # Ensure shares_outstanding is included
        shares_outstanding = final_assumptions.get('shares_outstanding')
        if shares_outstanding is None or shares_outstanding <= 0:
            shares_outstanding = get_shares_outstanding(stock_ticker)
            if shares_outstanding is None:
                shares_outstanding = 1  # Default to 1 to avoid division by zero
        else:
            shares_outstanding = float(shares_outstanding)
        final_assumptions['shares_outstanding'] = shares_outstanding

        # Save the generated assumptions to the database
        assumption_set = AssumptionSet(
            sector=sector,
            industry=industry,
            sub_industry=sub_industry,
            scenario=scenario,
            stock_ticker=stock_ticker,
            revenue_growth_rate=adjusted_assumptions.get('revenue_growth_rate', 0.05),
            tax_rate=adjusted_assumptions.get('tax_rate', 0.21),
            cogs_pct=adjusted_assumptions.get('cogs_pct', 0.6),
            discount_rate=adjusted_assumptions.get('discount_rate', 0.1),
            terminal_growth_rate=adjusted_assumptions.get('terminal_growth_rate', 0.02),
            operating_expenses_pct=adjusted_assumptions.get('operating_expenses_pct', 0.2)
        )
        db.session.add(assumption_set)
        db.session.commit()

        # Include the assumption_set_id in the results to pass to the front-end
        results = {}

        model = DCFModel(combined_data, final_assumptions)
        model.run_model()
        model_results = model.get_results()

        results.update(model_results)

        results['adjusted_assumptions'] = adjusted_assumptions
        results['assumption_set_id'] = assumption_set.id  # Include assumption_set_id

        return jsonify(results), 200

    except Exception as e:
        logger.error(f"Error in /calculate route: {e}\n{traceback.format_exc()}")
        return jsonify({'error': 'An error occurred during calculation.'}), 500

def get_shares_outstanding(stock_ticker):
    try:
        company = yf.Ticker(stock_ticker)
        shares_outstanding = company.info.get('sharesOutstanding')
        if shares_outstanding:
            return shares_outstanding
        else:
            logger.warning(f"Shares outstanding not found for {stock_ticker}.")
            return None
    except Exception as e:
        logger.error(f"Error fetching shares outstanding for {stock_ticker}: {e}")
        return None

@app.route('/feedback', methods=['POST'])
def receive_feedback():
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

        # Retrieve the corresponding AssumptionSet
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
            discount_rate_feedback=assumption_feedback.get('discount_rate'),
            assumption_set_id=assumption_set_id
        )
        db.session.add(feedback)
        db.session.commit()

        return jsonify({'message': 'Feedback received'}), 200

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error saving feedback: {e}\n{traceback.format_exc()}")
        return jsonify({'error': 'Failed to save feedback.'}), 500

if __name__ == '__main__':
    # Initialize the database
    try:
        with app.app_context():
            db.create_all()
            print("Database and tables created successfully.")
            logger.debug("Database and tables created successfully.")
            # Print the database URI and path
            print(f"Database URI: {app.config['SQLALCHEMY_DATABASE_URI']}")
            print(f"Database file should be at: {db_path}")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        print(f"Error creating database tables: {e}")

    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True)
