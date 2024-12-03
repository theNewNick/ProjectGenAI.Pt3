// main.js

// Initialize variables with default values
let revenueGrowth = 5; // in percent
let taxRate = 21; // in percent
let operatingExpensesPct = 20; // Operating Expenses (% of Revenue)
let cogsPct = 60; // Cost of Goods Sold (% of Revenue)
let discountRate = 10; // in percent
let sharesOutstanding = 1; // Default shares outstanding
let assumptionSetId = null; // To store assumption_set_id from the server

// Update display values
function updateDisplayValues() {
    document.getElementById('revenueGrowthValue').innerText = `${revenueGrowth}%`;
    document.getElementById('taxRateValue').innerText = `${taxRate}%`;
    document.getElementById('operatingExpensesValue').innerText = `${operatingExpensesPct}%`;
    document.getElementById('cogsPctValue').innerText = `${cogsPct}%`;
    document.getElementById('discountRateValue').innerText = `${discountRate}%`;
}

updateDisplayValues();

// Add event listeners to the sliders
document.getElementById('revenueGrowth').addEventListener('input', (event) => {
    revenueGrowth = parseFloat(event.target.value);
    document.getElementById('revenueGrowthValue').innerText = `${revenueGrowth}%`;
    calculateProjections();
});

document.getElementById('taxRate').addEventListener('input', (event) => {
    taxRate = parseFloat(event.target.value);
    document.getElementById('taxRateValue').innerText = `${taxRate}%`;
    calculateProjections();
});

document.getElementById('operatingExpenses').addEventListener('input', (event) => {
    operatingExpensesPct = parseFloat(event.target.value);
    document.getElementById('operatingExpensesValue').innerText = `${operatingExpensesPct}%`;
    calculateProjections();
});

document.getElementById('cogsPct').addEventListener('input', (event) => {
    cogsPct = parseFloat(event.target.value);
    document.getElementById('cogsPctValue').innerText = `${cogsPct}%`;
    calculateProjections();
});

document.getElementById('discountRate').addEventListener('input', (event) => {
    discountRate = parseFloat(event.target.value);
    document.getElementById('discountRateValue').innerText = `${discountRate}%`;
    calculateProjections();
});

// Capture shares outstanding input
document.getElementById('sharesOutstanding').addEventListener('input', (event) => {
    sharesOutstanding = parseFloat(event.target.value);
    calculateProjections();
});

// Comprehensive data structure for sectors, industries, and sub-industries
const sectorIndustryData = {
    "Energy": {
        "Oil, Gas & Consumable Fuels": ["Integrated Oil & Gas", "Oil & Gas Exploration & Production", "Coal & Consumable Fuels"],
        "Energy Equipment & Services": ["Oil & Gas Drilling", "Oil & Gas Equipment & Services"]
    },
    "Materials": {
        "Chemicals": ["Commodity Chemicals", "Specialty Chemicals", "Industrial Gases"],
        "Metals & Mining": ["Aluminum", "Diversified Metals & Mining", "Gold", "Precious Metals & Minerals"],
        "Paper & Forest Products": ["Paper Products", "Forest Products"],
        "Containers & Packaging": ["Metal & Glass Containers", "Paper Packaging", "Plastic Packaging"],
        "Construction Materials": ["Construction Materials"]
    },
    "Industrials": {
        "Capital Goods": ["Aerospace & Defense", "Building Products", "Construction & Engineering", "Electrical Equipment", "Industrial Conglomerates", "Machinery", "Trading Companies & Distributors"],
        "Commercial & Professional Services": ["Commercial Services & Supplies", "Professional Services"],
        "Transportation": ["Air Freight & Logistics", "Airlines", "Marine", "Road & Rail", "Transportation Infrastructure"]
    },
    "Consumer Discretionary": {
        "Automobiles & Components": ["Auto Parts & Equipment", "Automobile Manufacturers", "Motorcycle Manufacturers"],
        "Consumer Durables & Apparel": ["Household Durables", "Leisure Products", "Textiles, Apparel & Luxury Goods"],
        "Consumer Services": ["Hotels, Restaurants & Leisure", "Diversified Consumer Services"],
        "Retailing": ["Distributors", "Internet & Direct Marketing Retail", "Multiline Retail", "Specialty Retail", "Home Improvement Retail"]
    },
    "Consumer Staples": {
        "Food & Staples Retailing": ["Drug Retail", "Food Distributors", "Food Retail", "Hypermarkets & Super Centers"],
        "Food, Beverage & Tobacco": ["Brewers", "Distillers & Vintners", "Soft Drinks", "Agricultural Products", "Packaged Foods & Meats", "Tobacco"],
        "Household & Personal Products": ["Household Products", "Personal Products"]
    },
    "Health Care": {
        "Health Care Equipment & Services": ["Health Care Equipment", "Health Care Supplies", "Health Care Providers & Services", "Health Care Technology"],
        "Pharmaceuticals, Biotechnology & Life Sciences": ["Biotechnology", "Pharmaceuticals", "Life Sciences Tools & Services"]
    },
    "Financials": {
        "Banks": ["Diversified Banks", "Regional Banks", "Thrifts & Mortgage Finance"],
        "Diversified Financials": ["Other Diversified Financial Services", "Multi-Sector Holdings", "Specialized Finance", "Consumer Finance", "Asset Management & Custody Banks", "Investment Banking & Brokerage"],
        "Insurance": ["Insurance Brokers", "Life & Health Insurance", "Multi-line Insurance", "Property & Casualty Insurance", "Reinsurance"]
    },
    "Information Technology": {
        "Software & Services": ["IT Consulting & Other Services", "Data Processing & Outsourced Services", "Application Software", "Systems Software"],
        "Technology Hardware & Equipment": ["Communications Equipment", "Technology Hardware, Storage & Peripherals", "Electronic Equipment & Instruments", "Electronic Components", "Electronic Manufacturing Services"],
        "Semiconductors & Semiconductor Equipment": ["Semiconductor Equipment", "Semiconductors"]
    },
    "Communication Services": {
        "Telecommunication Services": ["Alternative Carriers", "Integrated Telecommunication Services", "Wireless Telecommunication Services"],
        "Media & Entertainment": ["Advertising", "Broadcasting", "Cable & Satellite", "Movies & Entertainment", "Interactive Home Entertainment", "Interactive Media & Services"]
    },
    "Utilities": {
        "Electric Utilities": ["Electric Utilities"],
        "Gas Utilities": ["Gas Utilities"],
        "Multi-Utilities": ["Multi-Utilities"],
        "Water Utilities": ["Water Utilities"],
        "Independent Power and Renewable Electricity Producers": ["Independent Power Producers & Energy Traders", "Renewable Electricity"]
    },
    "Real Estate": {
        "Equity Real Estate Investment Trusts (REITs)": ["Diversified REITs", "Industrial REITs", "Hotel & Resort REITs", "Office REITs", "Health Care REITs", "Residential REITs", "Retail REITs", "Specialized REITs"],
        "Real Estate Management & Development": ["Real Estate Operating Companies", "Real Estate Development", "Real Estate Services"]
    }
};

// Populate sector options
function populateSectors() {
    const sectorSelect = document.getElementById('sectorSelect');
    sectorSelect.innerHTML = '<option value="">Select Sector</option>';
    for (const sector in sectorIndustryData) {
        const option = document.createElement('option');
        option.value = sector;
        option.text = sector;
        sectorSelect.appendChild(option);
    }
}

// Populate industry options based on selected sector
function populateIndustries() {
    const sectorSelect = document.getElementById('sectorSelect');
    const industrySelect = document.getElementById('industrySelect');
    const subIndustrySelect = document.getElementById('subIndustrySelect');
    const selectedSector = sectorSelect.value;

    industrySelect.innerHTML = '<option value="">Select Industry</option>';
    subIndustrySelect.innerHTML = '<option value="">Select Sub-Industry</option>';

    if (selectedSector && sectorIndustryData[selectedSector]) {
        const industries = sectorIndustryData[selectedSector];
        for (const industry in industries) {
            const option = document.createElement('option');
            option.value = industry;
            option.text = industry;
            industrySelect.appendChild(option);
        }
    }
}

// Populate sub-industry options based on selected industry
function populateSubIndustries() {
    const sectorSelect = document.getElementById('sectorSelect');
    const industrySelect = document.getElementById('industrySelect');
    const subIndustrySelect = document.getElementById('subIndustrySelect');
    const selectedSector = sectorSelect.value;
    const selectedIndustry = industrySelect.value;

    subIndustrySelect.innerHTML = '<option value="">Select Sub-Industry</option>';

    if (
        selectedSector &&
        selectedIndustry &&
        sectorIndustryData[selectedSector] &&
        sectorIndustryData[selectedSector][selectedIndustry]
    ) {
        const subIndustries = sectorIndustryData[selectedSector][selectedIndustry];
        subIndustries.forEach(subIndustry => {
            const option = document.createElement('option');
            option.value = subIndustry;
            option.text = subIndustry;
            subIndustrySelect.appendChild(option);
        });
    }
}

// Event listeners for sector and industry selection
document.getElementById('sectorSelect').addEventListener('change', () => {
    populateIndustries();
    populateSubIndustries();
});

document.getElementById('industrySelect').addEventListener('change', () => {
    populateSubIndustries();
});

// Initialize the dropdowns on page load
populateSectors();

// File upload handling
let uploadedData = null;

document.getElementById('uploadButton').addEventListener('click', () => {
    let incomeStatementFile = document.getElementById('incomeStatementInput').files[0];
    let balanceSheetFile = document.getElementById('balanceSheetInput').files[0];
    let cashFlowStatementFile = document.getElementById('cashFlowStatementInput').files[0];

    if (!incomeStatementFile || !balanceSheetFile || !cashFlowStatementFile) {
        alert('Please select all three files before uploading.');
        return;
    }

    let formData = new FormData();
    formData.append('income_statement', incomeStatementFile);
    formData.append('balance_sheet', balanceSheetFile);
    formData.append('cash_flow_statement', cashFlowStatementFile);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.data) {
            uploadedData = data.data;
            document.getElementById('fileUploadMessage').innerText = 'Files uploaded and processed successfully!';
        } else {
            alert('Error processing files: ' + data.error);
        }
    })
    .catch(error => {
        console.error('Error uploading files:', error);
        alert('Error uploading files.');
    });
});

// Calculate projections
document.getElementById('calculateButton').addEventListener('click', () => {
    if (!uploadedData) {
        alert('Please upload financial data first.');
        return;
    }

    calculateProjections(true); // Pass 'true' for initial calculation
});

function calculateProjections(isInitialCalculation = false) {
    if (!uploadedData) {
        return; // Do nothing if data is not uploaded yet
    }

    let sector = document.getElementById('sectorSelect').value;
    let industry = document.getElementById('industrySelect').value;
    let subIndustry = document.getElementById('subIndustrySelect').value;
    let scenario = document.getElementById('scenarioSelect').value;
    let stockTicker = document.getElementById('stockTicker').value;

    // Validate sharesOutstanding
    sharesOutstanding = parseFloat(document.getElementById('sharesOutstanding').value);
    if (isNaN(sharesOutstanding) || sharesOutstanding <= 0) {
        alert('Please enter a valid number of shares outstanding.');
        return;
    }

    // Validate sector, industry, sub-industry
    if (!sector || !industry || !subIndustry) {
        alert('Please select a sector, industry, and sub-industry.');
        return;
    }

    let updatedAssumptions = {
        shares_outstanding: sharesOutstanding
    };

    if (!isInitialCalculation) {
        // Include other assumptions from sliders
        updatedAssumptions.revenue_growth_rate = revenueGrowth / 100;
        updatedAssumptions.tax_rate = taxRate / 100;
        updatedAssumptions.operating_expenses_pct = operatingExpensesPct / 100;
        updatedAssumptions.cogs_pct = cogsPct / 100;
        updatedAssumptions.discount_rate = discountRate / 100;
        updatedAssumptions.terminal_growth_rate = 0.02;
    }

    fetch('/calculate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            data: uploadedData,
            assumptions: updatedAssumptions,
            sector: sector,
            industry: industry,
            sub_industry: subIndustry,
            scenario: scenario,
            stock_ticker: stockTicker
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.projections) {
            // Store assumption_set_id for later use
            assumptionSetId = data.assumption_set_id;

            if (isInitialCalculation) {
                // Initialize sliders with adjusted assumptions
                setSlidersWithAdjustedAssumptions(data.adjusted_assumptions);
            }
            displayProjections(data);
            updateChart(data);
        } else {
            alert('Error calculating projections.');
        }
    })
    .catch(error => {
        console.error('Error calculating projections:', error);
    });
}

function setSlidersWithAdjustedAssumptions(adjustedAssumptions) {
    // Update variables
    revenueGrowth = adjustedAssumptions.revenue_growth_rate * 100;
    taxRate = adjustedAssumptions.tax_rate * 100;
    operatingExpensesPct = adjustedAssumptions.operating_expenses_pct * 100;
    cogsPct = adjustedAssumptions.cogs_pct * 100;
    discountRate = adjustedAssumptions.discount_rate * 100;
    // Shares outstanding remains as input by the user

    // Update sliders
    document.getElementById('revenueGrowth').value = revenueGrowth;
    document.getElementById('taxRate').value = taxRate;
    document.getElementById('operatingExpenses').value = operatingExpensesPct;
    document.getElementById('cogsPct').value = cogsPct;
    document.getElementById('discountRate').value = discountRate;

    updateDisplayValues();
}

function displayProjections(data) {
    let outputDiv = document.getElementById('projectionsOutput');
    outputDiv.innerHTML = `<h4>Intrinsic Value per Share: $${data.intrinsic_value_per_share.toFixed(2)}</h4>`;
}

let ctx = document.getElementById('projectionChart').getContext('2d');
let projectionChart;

function updateChart(data) {
    let df = data.projections;
    let years = df.map(item => 'Year ' + item.Year);
    let revenues = df.map(item => item.Revenue);
    let fcfs = df.map(item => item.FCF);

    if (projectionChart) {
        projectionChart.destroy();
    }

    projectionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: years,
            datasets: [
                {
                    label: 'Revenue',
                    data: revenues,
                    borderColor: 'blue',
                    fill: false
                },
                {
                    label: 'Free Cash Flow',
                    data: fcfs,
                    borderColor: 'purple',
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Financial Projections Over 5 Years'
                }
            },
            scales: {
                y: {
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toLocaleString();
                        }
                    }
                }
            }
        }
    });
}

// Feedback form submission
document.getElementById('feedbackForm').addEventListener('submit', (event) => {
    event.preventDefault();

    let sector = document.getElementById('sectorSelect').value;
    let industry = document.getElementById('industrySelect').value;
    let subIndustry = document.getElementById('subIndustrySelect').value;
    let scenario = document.getElementById('scenarioSelect').value;
    let score = parseInt(document.getElementById('accuracyRating').value);
    let comments = document.getElementById('feedbackComments').value;

    // Collect assumption feedback
    let assumptionFeedback = {
        revenue_growth_rate: document.getElementById('revenueGrowthFeedback').value,
        tax_rate: document.getElementById('taxRateFeedback').value,
        cogs_pct: document.getElementById('cogsPctFeedback').value,
        operating_expenses_pct: document.getElementById('operatingExpensesFeedback').value,
        discount_rate: document.getElementById('discountRateFeedback').value
    };

    // Check if assumptionSetId is available
    if (!assumptionSetId) {
        alert('Please perform the calculations first before submitting feedback.');
        return;
    }

    fetch('/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            sector: sector,
            industry: industry,
            sub_industry: subIndustry,
            scenario: scenario,
            score: score,
            comments: comments,
            assumption_feedback: assumptionFeedback,
            assumption_set_id: assumptionSetId  // Include the assumption_set_id
        })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('feedbackMessage').innerText = data.message;
    })
    .catch(error => {
        console.error('Error submitting feedback:', error);
    });

    // Reset form
    document.getElementById('feedbackForm').reset();
});
