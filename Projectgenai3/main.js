// Initialize variables with default values
let revenueGrowth = 5; // in percent
let taxRate = 25; // in percent
let operationalExpenses = 500000; // in dollars
let cogs = 40; // Cost of Goods Sold (% of Revenue)
let discountRate = 10; // in percent
let interestRate = 5; // in percent

// Scenario defaults
const scenarios = {
    optimistic: {
        revenueGrowth: 10,
        taxRate: 20,
        operationalExpenses: 400000,
        cogs: 35,
        discountRate: 8,
        interestRate: 4,
    },
    neutral: {
        revenueGrowth: 5,
        taxRate: 25,
        operationalExpenses: 500000,
        cogs: 40,
        discountRate: 10,
        interestRate: 5,
    },
    pessimistic: {
        revenueGrowth: 2,
        taxRate: 30,
        operationalExpenses: 600000,
        cogs: 45,
        discountRate: 12,
        interestRate: 6,
    },
};

// Update display values
function updateDisplayValues() {
    document.getElementById('revenueGrowthValue').innerText = `${revenueGrowth}%`;
    document.getElementById('taxRateValue').innerText = `${taxRate}%`;
    document.getElementById('operationalExpensesValue').innerText = `$${operationalExpenses.toLocaleString()}`;
    document.getElementById('cogsValue').innerText = `${cogs}%`;
    document.getElementById('discountRateValue').innerText = `${discountRate}%`;
    document.getElementById('interestRateValue').innerText = `${interestRate}%`;
}

updateDisplayValues();

// Add event listeners to the sliders
document.getElementById('revenueGrowth').addEventListener('input', (event) => {
    revenueGrowth = parseFloat(event.target.value);
    document.getElementById('revenueGrowthValue').innerText = `${revenueGrowth}%`;
    updateChart();
});

document.getElementById('taxRate').addEventListener('input', (event) => {
    taxRate = parseFloat(event.target.value);
    document.getElementById('taxRateValue').innerText = `${taxRate}%`;
    updateChart();
});

document.getElementById('operationalExpenses').addEventListener('input', (event) => {
    operationalExpenses = parseFloat(event.target.value);
    document.getElementById('operationalExpensesValue').innerText = `$${operationalExpenses.toLocaleString()}`;
    updateChart();
});

document.getElementById('cogs').addEventListener('input', (event) => {
    cogs = parseFloat(event.target.value);
    document.getElementById('cogsValue').innerText = `${cogs}%`;
    updateChart();
});

document.getElementById('discountRate').addEventListener('input', (event) => {
    discountRate = parseFloat(event.target.value);
    document.getElementById('discountRateValue').innerText = `${discountRate}%`;
    updateChart();
});

document.getElementById('interestRate').addEventListener('input', (event) => {
    interestRate = parseFloat(event.target.value);
    document.getElementById('interestRateValue').innerText = `${interestRate}%`;
    updateChart();
});

// Scenario selection
document.getElementById('scenarioSelect').addEventListener('change', (event) => {
    const scenario = event.target.value;
    applyScenario(scenario);
});

function applyScenario(scenario) {
    const settings = scenarios[scenario];
    revenueGrowth = settings.revenueGrowth;
    taxRate = settings.taxRate;
    operationalExpenses = settings.operationalExpenses;
    cogs = settings.cogs;
    discountRate = settings.discountRate;
    interestRate = settings.interestRate;

    // Update sliders
    document.getElementById('revenueGrowth').value = revenueGrowth;
    document.getElementById('taxRate').value = taxRate;
    document.getElementById('operationalExpenses').value = operationalExpenses;
    document.getElementById('cogs').value = cogs;
    document.getElementById('discountRate').value = discountRate;
    document.getElementById('interestRate').value = interestRate;

    updateDisplayValues();
    updateChart();
}

// Placeholder data for initial revenue and profit
let initialRevenue = 1000000; // $1,000,000

// Function to calculate projections
function calculateProjections() {
    let years = [1, 2, 3, 4, 5];
    let revenues = [];
    let profits = [];
    let npv = [];

    let revenue = initialRevenue;

    for (let i = 0; i < years.length; i++) {
        // Calculate revenue growth
        revenue = revenue * (1 + revenueGrowth / 100);
        // Calculate COGS
        let cogsAmount = revenue * (cogs / 100);
        // Calculate gross profit
        let grossProfit = revenue - cogsAmount;
        // Calculate operational profit
        let operatingProfit = grossProfit - operationalExpenses;
        // Calculate interest expense
        let interestExpense = operatingProfit * (interestRate / 100);
        // Calculate taxable income
        let taxableIncome = operatingProfit - interestExpense;
        // Calculate taxes
        let taxes = taxableIncome * (taxRate / 100);
        // Calculate net profit
        let netProfit = taxableIncome - taxes;
        // Discounted cash flow
        let discountedCashFlow = netProfit / Math.pow(1 + discountRate / 100, years[i]);

        // Push to arrays
        revenues.push(revenue);
        profits.push(netProfit);
        npv.push(discountedCashFlow);
    }

    // Calculate Net Present Value (NPV)
    let totalNPV = npv.reduce((acc, val) => acc + val, 0);

    return { years, revenues, profits, npv, totalNPV };
}

// Initialize chart
let ctx = document.getElementById('projectionChart').getContext('2d');
let projectionChart = new Chart(ctx, {
    type: 'line',
    data: {},
    options: {
        responsive: true,
        title: {
            display: true,
            text: 'Financial Projections Over 5 Years'
        },
        scales: {
            y: {
                beginAtZero: false,
                ticks: {
                    callback: function(value) {
                        return '$' + value.toLocaleString();
                    }
                }
            }
        }
    }
});

// Function to update chart
function updateChart() {
    let data = calculateProjections();

    projectionChart.data = {
        labels: data.years.map(year => `Year ${year}`),
        datasets: [
            {
                label: 'Revenue',
                data: data.revenues,
                borderColor: 'blue',
                fill: false
            },
            {
                label: 'Net Profit',
                data: data.profits,
                borderColor: 'green',
                fill: false
            },
            {
                label: 'Discounted Cash Flow',
                data: data.npv,
                borderColor: 'purple',
                fill: false
            }
        ]
    };
    projectionChart.update();

    // Display total NPV
    if (!document.getElementById('npvDisplay')) {
        let npvElement = document.createElement('h4');
        npvElement.id = 'npvDisplay';
        npvElement.className = 'mt-4';
        document.querySelector('.projection-section').appendChild(npvElement);
    }
    document.getElementById('npvDisplay').innerText = `Total NPV over 5 years: $${data.totalNPV.toLocaleString(undefined, {maximumFractionDigits: 2})}`;
}

// Initial chart update
updateChart();

// File upload handling (currently just displays the file name)
document.getElementById('fileInput').addEventListener('change', (event) => {
    let file = event.target.files[0];
    if (file) {
        document.getElementById('fileName').innerText = `Selected file: ${file.name}`;
        // Future implementation: parse the file and update initialRevenue
    } else {
        document.getElementById('fileName').innerText = '';
    }
});

// Feedback form submission
document.getElementById('feedbackForm').addEventListener('submit', (event) => {
    event.preventDefault();

    let accuracyRating = document.getElementById('accuracyRating').value;
    let feedbackComments = document.getElementById('feedbackComments').value;

    // For now, just display a thank you message
    document.getElementById('feedbackMessage').innerText = 'Thank you for your feedback!';

    // Future implementation: send feedback to the backend
    // Example:
    /*
    fetch('/submit_feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ accuracyRating, feedbackComments })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('feedbackMessage').innerText = data.message;
    })
    .catch(error => {
        console.error('Error submitting feedback:', error);
    });
    */

    // Reset form
    document.getElementById('feedbackForm').reset();
});
