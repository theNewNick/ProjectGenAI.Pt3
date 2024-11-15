// Initialize variables with default values
let revenueGrowth = 5; // in percent
let taxRate = 21; // in percent
let operatingExpenses = 200000; // in dollars
let cogsPct = 60; // Cost of Goods Sold (% of Revenue)
let discountRate = 10; // in percent
let interestRate = 5; // in percent

// Update display values
function updateDisplayValues() {
    document.getElementById('revenueGrowthValue').innerText = `${revenueGrowth}%`;
    document.getElementById('taxRateValue').innerText = `${taxRate}%`;
    document.getElementById('operatingExpensesValue').innerText = `$${operatingExpenses.toLocaleString()}`;
    document.getElementById('cogsPctValue').innerText = `${cogsPct}%`;
    document.getElementById('discountRateValue').innerText = `${discountRate}%`;
    document.getElementById('interestRateValue').innerText = `${interestRate}%`;
}

updateDisplayValues();

// Add event listeners to the sliders
document.getElementById('revenueGrowth').addEventListener('input', (event) => {
    revenueGrowth = parseFloat(event.target.value);
    document.getElementById('revenueGrowthValue').innerText = `${revenueGrowth}%`;
});

document.getElementById('taxRate').addEventListener('input', (event) => {
    taxRate = parseFloat(event.target.value);
    document.getElementById('taxRateValue').innerText = `${taxRate}%`;
});

document.getElementById('operatingExpenses').addEventListener('input', (event) => {
    operatingExpenses = parseFloat(event.target.value);
    document.getElementById('operatingExpensesValue').innerText = `$${operatingExpenses.toLocaleString()}`;
});

document.getElementById('cogsPct').addEventListener('input', (event) => {
    cogsPct = parseFloat(event.target.value);
    document.getElementById('cogsPctValue').innerText = `${cogsPct}%`;
});

document.getElementById('discountRate').addEventListener('input', (event) => {
    discountRate = parseFloat(event.target.value);
    document.getElementById('discountRateValue').innerText = `${discountRate}%`;
});

document.getElementById('interestRate').addEventListener('input', (event) => {
    interestRate = parseFloat(event.target.value);
    document.getElementById('interestRateValue').innerText = `${interestRate}%`;
});

// File upload handling
let uploadedData = null;

document.getElementById('fileInput').addEventListener('change', (event) => {
    let file = event.target.files[0];
    if (file) {
        document.getElementById('fileName').innerText = `Selected file: ${file.name}`;
        // Upload the file to the backend
        let formData = new FormData();
        formData.append('file', file);

        fetch('http://localhost:5500/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.data) {
                uploadedData = data.data;
                alert('File uploaded and processed successfully!');
            } else {
                alert('Error processing file: ' + data.error);
            }
        })
        .catch(error => {
            console.error('Error uploading file:', error);
            alert('Error uploading file.');
        });
    } else {
        document.getElementById('fileName').innerText = '';
    }
});

// Calculate projections
document.getElementById('calculateButton').addEventListener('click', () => {
    if (!uploadedData) {
        alert('Please upload financial data first.');
        return;
    }

    let sector = document.getElementById('sectorSelect').value;
    let scenario = document.getElementById('scenarioSelect').value;

    let initialAssumptions = {
        revenue_growth_rate: revenueGrowth / 100,
        tax_rate: taxRate / 100,
        operating_expenses: operatingExpenses,
        cogs_pct: cogsPct / 100,
        discount_rate: discountRate / 100,
        interest_rate: interestRate / 100,
        terminal_growth_rate: 0.02,
        shares_outstanding: 1000000
    };

    fetch('http://localhost:5000/calculate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            data: uploadedData,
            assumptions: initialAssumptions,
            sector: sector,
            scenario: scenario
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.projections) {
            // Update sliders with adjusted assumptions
            updateSlidersWithAdjustedAssumptions(data.adjusted_assumptions);
            displayProjections(data);
            updateChart(data);
        } else {
            alert('Error calculating projections.');
        }
    })
    .catch(error => {
        console.error('Error calculating projections:', error);
        alert('Error calculating projections.');
    });
});

function updateSlidersWithAdjustedAssumptions(adjustedAssumptions) {
    // Update variables
    revenueGrowth = adjustedAssumptions.revenue_growth_rate * 100;
    taxRate = adjustedAssumptions.tax_rate * 100;
    operatingExpenses = adjustedAssumptions.operating_expenses;
    cogsPct = adjustedAssumptions.cogs_pct * 100;
    discountRate = adjustedAssumptions.discount_rate * 100;
    interestRate = adjustedAssumptions.interest_rate * 100;

    // Update sliders
    document.getElementById('revenueGrowth').value = revenueGrowth;
    document.getElementById('taxRate').value = taxRate;
    document.getElementById('operatingExpenses').value = operatingExpenses;
    document.getElementById('cogsPct').value = cogsPct;
    document.getElementById('discountRate').value = discountRate;
    document.getElementById('interestRate').value = interestRate;

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
    let netProfits = df.map(item => item['Net Profit']);
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
                    label: 'Net Profit',
                    data: netProfits,
                    borderColor: 'green',
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
}

// Feedback form submission
document.getElementById('feedbackForm').addEventListener('submit', (event) => {
    event.preventDefault();

    let sector = document.getElementById('sectorSelect').value;
    let scenario = document.getElementById('scenarioSelect').value;
    let score = parseInt(document.getElementById('accuracyRating').value);
    let comments = document.getElementById('feedbackComments').value;

    fetch('http://localhost:5000/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            sector: sector,
            scenario: scenario,
            score: score,
            comments: comments
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
