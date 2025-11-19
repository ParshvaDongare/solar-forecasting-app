// Solar Power Forecasting App - JavaScript

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeDatePicker();
    setupSliderListeners();
    setupFormSubmission();
    loadDataAnalysis();
    checkDeploymentStatus();
});

// Check deployment status and show banner if needed
function checkDeploymentStatus() {
    // Banner will be shown when we get demo_mode response from data-analysis
    // This is done in the loadDataAnalysis function
}

// Initialize date picker with today's date
function initializeDatePicker() {
    const dateInput = document.getElementById('date');
    const today = new Date().toISOString().split('T')[0];
    dateInput.value = today;
    dateInput.max = today;
}

// Update slider display values
function setupSliderListeners() {
    const sliders = [
        { id: 'irradiation', displayId: 'irradiation-value', suffix: '' },
        { id: 'ambient_temp', displayId: 'ambient_temp-value', suffix: '' },
        { id: 'module_temp', displayId: 'module_temp-value', suffix: '' },
        { id: 'hour', displayId: 'hour-value', suffix: '', isTime: true }
    ];

    sliders.forEach(slider => {
        const element = document.getElementById(slider.id);
        const displayElement = document.getElementById(slider.displayId);
        
        element.addEventListener('input', function() {
            if (slider.isTime) {
                displayElement.textContent = String(this.value).padStart(2, '0');
            } else {
                displayElement.textContent = this.value;
            }
        });
    });
}

// Setup form submission
function setupFormSubmission() {
    const form = document.getElementById('predictionForm');
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        await makePrediction();
    });
}

// Make prediction
async function makePrediction() {
    const irradiation = parseFloat(document.getElementById('irradiation').value);
    const ambient_temp = parseFloat(document.getElementById('ambient_temp').value);
    const module_temp = parseFloat(document.getElementById('module_temp').value);
    const hour = parseInt(document.getElementById('hour').value);
    const date = new Date(document.getElementById('date').value);
    const day = date.getDate();

    // Show loading spinner
    showLoading(true);

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                irradiation: irradiation,
                ambient_temp: ambient_temp,
                module_temp: module_temp,
                hour: hour,
                day: day
            })
        });

        const data = await response.json();

        if (data.success) {
            displayPredictionResults(data);
        } else {
            showError('Prediction failed: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        showError('Error: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// Display prediction results
function displayPredictionResults(data) {
    // Hide no prediction message and show results
    document.getElementById('noPrediction').style.display = 'none';
    document.getElementById('predictionResults').style.display = 'block';

    // Update metrics
    document.getElementById('predictionValue').textContent = data.prediction.toFixed(2);
    document.getElementById('efficiencyValue').textContent = data.efficiency.toFixed(2);
    document.getElementById('dailyEstimate').textContent = data.daily_estimate.toFixed(2);

    // Show/hide demo mode card
    if (data.demo_mode) {
        document.getElementById('demoModeCard').style.display = 'block';
    } else {
        document.getElementById('demoModeCard').style.display = 'none';
    }

    // Draw gauge chart
    drawGaugeChart(data.prediction);

    // Show success message
    showSuccess('Prediction completed successfully!');
}

// Draw gauge chart using Plotly
function drawGaugeChart(value) {
    const data = [{
        type: "indicator",
        mode: "gauge+number+delta",
        value: value,
        title: { text: "AC Power (kW)" },
        domain: { x: [0, 1], y: [0, 1] },
        gauge: {
            axis: { range: [null, 1000] },
            bar: { color: "darkgreen" },
            steps: [
                { range: [0, 300], color: "rgba(200, 200, 200, 0.3)" },
                { range: [300, 600], color: "rgba(150, 150, 150, 0.3)" },
                { range: [600, 1000], color: "rgba(100, 100, 100, 0.3)" }
            ],
            threshold: {
                line: { color: "red", width: 4 },
                thickness: 0.75,
                value: 800
            }
        }
    }];

    const layout = {
        margin: { l: 25, r: 25, t: 25, b: 25 },
        paper_bgcolor: "white",
        font: { color: "black" }
    };

    Plotly.newPlot('gaugeChart', data, layout, { responsive: true });
}

// Load data analysis
async function loadDataAnalysis() {
    try {
        const response = await fetch('/api/data-analysis');
        const data = await response.json();

        if (data.success) {
            displayDataAnalysis(data);
        } else {
            // Show deployment banner if in demo mode
            if (data.demo_mode) {
                document.getElementById('deploymentBanner').style.display = 'block';
            }
            document.getElementById('statsContainer').style.display = 'none';
            document.getElementById('noData').style.display = 'block';
        }
    } catch (error) {
        console.error('Error loading data analysis:', error);
        document.getElementById('statsContainer').style.display = 'none';
        document.getElementById('noData').style.display = 'block';
    }
}

// Display data analysis
function displayDataAnalysis(data) {
    const stats = data.stats;
    
    document.getElementById('totalRecords').textContent = stats.total_records.toLocaleString();
    document.getElementById('dateRange').textContent = stats.date_range_days;
    document.getElementById('avgPower').textContent = stats.avg_ac_power.toFixed(2);
    document.getElementById('maxPower').textContent = stats.max_ac_power.toFixed(2);

    // Draw correlation heatmap
    drawCorrelationChart(data.correlations);
}

// Draw correlation heatmap
function drawCorrelationChart(correlations) {
    // Convert correlation data to format Plotly expects
    const features = Object.keys(correlations);
    const z = [];
    
    features.forEach(feature => {
        const row = [];
        features.forEach(f => {
            row.push(correlations[feature][f]);
        });
        z.push(row);
    });

    const data = [{
        z: z,
        x: features,
        y: features,
        type: 'heatmap',
        colorscale: 'RdYlGn',
        zmid: 0,
        zmin: -1,
        zmax: 1,
        text: z.map(row => row.map(val => val.toFixed(2))),
        texttemplate: '%{text}',
        textfont: { size: 12 },
        hoverongaps: false
    }];

    const layout = {
        title: 'Feature Correlation Matrix',
        xaxis: { side: 'bottom' },
        yaxis: { autorange: 'reversed' },
        margin: { l: 100, r: 50, b: 100, t: 50 }
    };

    Plotly.newPlot('correlationChart', data, layout, { responsive: true });
}

// Utility functions
function showLoading(show) {
    const spinner = document.getElementById('loadingSpinner');
    if (show) {
        spinner.style.display = 'flex';
    } else {
        spinner.style.display = 'none';
    }
}

function showError(message) {
    showAlert(message, 'danger');
}

function showSuccess(message) {
    showAlert(message, 'success');
}

function showAlert(message, type = 'info') {
    // Create alert element
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show`;
    alert.setAttribute('role', 'alert');
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;

    // Insert at top of page
    const container = document.querySelector('.container-fluid');
    container.insertBefore(alert, container.firstChild);

    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        alert.remove();
    }, 5000);
}

// Handle tab changes
document.getElementById('analysis-tab').addEventListener('shown.bs.tab', function() {
    // Trigger resize on Plotly charts when tab becomes visible
    setTimeout(() => {
        Plotly.Plots.resize('correlationChart');
    }, 100);
});

document.getElementById('prediction-tab').addEventListener('shown.bs.tab', function() {
    setTimeout(() => {
        if (document.getElementById('predictionResults').style.display !== 'none') {
            Plotly.Plots.resize('gaugeChart');
        }
    }, 100);
});
