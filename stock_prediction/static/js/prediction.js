document.addEventListener('DOMContentLoaded', function() {
    const stockForm = document.getElementById('stock-form');
    const predictionResults = document.getElementById('prediction-results');
    const errorContainer = document.getElementById('error-container');

    // Get CSRF token from cookies
    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }

    // Get CSRF token
    const csrfToken = getCookie('csrftoken');

    stockForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Clear previous results and errors
        predictionResults.innerHTML = '';
        errorContainer.innerHTML = '';
        
        // Show loading spinner
        const loadingSpinner = document.createElement('div');
        loadingSpinner.innerHTML = `
            <div class="text-center py-3">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mt-2">Generating predictions...</p>
            </div>
        `;
        predictionResults.appendChild(loadingSpinner);

        // Send AJAX request
        fetch('/predict/', {
            method: 'POST',
            headers: {
                'X-CSRFToken': csrfToken,
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: new URLSearchParams(new FormData(stockForm))
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Remove loading spinner
            loadingSpinner.remove();

            if (data.success) {
                // Prepare prediction results HTML
                const resultsHTML = `
                    <table class="table table-hover">
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th>Predicted Price</th>
                                <th>Confidence</th>
                                <th>Change %</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>LSTM Model</td>
                                <td>£${data.predictions.lstm.toFixed(2)}</td>
                                <td>${(data.predictions.confidence * 100).toFixed(2)}%</td>
                                <td>${data.predictions.lstm_change_percentage.toFixed(2)}%</td>
                            </tr>
                            <tr>
                                <td>Random Forest Model</td>
                                <td>£${data.predictions.rf.toFixed(2)}</td>
                                <td>${(data.predictions.confidence * 100).toFixed(2)}%</td>
                                <td>${data.predictions.rf_change_percentage.toFixed(2)}%</td>
                            </tr>
                        </tbody>
                    </table>
                    <div class="alert alert-info mt-3">
                        <strong>Company:</strong> ${data.company} 
                        <strong>Sector:</strong> ${data.sector}
                    </div>
                `;
                predictionResults.innerHTML = resultsHTML;

                // Update prediction chart
                updatePredictionChart(
                    parseFloat(document.getElementById('current-price').value), 
                    data.predictions.lstm, 
                    data.predictions.rf
                );
            } else {
                // Display error message
                errorContainer.innerHTML = `
                    <div class="alert alert-danger" role="alert">
                        ${data.error || 'An unexpected error occurred'}
                    </div>
                `;
            }
        })
        .catch(error => {
            // Remove loading spinner
            loadingSpinner.remove();

            // Display network or unexpected error
            errorContainer.innerHTML = `
                <div class="alert alert-danger" role="alert">
                    Network error or unexpected issue. Please try again.
                </div>
            `;
            console.error('Error:', error);
        });
    });

    // Function to update prediction chart
    function updatePredictionChart(currentPrice, lstmPrediction, rfPrediction) {
        const predictionChart = Chart.getChart('predictionChart');
        if (predictionChart) {
            predictionChart.data.datasets = [
                {
                    label: 'Current Price',
                    data: [currentPrice],
                    borderColor: 'rgb(54, 162, 235)',
                    backgroundColor: 'rgb(54, 162, 235)',
                    type: 'scatter'
                },
                {
                    label: 'LSTM Prediction',
                    data: [lstmPrediction],
                    borderColor: 'rgb(75, 192, 192)',
                    backgroundColor: 'rgb(75, 192, 192)',
                    type: 'scatter'
                },
                {
                    label: 'Random Forest Prediction',
                    data: [rfPrediction],
                    borderColor: 'rgb(255, 99, 132)',
                    backgroundColor: 'rgb(255, 99, 132)',
                    type: 'scatter'
                }
            ];
            predictionChart.update();
        }
    }
});