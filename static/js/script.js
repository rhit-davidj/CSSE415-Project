document.addEventListener('DOMContentLoaded', function () {
    const tastingForm = document.getElementById('tastingForm');
    const resultsArea = document.getElementById('resultsArea');
    const predictedRatingEl = document.getElementById('predictedRating');
    const similarCoffeesListEl = document.getElementById('similarCoffeesList');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const errorMessageEl = document.getElementById('errorMessage');

    tastingForm.addEventListener('submit', async function (e) {
        e.preventDefault();
        loadingIndicator.style.display = 'block';
        resultsArea.style.display = 'none';
        errorMessageEl.style.display = 'none';


        const notes = document.getElementById('tastingNotes').value;

        if (!notes.trim()) {
            displayError("Please enter some tasting notes.");
            loadingIndicator.style.display = 'none';
            return;
        }

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ notes: notes }),
            });

            loadingIndicator.style.display = 'none';

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({error: "An unknown error occurred."})); // Catch if JSON parsing fails
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            if (data.error) {
                 displayError(data.error);
            } else {
                predictedRatingEl.textContent = `${data.predicted_rating} / 100`;

                similarCoffeesListEl.innerHTML = ''; // Clear previous results
                if (data.similar_coffees && data.similar_coffees.length > 0) {
                    data.similar_coffees.forEach(coffee => {
                        const coffeeDiv = document.createElement('div');
                        coffeeDiv.className = 'col-md-6 mb-3'; // Bootstrap column
                        coffeeDiv.innerHTML = `
                            <div class="similar-coffee-card h-100">
                                <h6>${escapeHTML(coffee.name)}</h6>
                                <p><small><em>Notes: ${escapeHTML(coffee.notes.substring(0,100))}...</em></small></p>
                                <p class="similarity-score">Similarity: ${(coffee.similarity_score * 100).toFixed(1)}%</p>
                            </div>
                        `;
                        similarCoffeesListEl.appendChild(coffeeDiv);
                    });
                } else {
                    similarCoffeesListEl.innerHTML = '<p class="text-muted">No highly similar coffees found in our dataset.</p>';
                }
                resultsArea.style.display = 'block';
            }

        } catch (error) {
            console.error('Error:', error);
            displayError(error.message || 'Failed to get results. Please try again.');
            loadingIndicator.style.display = 'none';
        }
    });

    function displayError(message) {
        errorMessageEl.textContent = message;
        errorMessageEl.style.display = 'block';
        resultsArea.style.display = 'none'; // Hide results area on error
    }

    function escapeHTML(str) {
        var p = document.createElement("p");
        p.appendChild(document.createTextNode(str));
        return p.innerHTML;
    }
});