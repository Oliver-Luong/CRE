document.addEventListener('DOMContentLoaded', function() {
    const contractText = document.getElementById('contractText');
    const clearBtn = document.getElementById('clearBtn');
    const pasteBtn = document.getElementById('pasteBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resultsSection = document.getElementById('resultsSection');
    const overallScore = document.getElementById('overallScore');
    const analysisCategories = document.getElementById('analysisCategories');
    const recommendationsList = document.getElementById('recommendationsList');
    const exportBtn = document.getElementById('exportResults');
    const errorAlert = document.createElement('div');
    errorAlert.className = 'alert alert-danger alert-dismissible fade show';
    errorAlert.style.display = 'none';
    resultsSection.parentNode.insertBefore(errorAlert, resultsSection);

    // Clear button handler
    clearBtn.addEventListener('click', () => {
        contractText.value = '';
        errorAlert.style.display = 'none';
    });

    // Paste button handler
    pasteBtn.addEventListener('click', async () => {
        try {
            const text = await navigator.clipboard.readText();
            contractText.value = text;
            errorAlert.style.display = 'none';
        } catch (err) {
            console.error('Failed to read clipboard:', err);
            showError('Unable to paste from clipboard. Please paste manually.');
        }
    });

    function showError(message) {
        errorAlert.innerHTML = `
            <strong>Error:</strong> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        errorAlert.style.display = 'block';
        resultsSection.style.display = 'none';
    }

    // Analyze button handler
    analyzeBtn.addEventListener('click', async () => {
        const text = contractText.value.trim();
        
        if (!text) {
            showError('Please enter or paste contract text for analysis.');
            return;
        }

        // Show loading state
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Analyzing...';
        errorAlert.style.display = 'none';
        resultsSection.style.display = 'none';

        try {
            console.log('Sending analysis request...');
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text })
            });

            console.log('Received response:', response.status);
            const data = await response.json();
            console.log('Response data:', data);

            if (!response.ok) {
                const errorMessage = data.error || data.details || 'Failed to analyze contract';
                console.error('Server error:', errorMessage);
                throw new Error(errorMessage);
            }

            if (data.error) {
                console.error('Analysis error:', data.error);
                throw new Error(data.error);
            }

            // Check if we have valid analysis data
            if (!data.score || !data.analysis || !data.recommendations) {
                console.error('Invalid analysis results:', data);
                throw new Error('Invalid analysis results received from server');
            }

            console.log('Analysis successful, displaying results...');
            displayResults(data);
            resultsSection.style.display = 'block';
            exportBtn.style.display = 'block';

        } catch (error) {
            console.error('Analysis error:', error);
            showError(error.message);
        } finally {
            // Reset button state
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = 'Analyze Contract';
        }
    });

    function displayResults(data) {
        // Display overall score
        overallScore.textContent = `${data.score}`;
        overallScore.className = `score-number ${getScoreClass(data.score)}`;

        // Clear previous results
        analysisCategories.innerHTML = '';
        recommendationsList.innerHTML = '';

        // Create analysis categories HTML
        const categories = {
            'Similarity': data.analysis.similarity,
            'Legal Jargon': data.analysis.jargon_density,
            'Structure': data.analysis.structure,
            'Readability': data.analysis.readability
        };

        // Display category scores
        for (const [category, score] of Object.entries(categories)) {
            if (score !== undefined) {
                const categoryDiv = document.createElement('div');
                categoryDiv.className = 'col-md-6 mb-4';
                categoryDiv.innerHTML = `
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">${category}</h5>
                            <div class="d-flex align-items-center">
                                <div class="score-badge ${getScoreClass(score)}">${score}%</div>
                            </div>
                        </div>
                    </div>
                `;
                analysisCategories.appendChild(categoryDiv);
            }
        }

        // Display key phrases if available
        if (data.analysis.key_phrases && data.analysis.key_phrases.length > 0) {
            const phrasesDiv = document.createElement('div');
            phrasesDiv.className = 'col-12 mb-4';
            phrasesDiv.innerHTML = `
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">Key Phrases</h5>
                        <p class="card-text">${data.analysis.key_phrases.join(', ')}</p>
                    </div>
                </div>
            `;
            analysisCategories.appendChild(phrasesDiv);
        }

        // Display recommendations
        data.recommendations.forEach(rec => {
            const li = document.createElement('li');
            li.className = 'list-group-item';
            li.textContent = rec;
            recommendationsList.appendChild(li);
        });
    }

    function getScoreClass(score) {
        if (score >= 80) return 'text-success';
        if (score >= 60) return 'text-warning';
        return 'text-danger';
    }

    // Export functionality
    exportBtn.addEventListener('click', () => {
        const text = generateExportText();
        const blob = new Blob([text], { type: 'text/plain' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'contract_analysis.txt';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    });

    function generateExportText() {
        let text = 'Contract Analysis Report\n';
        text += '=======================\n\n';
        text += `Overall Score: ${overallScore.textContent}\n\n`;
        
        text += 'Category Analysis\n';
        text += '----------------\n';
        for (const categoryDiv of analysisCategories.children) {
            const title = categoryDiv.querySelector('h5').textContent;
            const score = categoryDiv.querySelector('.score-badge').textContent;
            
            text += `\n${title}\n`;
            text += `Score: ${score}\n`;
        }
        
        text += '\nRecommendations\n';
        text += '---------------\n';
        for (const rec of recommendationsList.children) {
            text += `- ${rec.textContent}\n`;
        }
        
        return text;
    }
});
