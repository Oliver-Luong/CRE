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

    // Clear button handler
    clearBtn.addEventListener('click', () => {
        contractText.value = '';
    });

    // Paste button handler
    pasteBtn.addEventListener('click', async () => {
        try {
            const text = await navigator.clipboard.readText();
            contractText.value = text;
        } catch (err) {
            console.error('Failed to read clipboard:', err);
            alert('Unable to paste from clipboard. Please paste manually.');
        }
    });

    // Analyze button handler
    analyzeBtn.addEventListener('click', async () => {
        const text = contractText.value.trim();
        
        if (!text) {
            alert('Please enter or paste contract text for analysis.');
            return;
        }

        // Show loading state
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Analyzing...';

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ original_text: text })
            });

            const data = await response.json();

            if (response.ok) {
                displayResults(data);
            } else {
                throw new Error(data.error || 'Analysis failed');
            }
        } catch (err) {
            console.error('Analysis error:', err);
            alert('Failed to analyze contract. Please try again.');
        } finally {
            // Reset button state
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '<i class="bi bi-search"></i> Analyze Contract';
        }
    });

    // Export button handler
    exportBtn.addEventListener('click', () => {
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const filename = `contract-analysis-${timestamp}.txt`;
        
        // Gather all analysis data
        const analysisText = generateExportText();
        
        // Create and trigger download
        const blob = new Blob([analysisText], { type: 'text/plain' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = filename;
        
        document.body.appendChild(a);
        a.click();
        
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    });

    // Helper function to display results
    function displayResults(data) {
        // Display overall score
        overallScore.textContent = Math.round(data.score);
        
        // Display category scores
        analysisCategories.innerHTML = '';
        Object.entries(data.analysis).forEach(([category, details]) => {
            const scoreClass = getScoreClass(details.score);
            const categoryHtml = `
                <div class="col-md-6">
                    <div class="analysis-category">
                        <div class="category-header">
                            <span class="category-name">${category}</span>
                            <span class="category-score">${Math.round(details.score)}%</span>
                        </div>
                        <div class="score-bar">
                            <div class="score-fill ${scoreClass}" style="width: ${details.score}%"></div>
                        </div>
                        <div class="category-details">
                            <small>${details.details}</small>
                        </div>
                    </div>
                </div>
            `;
            analysisCategories.innerHTML += categoryHtml;
        });

        // Display recommendations
        recommendationsList.innerHTML = '';
        data.recommendations.forEach(rec => {
            recommendationsList.innerHTML += `<li>${rec}</li>`;
        });

        // Show results section
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    // Helper function to determine score class
    function getScoreClass(score) {
        if (score >= 80) return 'score-high';
        if (score >= 60) return 'score-medium';
        return 'score-low';
    }

    // Helper function to generate export text
    function generateExportText() {
        const sections = [];
        
        // Overall Score
        sections.push(`LEGAL CONTRACT ANALYSIS REPORT\n${'='.repeat(30)}\n`);
        sections.push(`Overall Score: ${overallScore.textContent}%\n`);
        
        // Category Scores
        sections.push('\nDETAILED ANALYSIS\n' + '-'.repeat(20));
        document.querySelectorAll('.analysis-category').forEach(category => {
            const name = category.querySelector('.category-name').textContent;
            const score = category.querySelector('.category-score').textContent;
            const details = category.querySelector('.category-details small').textContent;
            sections.push(`\n${name}\nScore: ${score}\n${details}`);
        });
        
        // Recommendations
        sections.push('\nRECOMMENDATIONS\n' + '-'.repeat(20));
        document.querySelectorAll('.recommendations-list li').forEach(rec => {
            sections.push(`• ${rec.textContent}`);
        });
        
        // Timestamp
        sections.push(`\n\nReport generated on: ${new Date().toLocaleString()}`);
        
        return sections.join('\n');
    }
});
