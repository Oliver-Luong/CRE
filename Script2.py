import os
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
import pandas as pd

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')

# Define paths
BASE_DIR = r"D:\Legal Linguistic Comparator\Sample Documents"
TEST_DIR = r"D:\Legal Linguistic Comparator\Test Documents" 
html_template_path = r"D:\Legal Linguistic Comparator\template.html"
html_output_path = r"D:\Legal Linguistic Comparator\Results\nda_results.html"


# Comprehensive legal jargon list
LEGAL_JARGON = set([
    "confidentiality", "disclosure", "jurisdiction", "indemnity", "liability",
    "obligation", "party", "contract", "agreement", "warranties", "disputes",
    "damages", "arbitration", "governing", "termination", "effective", "breach",
    "severability", "assignability", "intellectual property", "non-solicitation", "Definition"
])

# Comprehensive Critical Clause list
CRITICAL_CLAUSES = ["Return of Confidential Information", "Confidential Information", "non-disclosure", "jurisdiction","nondisclosure"]

# Define clause importance weights
CLAUSE_IMPORTANCE = {
    "Return of Confidential Information": 0.4,
    "Confidential Information": 0.4,
    "non-disclosure": 0.3,
    "governing law": 0.2,
    "jurisdiction": 0.2,
    "nondisclosure": 0.3
}


from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api

# Load GloVe embeddings (or any pre-trained embeddings)
print("Loading GloVe embeddings...")
glove_vectors = api.load("glove-wiki-gigaword-50")  # Adjust dimensions as needed

def calculate_similarity(sentence, clause):
    # Create word vectors for the sentence and clause
    sentence_vectors = [glove_vectors[word] for word in word_tokenize(sentence.lower()) if word in glove_vectors]
    clause_vectors = [glove_vectors[word] for word in word_tokenize(clause.lower()) if word in glove_vectors]
    
    if not sentence_vectors or not clause_vectors:
        # If either vector list is empty, return 0 similarity
        return 0

    sentence_vec = np.mean(sentence_vectors, axis=0)
    clause_vec = np.mean(clause_vectors, axis=0)
    
    # Compute cosine similarity
    similarity = np.dot(sentence_vec, clause_vec) / (np.linalg.norm(sentence_vec) * np.linalg.norm(clause_vec))
    return similarity

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

def check_clause_presence(text, reference_clauses):
    """
    Check the presence of critical clauses in the text using exact matching (case-insensitive).
    """
    # Initialize clause scores
    clause_scores = {clause: 0 for clause in reference_clauses}
    
    # Process sentences in the text
    sentences = sent_tokenize(text)
    for clause in reference_clauses:
        for sentence in sentences:
            if clause.lower() in sentence.lower():
                clause_scores[clause] += 1
                break  # Move to the next clause once found in a sentence

    # Optional debug: Print clauses that are not found
    for clause, count in clause_scores.items():
        if count == 0:
            print(f"Clause not found: {clause}")
    
    return clause_scores



def check_clause_presence_with_feedback(text):
    """
    Checks for the presence of critical clauses and provides feedback.
    Returns a score and a list of missing clauses.
    """
    present_clauses = []
    missing_clauses = []
    clause_score = 0

    for clause, weight in CLAUSE_IMPORTANCE.items():
        if clause.lower() in text.lower():
            present_clauses.append(clause)
            clause_score += weight * 100  # Scale to a percentage
        else:
            missing_clauses.append(clause)

    return clause_score, present_clauses, missing_clauses

# Function to read .txt files
def load_txt_files(base_dir):
    documents = []
    for file_name in os.listdir(base_dir):
        if file_name.endswith(".txt"):
            file_path = os.path.join(base_dir, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                documents.append(text)
    return documents

# Function to preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalnum()]  # Remove punctuation
    return tokens

# Function to label NDAs as good or bad
def label_nda(text):
    tokens = preprocess_text(text)
    sentences = sent_tokenize(text)

    # Extract metrics
    clauses_present = sum(1 for clause in CRITICAL_CLAUSES if clause.lower() in text.lower())
    jargon_density = sum(1 for token in tokens if token in LEGAL_JARGON) / len(tokens) if tokens else 0
    avg_sentence_length = len(tokens) / len(sentences) if sentences else 0

    # Debugging info
    print(f"Debug: Clauses={clauses_present}, Jargon={jargon_density:.2f}, Sentence Length={avg_sentence_length:.2f}")

    # Refined logic
    if clauses_present < len(CRITICAL_CLAUSES) and jargon_density < 0.02 and avg_sentence_length > 40:
        return "bad"  # Multiple negative indicators
    elif clauses_present >= len(CRITICAL_CLAUSES) or jargon_density >= 0.03:
        return "good"  # Strong presence of clauses or jargon
    else:
        return "bad"



# Process NDAs
def label_ndas(directory):
    results = {}
    for file_name in os.listdir(directory):
        if file_name.endswith(".txt"):
            file_path = os.path.join(directory, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                label = label_nda(text)
                results[file_name] = label
    return results

# Function to analyze NDAs and create a reference model
def analyze_training_ndas(documents):
    clause_counts, jargon_density, avg_sentence_lengths = [], [], []
    
    for doc in documents:
        tokens = preprocess_text(doc)
        sentences = sent_tokenize(doc)
        
        # Clause count (approximate using keywords)
        clause_count = sum(1 for sentence in sentences if any(keyword in sentence.lower() for keyword in LEGAL_JARGON))
        clause_counts.append(clause_count)
        
        # Jargon density
        jargon_count = sum(1 for token in tokens if token in LEGAL_JARGON)
        jargon_density.append(jargon_count / len(tokens) if tokens else 0)
        
        # Average sentence length
        avg_sentence_lengths.append(len(tokens) / len(sentences) if sentences else 0)

    # Compute baseline metrics
    baseline = {
        "avg_clause_count": np.mean(clause_counts),
        "avg_jargon_density": np.mean(jargon_density),
        "avg_sentence_length": np.mean(avg_sentence_lengths),
    }
    return baseline



# Define contextual keywords
KEYWORD_CONTEXT = {
    "effective date": ["termination", "agreement"],
    "disclosure": ["confidentiality", "obligation"],
    "jurisdiction": ["governing law", "disputes"],
    "termination": ["effective date", "notice"],
}

def evaluate_keyword_relevance(text):
    """
    Evaluate if key terms appear in appropriate contextual clauses.
    """
    sentences = sent_tokenize(text.lower())
    keyword_relevance_score = 0
    max_score = len(KEYWORD_CONTEXT)

    for keyword, contexts in KEYWORD_CONTEXT.items():
        for sentence in sentences:
            if keyword in sentence and any(context in sentence for context in contexts):
                keyword_relevance_score += 1
                break

    relevance_score = (keyword_relevance_score / max_score) * 100 if max_score else 0
    return relevance_score


import re

def evaluate_formatting_consistency(text):
    """
    Evaluate formatting consistency in terms of bullet points, numbered lists, and headings.
    """
    bullets = re.findall(r"^\s*[-•]\s+", text, re.MULTILINE)
    numbered = re.findall(r"^\s*\d+\.\s+", text, re.MULTILINE)
    headings = re.findall(r"^[A-Z][A-Z\s]+$", text, re.MULTILINE)

    # Assign scores based on formatting presence
    bullet_score = 1 if len(bullets) > 0 else 0
    numbered_score = 1 if len(numbered) > 0 else 0
    heading_score = 1 if len(headings) > 0 else 0

    # Calculate overall formatting consistency score
    formatting_score = (bullet_score + numbered_score + heading_score) / 3 * 100
    return formatting_score


import textstat

def calculate_readability(text):
    """
    Calculate Flesch Reading Ease score.
    """
    return textstat.flesch_reading_ease(text)

# Function to calculate readability
def calculate_readability(text):
    """
    Calculates a readability score for a given text using a basic formula.
    Flesch Reading Ease Score:
    - Higher scores indicate easier readability.
    - Formula: 206.835 - (1.015 * ASL) - (84.6 * ASW)
    Where:
      ASL = Average Sentence Length (words/sentences)
      ASW = Average Syllables per Word (syllables/words)
    """
    import re

    # Helper function to count syllables in a word
    def count_syllables(word):
        word = word.lower()
        vowels = "aeiouy"
        count = 0
        prev_char_was_vowel = False

        for char in word:
            if char in vowels:
                if not prev_char_was_vowel:
                    count += 1
                prev_char_was_vowel = True
            else:
                prev_char_was_vowel = False

        if word.endswith("e"):
            count -= 1

        return max(count, 1)

    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    num_sentences = len(sentences)
    num_words = len(words)
    num_syllables = sum(count_syllables(word) for word in words if word.isalnum())

    if num_sentences == 0 or num_words == 0:
        return 0  # Avoid division by zero

    # Calculate ASL (Average Sentence Length) and ASW (Average Syllables per Word)
    asl = num_words / num_sentences
    asw = num_syllables / num_words

    # Calculate Flesch Reading Ease Score
    readability_score = 206.835 - (1.015 * asl) - (84.6 * asw)
    return max(readability_score, 0)  # Ensure the score is non-negative


# Function to score a test NDA
def score_test_nda_with_feedback(test_document, baseline, weights=None):
    if weights is None:
        weights = {
            "clause_weight": 0.3,
            "jargon_weight": 0.2,
            "sentence_length_weight": 0.2,
            "keyword_relevance_weight": 0.1,
            "formatting_weight": 0.1,
            "readability_weight": 0.1,
        }

    tokens = preprocess_text(test_document)
    sentences = sent_tokenize(test_document)

    # Extract features
    clause_score, present_clauses, missing_clauses = check_clause_presence_with_feedback(test_document)
    jargon_density = sum(1 for token in tokens if token in LEGAL_JARGON) / len(tokens) if tokens else 0
    avg_sentence_length = len(tokens) / len(sentences) if sentences else 0
    readability_score = calculate_readability(test_document)
    keyword_relevance_score = evaluate_keyword_relevance(test_document)
    formatting_score = evaluate_formatting_consistency(test_document)

    # Compare with baseline
    jargon_score = min(jargon_density / baseline["avg_jargon_density"], 1) * 100 if baseline["avg_jargon_density"] else 0
    sentence_length_score = min(avg_sentence_length / baseline["avg_sentence_length"], 1) * 100 if baseline["avg_sentence_length"] else 0

    # Weighted final score
    final_score = (
        clause_score * weights["clause_weight"] +
        jargon_score * weights["jargon_weight"] +
        sentence_length_score * weights["sentence_length_weight"] +
        readability_score * weights["readability_weight"] +
        keyword_relevance_score * weights["keyword_relevance_weight"] +
        formatting_score * weights["formatting_weight"]
    )

    # Generate feedback
    feedback = []
    if missing_clauses:
        feedback.append(f"Consider adding missing clauses: {', '.join(missing_clauses)}.")
    if jargon_density < 0.02:
        feedback.append("Increase legal jargon usage for precision and professionalism.")
    if avg_sentence_length > 30:
        feedback.append("Shorten sentences to improve readability.")
    if keyword_relevance_score < 80:
        feedback.append("Improve the placement of key terms in appropriate clauses.")
    if formatting_score < 70:
        feedback.append("Ensure consistent formatting, including proper headings and bullet points.")

    return {
        "clause_score": clause_score,
        "jargon_score": jargon_score,
        "sentence_length_score": sentence_length_score,
        "readability_score": readability_score,
        "keyword_relevance_score": keyword_relevance_score,
        "formatting_score": formatting_score,
        "final_score": final_score,
        "feedback": feedback
    }








# Function to save results to HTML
from jinja2 import Environment, FileSystemLoader

def save_results_to_html(results, template_directory, template_file_name, output_path, charts=None):
    """
    Save results to an HTML report, optionally including charts.
    """
    # Load the HTML template
    env = Environment(loader=FileSystemLoader(template_directory))
    template = env.get_template(template_file_name)

    # Add the charts (if any) to the results
    template_data = {
        "results": results,
        "charts": charts or {}
    }

    # Render the HTML content with charts
    html_content = template.render(template_data)

    # Write the HTML content to the output file
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(html_content)

    print(f"HTML report saved to {output_path}")

# Write template file if it doesn't exist
if not os.path.exists(html_template_path):    
    with open(html_template_path, 'w', encoding='utf-8') as template_file:
            template_file.write(html_template_content)
    print(f"Template created at: {html_template_path}")


# Function to create bar charts for the report
import matplotlib.pyplot as plt

def generate_visuals_by_document(results, output_dir):
    """
    Generate bar charts for each document, displaying all metric scores, and save them as images.
    """
    import matplotlib.pyplot as plt
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    charts = {}
    for result in results:
        metrics = [
            "Clause Score", "Jargon Score", "Sentence Length Score", 
            "Readability Score", "Keyword Relevance Score", "Formatting Score", "Final Score"
        ]

        # Extract metric values
        values = [
            result["Clause Score"], result["Jargon Score"], result["Sentence Length Score"],
            result["Readability Score"], result["Keyword Relevance Score"],
            result["Formatting Score"], result["Final Score"]
        ]

        # Create bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(metrics, values, color="skyblue")
        plt.title(f"Scores for {result['Document Name']}")
        plt.xlabel("Metrics")
        plt.ylabel("Scores")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Save chart
        chart_filename = f"{result['Document Name'].lower().replace(' ', '_')}_chart.png"
        chart_path = os.path.join(output_dir, chart_filename)
        plt.savefig(chart_path)
        plt.close()

        charts[result["Document Name"]] = chart_path
    return charts




# Function to save results to CSV
def save_results_to_csv(results, output_path):
    """
    Save test results to a CSV file.
    """
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")




# Main Execution
if __name__ == "__main__":
    # Label training documents
    print(f"Labeling training documents in: {BASE_DIR}")
    training_labels = label_ndas(BASE_DIR)

    # Ensure there are good NDAs for training
    good_ndas = [doc for doc, label in training_labels.items() if label == "good"]
    if not good_ndas:
        raise ValueError("No good NDAs available for training. Please check your dataset.")

    # Analyze training NDAs (good NDAs only)
    print(f"\nAnalyzing {len(good_ndas)} good NDAs to create baseline...")
    training_documents = [open(os.path.join(BASE_DIR, doc), 'r', encoding='utf-8').read() for doc in good_ndas]
    baseline_metrics = analyze_training_ndas(training_documents)

    # Test the model
    if os.path.exists(TEST_DIR):
        print(f"\nTesting the model using .txt files from: {TEST_DIR}")
        
        # Properly load test documents
        test_documents = load_txt_files(TEST_DIR)  # Load test documents here

        # Collect results
        csv_results = []
        for test_doc_name, test_doc_content in zip(os.listdir(TEST_DIR), test_documents):
            scores = score_test_nda_with_feedback(test_doc_content, baseline_metrics)
            csv_results.append({
                "Document Name": test_doc_name,
                "Clause Score": scores['clause_score'],
                "Jargon Score": scores['jargon_score'],
                "Sentence Length Score": scores['sentence_length_score'],
                "Readability Score": scores['readability_score'],
                "Keyword Relevance Score": scores['keyword_relevance_score'],
                "Formatting Score": scores['formatting_score'],
                "Final Score": scores['final_score'],
                "Feedback": "; ".join(scores['feedback'])
            })

        # Save results to CSV
        csv_output_path = r"D:\Legal Linguistic Comparator\Results\nda_results.csv"
        save_results_to_csv(csv_results, csv_output_path)

   
   
    # Save results to CSV
    csv_output_path = r"D:\Legal Linguistic Comparator\Results\nda_results.csv"
    save_results_to_csv(csv_results, csv_output_path)

    # Generate visuals
    df = pd.DataFrame(csv_results)
    charts_dir = r"D:\Legal Linguistic Comparator\Results\charts"
    charts = generate_visuals_by_document(csv_results, charts_dir)
    generate_visuals_by_document(csv_results, charts_dir)


    # Save results to HTML
    html_template_directory = r"D:\Legal Linguistic Comparator"
    template_file_name = "template.html"
    html_output_path = r"D:\Legal Linguistic Comparator\Results\nda_results.html"
    
    # Save results to HTML
    save_results_to_html(csv_results, html_template_directory, template_file_name, html_output_path, charts)





def save_results_to_csv(results, output_path):
    """
    Save test results to a CSV file.
    """
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

# Collect results for CSV
csv_results = []
for test_doc_name, test_doc_content in zip(os.listdir(TEST_DIR), test_documents):
    scores = score_test_nda_with_feedback(test_doc_content, baseline_metrics)
    csv_results.append({
        "Document Name": test_doc_name,
        "Clause Score": scores['clause_score'],
        "Jargon Score": scores['jargon_score'],
        "Sentence Length Score": scores['sentence_length_score'],
        "Readability Score": scores['readability_score'],
        "Keyword Relevance Score": scores['keyword_relevance_score'],
        "Formatting Score": scores['formatting_score'],
        "Final Score": scores['final_score'],
        "Feedback": "; ".join(scores['feedback'])
    })




from jinja2 import Environment, FileSystemLoader

def save_results_to_html(results, template_path, output_path, charts=None):
    """
    Save results to an HTML report, optionally including charts.
    """
    # Load the HTML template
    env = Environment(loader=FileSystemLoader(os.path.dirname(html_template_path)))
    template = env.get_template(os.path.basename(html_template_path))
    
    # Add the charts (if any) to the results
    template_data = {
        "results": results,
        "charts": charts or {}
    }
    
    # Render the HTML content with charts
    html_content = template.render(template_data)
    
    # Write the HTML content to the output file
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(html_content)
    
    print(f"HTML report saved to {output_path}")



from fpdf import FPDF

def save_results_to_pdf_with_charts(results, output_path, charts_dir):
    """
    Save test results to a PDF file, with each document displayed on a separate page
    and charts reflecting scores for all metrics included in the report.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    for result in results:
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Title
        pdf.set_font("Arial", style="B", size=16)
        pdf.cell(200, 10, txt="NDA Quality Report", ln=True, align="C")
        pdf.ln(10)

        # Document Name
        pdf.set_font("Arial", style="B", size=12)
        pdf.cell(200, 10, txt=f"Document Name: {result['Document Name']}", ln=True, align="L")
        pdf.ln(10)

        # Scores & Feedback
        pdf.set_font("Arial", size=10)
        scores_feedback = (
            f"Clause Score: {result['Clause Score']:.2f}\n"
            f"Jargon Score: {result['Jargon Score']:.2f}\n"
            f"Sentence Length Score: {result['Sentence Length Score']:.2f}\n"
            f"Readability Score: {result['Readability Score']:.2f}\n"
            f"Keyword Relevance Score: {result['Keyword Relevance Score']:.2f}\n"
            f"Formatting Score: {result['Formatting Score']:.2f}\n"
        )

        # Add scores to the PDF
        pdf.multi_cell(0, 10, txt=scores_feedback, border=0, align="L")

        # Bold the Final Score
        pdf.set_font("Arial", style="B", size=10)
        pdf.cell(0, 10, txt=f"Final Score: {result['Final Score']:.2f}", ln=True, align="L")

        # Reset font to normal for feedback
        pdf.set_font("Arial", size=10)
        feedback_text = f"Feedback: {result['Feedback']}"
        pdf.multi_cell(0, 10, txt=feedback_text, border=0, align="L")
        pdf.ln()

        # Add chart
        chart_path = os.path.join(charts_dir, f"{result['Document Name'].lower().replace(' ', '_')}_chart.png")
        if os.path.exists(chart_path):
            pdf.image(chart_path, x=10, y=pdf.get_y(), w=180)  # Adjust placement and size as needed
            pdf.ln(100)

    # Save the PDF
    pdf.output(output_path)
    print(f"PDF report with charts saved to {output_path}")





# Save to CSV
csv_output_path = r"D:\Legal Linguistic Comparator\Results\nda_results.csv"
save_results_to_csv(csv_results, csv_output_path)


# Save to HTML
html_template_directory = r"D:\Legal Linguistic Comparator"
template_file_name = "template.html"
html_output_path = r"D:\Legal Linguistic Comparator\Results\nda_results.html"
save_results_to_html(csv_results, html_template_path, html_output_path, charts)


# Save to PDF
pdf_output_path = r"D:\Legal Linguistic Comparator\Results\nda_report_with_charts.pdf"
charts_dir = r"D:\Legal Linguistic Comparator\Results\charts"
save_results_to_pdf_with_charts(csv_results, pdf_output_path, charts_dir)