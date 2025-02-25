import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import string
import numpy as np

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load SpaCy language model
nlp = spacy.load('en_core_web_sm')

# Legal jargon dictionary
legal_jargon = {
    # Common legal terms
    "plaintiff", "defendant", "affidavit", "jurisdiction", "contract",
    "whereas", "hereinafter", "pursuant", "notwithstanding", "forthwith",
    "indemnify", "liability", "termination", "breach", "covenant",
    "warranty", "arbitration", "consideration", "force majeure", "governing law",
    
    # Contract-specific terms
    "agreement", "parties", "obligations", "terms", "conditions",
    "confidentiality", "intellectual property", "compensation", "termination",
    "severability", "assignment", "modification", "waiver", "notices",
    "representations"
}

def preprocess_text(text):
    """
    Tokenizes and preprocesses input text by:
    - Lowercasing
    - Removing stopwords and punctuation
    """
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return " ".join(tokens)

def extract_phrases_with_spacy(text):
    """
    Uses SpaCy to extract noun chunks as key phrases.
    """
    doc = nlp(text)
    phrases = [chunk.text for chunk in doc.noun_chunks]
    return phrases

def compute_similarity(repository, input_document):
    """
    Computes cosine similarity between the input document and a repository of legal documents.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(repository)
    input_vector = vectorizer.transform([input_document])
    similarities = cosine_similarity(input_vector, tfidf_matrix)
    return np.mean(similarities) * 100

def calculate_jargon_density(text, legal_jargon):
    """
    Calculates the density of legal jargon in the text.
    """
    tokens = word_tokenize(text)
    jargon_count = sum(1 for token in tokens if token in legal_jargon)
    return (jargon_count / len(tokens)) * 100 if tokens else 0

def analyze_contract_structure(text):
    """
    Analyzes the structural elements of a contract.
    Returns a score based on presence of key contract sections.
    """
    doc = nlp(text.lower())
    
    # Key sections to look for
    sections = {
        'parties': ['parties', 'between', 'agreement between'],
        'recitals': ['whereas', 'background', 'recitals'],
        'definitions': ['definitions', 'terms defined', 'meaning of'],
        'obligations': ['shall', 'must', 'agrees to', 'obligations'],
        'term': ['term', 'duration', 'period of agreement'],
        'termination': ['termination', 'terminate', 'end of agreement'],
        'governing_law': ['governing law', 'jurisdiction', 'applicable law'],
        'signatures': ['in witness whereof', 'signed', 'executed', 'signature']
    }
    
    found_sections = {section: [] for section in sections}
    section_weights = {
        'parties': 0.15,
        'recitals': 0.10,
        'definitions': 0.15,
        'obligations': 0.20,
        'term': 0.10,
        'termination': 0.10,
        'governing_law': 0.10,
        'signatures': 0.10
    }
    
    # Search for sections
    for sent in doc.sents:
        sent_text = sent.text.lower()
        for section, phrases in sections.items():
            if not found_sections[section]:  # Only look if we haven't found this section
                for phrase in phrases:
                    if phrase in sent_text:
                        found_sections[section].append(sent_text)
                        break
    
    # Calculate weighted score
    score = sum(section_weights[section] * 100 for section, found in found_sections.items() if found)
    
    return {
        'score': score,
        'found_sections': found_sections,
        'missing_sections': [section for section, found in found_sections.items() if not found]
    }

def analyze_readability(text):
    """
    Analyzes the readability of the contract using multiple metrics.
    """
    # Tokenize text
    sentences = [sent.text for sent in nlp(text).sents]
    words = word_tokenize(text.lower())
    
    # Basic metrics
    num_sentences = len(sentences)
    num_words = len(words)
    avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0
    
    # Calculate word complexity (words > 6 characters)
    complex_words = len([word for word in words if len(word) > 6])
    word_complexity = (complex_words / num_words) if num_words > 0 else 0
    
    # Legal jargon density
    jargon_density = calculate_jargon_density(text, legal_jargon)
    
    # Scoring components
    sentence_length_score = max(0, 100 - (avg_sentence_length - 20) * 2)  # Optimal ~20 words
    complexity_score = max(0, 100 - (word_complexity * 200))  # Penalize high complexity
    jargon_score = max(0, 100 - jargon_density)  # Lower jargon is better
    
    # Weighted final score
    final_score = (
        sentence_length_score * 0.4 +
        complexity_score * 0.3 +
        jargon_score * 0.3
    )
    
    return {
        'score': min(100, max(0, final_score)),  # Ensure score is between 0-100
        'metrics': {
            'avg_sentence_length': round(avg_sentence_length, 1),
            'word_complexity': round(word_complexity * 100, 1),
            'jargon_density': round(jargon_density, 1)
        }
    }

def analyze_assent_and_mutuality(text):
    """
    Analyzes mutual assent and intent clarity in the contract.
    Evaluates offer, acceptance, and meeting of the minds.
    """
    doc = nlp(text.lower())
    
    # Key indicators for mutual assent components
    indicators = {
        'offer': {
            'terms': ['offer', 'propose', 'agrees to provide', 'shall provide', 'will deliver'],
            'weight': 0.3
        },
        'acceptance': {
            'terms': ['accept', 'agrees', 'acknowledges', 'confirms', 'hereby accepts'],
            'weight': 0.3
        },
        'mutual_intent': {
            'terms': ['mutual', 'both parties', 'mutually agree', 'jointly', 'between the parties'],
            'weight': 0.2
        },
        'clarity': {
            'terms': ['clearly', 'expressly', 'specifically', 'unambiguously', 'explicitly'],
            'weight': 0.2
        }
    }
    
    found_elements = {category: [] for category in indicators}
    scores = {category: 0 for category in indicators}
    
    # Analyze each sentence for indicators
    for sent in doc.sents:
        sent_text = sent.text.lower()
        for category, info in indicators.items():
            for term in info['terms']:
                if term in sent_text:
                    found_elements[category].append(sent.text)
                    # Only count unique matches
                    if len(found_elements[category]) == 1:
                        scores[category] += 100
                    break
    
    # Calculate weighted score
    final_score = sum(
        min(100, scores[category]) * info['weight']
        for category, info in indicators.items()
    )
    
    # Generate specific suggestions
    suggestions = []
    if not found_elements['offer']:
        suggestions.append("Add clear offer language specifying what each party is providing")
    if not found_elements['acceptance']:
        suggestions.append("Include explicit acceptance terms from all parties")
    if not found_elements['mutual_intent']:
        suggestions.append("Add language demonstrating mutual agreement between parties")
    if not found_elements['clarity']:
        suggestions.append("Consider adding explicit language to clarify intentions")
    
    return {
        'score': final_score,
        'elements': found_elements,
        'suggestions': suggestions
    }

def analyze_completeness(text):
    """
    Analyzes the completeness and definiteness of essential terms.
    Evaluates critical elements and gap-fillers.
    """
    doc = nlp(text.lower())
    
    # Essential terms to check
    essential_terms = {
        'price_payment': {
            'terms': ['price', 'payment', 'consideration', 'fee', 'cost', 'amount'],
            'weight': 0.2
        },
        'duration': {
            'terms': ['term', 'duration', 'period', 'effective date', 'termination date'],
            'weight': 0.15
        },
        'scope': {
            'terms': ['scope', 'services', 'deliverables', 'work', 'obligations'],
            'weight': 0.2
        },
        'performance': {
            'terms': ['perform', 'deliver', 'provide', 'complete', 'fulfill'],
            'weight': 0.15
        },
        'definitions': {
            'terms': ['means', 'defined', 'shall mean', 'refers to', 'definition'],
            'weight': 0.15
        },
        'gap_fillers': {
            'terms': ['unforeseen', 'force majeure', 'reasonable efforts', 'best efforts', 'commercially reasonable'],
            'weight': 0.15
        }
    }
    
    found_terms = {category: [] for category in essential_terms}
    scores = {category: 0 for category in essential_terms}
    
    # Analyze sentences for essential terms
    for sent in doc.sents:
        sent_text = sent.text.lower()
        for category, info in essential_terms.items():
            if not found_terms[category]:  # Only look if we haven't found this term
                for term in info['terms']:
                    if term in sent_text:
                        found_terms[category].append(sent.text)
                        scores[category] = 100
                        break
    
    # Calculate weighted score
    final_score = sum(
        scores[category] * info['weight']
        for category, info in essential_terms.items()
    )
    
    # Identify missing terms
    missing_terms = [
        category.replace('_', ' ').title()
        for category, found in found_terms.items()
        if not found
    ]
    
    return {
        'score': final_score,
        'found_terms': found_terms,
        'missing_terms': missing_terms
    }

def analyze_consideration(text):
    """
    Analyzes consideration and bargain principles.
    Evaluates exchange of value and reasonableness.
    """
    doc = nlp(text.lower())
    
    consideration_elements = {
        'value_exchange': {
            'terms': ['payment', 'consideration', 'exchange', 'compensation', 'fee'],
            'weight': 0.3
        },
        'promises': {
            'terms': ['agrees to', 'shall', 'will provide', 'commits to', 'obligations'],
            'weight': 0.3
        },
        'mutuality': {
            'terms': ['mutual', 'both parties', 'each party', 'respectively', 'in return for'],
            'weight': 0.2
        },
        'fairness': {
            'terms': ['fair', 'reasonable', 'market value', 'good faith', 'equitable'],
            'weight': 0.2
        }
    }
    
    found_elements = {category: [] for category in consideration_elements}
    scores = {category: 0 for category in consideration_elements}
    
    # Analyze consideration elements
    for sent in doc.sents:
        sent_text = sent.text.lower()
        for category, info in consideration_elements.items():
            for term in info['terms']:
                if term in sent_text:
                    found_elements[category].append(sent.text)
                    if len(found_elements[category]) == 1:  # Only count first instance
                        scores[category] = 100
                    break
    
    # Calculate weighted score
    final_score = sum(
        scores[category] * info['weight']
        for category, info in consideration_elements.items()
    )
    
    # Generate suggestions
    suggestions = []
    if not found_elements['value_exchange']:
        suggestions.append("Specify the exchange of value or consideration")
    if not found_elements['promises']:
        suggestions.append("Include clear promises or commitments from both parties")
    if not found_elements['mutuality']:
        suggestions.append("Add language showing mutual exchange of value")
    if not found_elements['fairness']:
        suggestions.append("Consider adding terms about fairness or reasonableness")
    
    return {
        'score': final_score,
        'elements': found_elements,
        'suggestions': suggestions
    }

def analyze_integration_consistency(text):
    """
    Analyzes integration clauses and internal consistency.
    Evaluates completeness of agreement and conflicting terms.
    """
    doc = nlp(text.lower())
    
    # Integration and consistency elements
    elements = {
        'integration_clause': {
            'terms': ['entire agreement', 'complete agreement', 'integrated agreement', 'merger', 'supersedes'],
            'weight': 0.3
        },
        'prior_agreements': {
            'terms': ['prior', 'previous', 'supersede', 'replace', 'override'],
            'weight': 0.2
        },
        'amendments': {
            'terms': ['amendment', 'modification', 'change', 'alter', 'revise'],
            'weight': 0.2
        },
        'consistency': {
            'terms': ['consistent', 'accordance', 'pursuant to', 'subject to', 'notwithstanding'],
            'weight': 0.3
        }
    }
    
    found_elements = {category: [] for category in elements}
    scores = {category: 0 for category in elements}
    
    # Analyze each sentence
    for sent in doc.sents:
        sent_text = sent.text.lower()
        for category, info in elements.items():
            for term in info['terms']:
                if term in sent_text:
                    found_elements[category].append(sent.text)
                    if len(found_elements[category]) == 1:
                        scores[category] = 100
                    break
    
    # Calculate weighted score
    final_score = sum(
        scores[category] * info['weight']
        for category, info in elements.items()
    )
    
    # Check for integration clause
    has_integration = bool(found_elements['integration_clause'])
    
    # Analyze potential conflicts
    conflicts = []
    sentences = list(doc.sents)
    for i, sent1 in enumerate(sentences):
        for sent2 in sentences[i+1:]:
            # Look for potentially conflicting statements
            if any(term in sent1.text.lower() for term in ['shall', 'must', 'will', 'agree']) and \
               any(term in sent2.text.lower() for term in ['shall not', 'must not', 'will not', 'disagree']):
                conflicts.append(f"Potential conflict between: '{sent1.text}' and '{sent2.text}'")
    
    return {
        'score': final_score,
        'elements': found_elements,
        'has_integration_clause': has_integration,
        'conflicts': conflicts
    }

def analyze_risk_allocation(text):
    """
    Analyzes risk allocation, remedies, and dispute resolution.
    Evaluates damages, dispute resolution methods, and risk factors.
    """
    doc = nlp(text.lower())
    
    risk_elements = {
        'remedies': {
            'terms': ['damages', 'compensation', 'restitution', 'specific performance', 'relief'],
            'weight': 0.2
        },
        'dispute_resolution': {
            'terms': ['arbitration', 'mediation', 'litigation', 'dispute', 'court'],
            'weight': 0.2
        },
        'force_majeure': {
            'terms': ['force majeure', 'act of god', 'unforeseen', 'beyond control', 'extraordinary event'],
            'weight': 0.2
        },
        'indemnification': {
            'terms': ['indemnify', 'hold harmless', 'liability', 'responsible for', 'reimburse'],
            'weight': 0.2
        },
        'limitations': {
            'terms': ['limitation of liability', 'cap on damages', 'maximum liability', 'not liable for', 'exclude'],
            'weight': 0.2
        }
    }
    
    found_elements = {category: [] for category in risk_elements}
    scores = {category: 0 for category in risk_elements}
    
    # Analyze risk elements
    for sent in doc.sents:
        sent_text = sent.text.lower()
        for category, info in risk_elements.items():
            for term in info['terms']:
                if term in sent_text:
                    found_elements[category].append(sent.text)
                    if len(found_elements[category]) == 1:
                        scores[category] = 100
                    break
    
    # Calculate weighted score
    final_score = sum(
        scores[category] * info['weight']
        for category, info in risk_elements.items()
    )
    
    # Identify missing elements
    missing_elements = [
        category.replace('_', ' ').title()
        for category, found in found_elements.items()
        if not found
    ]
    
    return {
        'score': final_score,
        'elements': found_elements,
        'missing_elements': missing_elements
    }

def analyze_modification_provisions(text):
    """
    Analyzes modification, waiver, and termination provisions.
    Evaluates clarity of amendment processes and termination conditions.
    """
    doc = nlp(text.lower())
    
    provisions = {
        'modification': {
            'terms': ['amendment', 'modify', 'change', 'revise', 'update'],
            'weight': 0.25
        },
        'waiver': {
            'terms': ['waive', 'waiver', 'relinquish', 'forbearance', 'consent'],
            'weight': 0.25
        },
        'termination': {
            'terms': ['terminate', 'cancellation', 'end', 'discontinue', 'cease'],
            'weight': 0.25
        },
        'process': {
            'terms': ['writing', 'notice', 'mutual agreement', 'signed by', 'authorized'],
            'weight': 0.25
        }
    }
    
    found_elements = {category: [] for category in provisions}
    scores = {category: 0 for category in provisions}
    
    # Analyze provisions
    for sent in doc.sents:
        sent_text = sent.text.lower()
        for category, info in provisions.items():
            for term in info['terms']:
                if term in sent_text:
                    found_elements[category].append(sent.text)
                    if len(found_elements[category]) == 1:
                        scores[category] = 100
                    break
    
    # Calculate weighted score
    final_score = sum(
        scores[category] * info['weight']
        for category, info in provisions.items()
    )
    
    # Identify missing elements
    missing_elements = [
        category.title()
        for category, found in found_elements.items()
        if not found
    ]
    
    return {
        'score': final_score,
        'elements': found_elements,
        'missing_elements': missing_elements
    }

def analyze_statutory_compliance(text):
    """
    Analyzes compliance with statutory and doctrinal standards.
    Evaluates adherence to legal frameworks and fairness principles.
    """
    doc = nlp(text.lower())
    
    compliance_elements = {
        'ucc_references': {
            'terms': ['ucc', 'uniform commercial code', 'article 2', 'commercial law', 'sale of goods'],
            'weight': 0.2
        },
        'regulatory': {
            'terms': ['comply', 'accordance with law', 'applicable law', 'regulations', 'statutory'],
            'weight': 0.2
        },
        'consumer_protection': {
            'terms': ['consumer', 'disclosure', 'right to cancel', 'cooling off', 'notice period'],
            'weight': 0.2
        },
        'fairness': {
            'terms': ['fair', 'reasonable', 'equitable', 'good faith', 'commercially reasonable'],
            'weight': 0.2
        },
        'jurisdiction': {
            'terms': ['jurisdiction', 'governing law', 'venue', 'forum', 'choice of law'],
            'weight': 0.2
        }
    }
    
    found_elements = {category: [] for category in compliance_elements}
    scores = {category: 0 for category in compliance_elements}
    
    # Analyze compliance elements
    for sent in doc.sents:
        sent_text = sent.text.lower()
        for category, info in compliance_elements.items():
            for term in info['terms']:
                if term in sent_text:
                    found_elements[category].append(sent.text)
                    if len(found_elements[category]) == 1:
                        scores[category] = 100
                    break
    
    # Calculate weighted score
    final_score = sum(
        scores[category] * info['weight']
        for category, info in compliance_elements.items()
    )
    
    # Identify missing elements
    missing_elements = [
        category.replace('_', ' ').title()
        for category, found in found_elements.items()
        if not found
    ]
    
    return {
        'score': final_score,
        'elements': found_elements,
        'missing_elements': missing_elements
    }

def analyze_unconscionability(text):
    """
    Analyzes contract for potential unconscionability issues.
    Evaluates procedural and substantive fairness.
    """
    doc = nlp(text.lower())
    
    # Risk factors for unconscionability
    risk_factors = {
        'unilateral_terms': {
            'terms': ['sole discretion', 'absolute right', 'exclusive right', 'unilateral', 'reserves the right'],
            'weight': 0.25,
            'negative': True
        },
        'waiver_of_rights': {
            'terms': ['waive right', 'forfeit', 'relinquish claim', 'give up right', 'no right to'],
            'weight': 0.25,
            'negative': True
        },
        'fairness_indicators': {
            'terms': ['fair', 'reasonable', 'equitable', 'mutual', 'balanced'],
            'weight': 0.25,
            'negative': False
        },
        'negotiation_opportunity': {
            'terms': ['negotiate', 'discuss', 'agree upon', 'mutual agreement', 'both parties agree'],
            'weight': 0.25,
            'negative': False
        }
    }
    
    found_elements = {category: [] for category in risk_factors}
    scores = {category: 0 for category in risk_factors}
    
    # Analyze for unconscionability indicators
    for sent in doc.sents:
        sent_text = sent.text.lower()
        for category, info in risk_factors.items():
            for term in info['terms']:
                if term in sent_text:
                    found_elements[category].append(sent.text)
                    if len(found_elements[category]) == 1:
                        scores[category] = 100 if not info['negative'] else 0
                    break
                else:
                    scores[category] = 0 if not info['negative'] else 100
    
    # Calculate weighted score
    final_score = sum(
        scores[category] * info['weight']
        for category, info in risk_factors.items()
    )
    
    # Generate warnings for potentially unconscionable terms
    warnings = []
    for category, found in found_elements.items():
        if found and risk_factors[category]['negative']:
            warnings.append(f"Potentially unfair {category.replace('_', ' ')}: {found[0]}")
    
    return {
        'score': final_score,
        'elements': found_elements,
        'warnings': warnings
    }

def analyze_document(text):
    """
    Main function to analyze a legal document.
    Returns a comprehensive analysis with scores and suggestions.
    """
    try:
        # Perform all analyses
        analyses = {
            'Assent and Mutuality': analyze_assent_and_mutuality(text),
            'Completeness': analyze_completeness(text),
            'Consideration': analyze_consideration(text),
            'Integration and Consistency': analyze_integration_consistency(text),
            'Risk Allocation': analyze_risk_allocation(text),
            'Contract Structure': analyze_contract_structure(text),
            'Readability': analyze_readability(text),
            'Modification Provisions': analyze_modification_provisions(text),
            'Statutory Compliance': analyze_statutory_compliance(text),
            'Contract Fairness': analyze_unconscionability(text)
        }
        
        # Component weights
        weights = {
            'Assent and Mutuality': 0.12,
            'Completeness': 0.12,
            'Consideration': 0.12,
            'Integration and Consistency': 0.08,
            'Risk Allocation': 0.12,
            'Contract Structure': 0.08,
            'Readability': 0.08,
            'Modification Provisions': 0.08,
            'Statutory Compliance': 0.06,
            'Contract Fairness': 0.06
        }
        
        # Calculate overall score
        overall_score = 0
        for category, analysis in analyses.items():
            score = analysis.get('score', 0)  # Ensure we get a score even if missing
            overall_score += score * weights[category]
        
        # Generate response
        response = {
            'score': round(overall_score, 1),
            'analysis': {}
        }
        
        # Format each analysis component
        for category, analysis in analyses.items():
            response['analysis'][category] = {
                'score': round(analysis.get('score', 0), 1),
                'details': get_analysis_details(category, analysis),
                'findings': get_analysis_findings(category, analysis)
            }
        
        # Add recommendations
        response['recommendations'] = generate_recommendations(analyses)
        
        return response
        
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        return {
            'error': str(e),
            'status': 'failed'
        }

def get_analysis_details(category, analysis):
    """Helper function to get human-readable details for each category"""
    category_descriptions = {
        'Assent and Mutuality': 'Analysis of offer, acceptance, and mutual intent',
        'Completeness': 'Evaluation of essential terms and definitions',
        'Consideration': 'Analysis of value exchange and promises',
        'Integration and Consistency': 'Evaluation of merger clauses and internal consistency',
        'Risk Allocation': 'Analysis of risk distribution and remedies',
        'Contract Structure': 'Evaluation of key contract sections and organization',
        'Readability': 'Analysis of language clarity and complexity',
        'Modification Provisions': 'Evaluation of amendment and termination clauses',
        'Statutory Compliance': 'Analysis of legal framework compliance',
        'Contract Fairness': 'Detection of potentially unfair terms'
    }
    return category_descriptions.get(category, 'Analysis of contract elements')

def get_analysis_findings(category, analysis):
    """Helper function to extract relevant findings from analysis results"""
    if not analysis:
        return {}
        
    findings = {}
    
    if 'elements' in analysis:
        findings['found_elements'] = analysis['elements']
    if 'missing_sections' in analysis:
        findings['missing_elements'] = analysis['missing_sections']
    if 'metrics' in analysis:
        findings['metrics'] = analysis['metrics']
    if 'suggestions' in analysis:
        findings['suggestions'] = analysis['suggestions']
        
    return findings

def generate_recommendations(analyses):
    """
    Generates specific recommendations based on analysis results.
    """
    recommendations = []
    
    # Assent recommendations
    if analyses['Assent and Mutuality']['score'] < 100:
        recommendations.append("Add clear language demonstrating mutual assent and agreement between parties")
    
    # Completeness recommendations
    if analyses['Completeness']['missing_terms']:
        recommendations.append(f"Define essential terms: {', '.join(analyses['Completeness']['missing_terms'])}")
    
    # Consideration recommendations
    if analyses['Consideration']['score'] < 100:
        recommendations.append("Include explicit consideration clauses detailing the exchange of value")
    
    # Integration recommendations
    if not analyses['Integration and Consistency']['has_integration_clause']:
        recommendations.append("Add an integration/merger clause to establish this as the complete agreement")
    
    # Risk allocation recommendations
    if analyses['Risk Allocation']['missing_elements']:
        recommendations.append(f"Address risk allocation for: {', '.join(analyses['Risk Allocation']['missing_elements'])}")
    
    # Modification recommendations
    if analyses['Modification Provisions']['missing_elements']:
        recommendations.append(f"Include provisions for: {', '.join(analyses['Modification Provisions']['missing_elements'])}")
    
    # Compliance recommendations
    if analyses['Statutory Compliance']['missing_elements']:
        recommendations.append(f"Consider adding compliance elements for: {', '.join(analyses['Statutory Compliance']['missing_elements'])}")
    
    # Unconscionability warnings
    if analyses['Contract Fairness']['warnings']:
        recommendations.extend(analyses['Contract Fairness']['warnings'])
    
    return recommendations

# Default repository - in production, this should be loaded from files
repository = [
    "The plaintiff filed a lawsuit for breach of contract.",
    "An affidavit was presented in court to support the case.",
    "Jurisdiction is determined by the geographical boundaries of the court."
]
