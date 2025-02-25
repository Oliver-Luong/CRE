"""
Contract evaluation criteria and patterns.
"""

# Assent and Mutuality patterns
ASSENT_PATTERNS = {
    'offer_patterns': [
        'offer',
        'propose',
        'agrees to provide',
        'shall provide',
        'will provide'
    ],
    'acceptance_patterns': [
        'accept',
        'agrees',
        'acknowledges',
        'confirms',
        'hereby agrees'
    ],
    'mutual_intent_patterns': [
        'mutual',
        'both parties',
        'mutually agree',
        'jointly',
        'between the parties'
    ]
}

# Essential Terms patterns
ESSENTIAL_TERMS = {
    'price_terms': [
        'price',
        'payment',
        'consideration',
        'fee',
        'cost',
        'compensation'
    ],
    'duration_terms': [
        'term',
        'duration',
        'period',
        'timeline',
        'schedule',
        'effective date'
    ],
    'scope_terms': [
        'scope',
        'services',
        'deliverables',
        'work',
        'obligations',
        'responsibilities'
    ],
    'performance_terms': [
        'perform',
        'deliver',
        'provide',
        'complete',
        'fulfill',
        'execute'
    ]
}

# Risk Allocation patterns
RISK_PATTERNS = {
    'force_majeure': [
        'force majeure',
        'act of god',
        'beyond reasonable control',
        'unforeseen circumstances'
    ],
    'indemnification': [
        'indemnify',
        'hold harmless',
        'liability',
        'indemnification',
        'defend'
    ],
    'limitation_liability': [
        'limitation of liability',
        'cap on damages',
        'maximum liability',
        'not liable for'
    ],
    'dispute_resolution': [
        'arbitration',
        'mediation',
        'jurisdiction',
        'venue',
        'governing law',
        'dispute resolution'
    ]
}

# Integration and Merger patterns
INTEGRATION_PATTERNS = [
    'entire agreement',
    'complete agreement',
    'integrated agreement',
    'merger',
    'integration clause',
    'supersedes',
    'complete understanding'
]

# Modification and Amendment patterns
MODIFICATION_PATTERNS = [
    'amendment',
    'modification',
    'alter',
    'change',
    'revise',
    'supplement'
]

# Contract Structure sections
CONTRACT_SECTIONS = {
    'parties': [
        'between',
        'party',
        'parties',
        'agreement between'
    ],
    'recitals': [
        'whereas',
        'background',
        'recitals',
        'purpose'
    ],
    'definitions': [
        'means',
        'defined',
        'definitions',
        'interpreted as'
    ],
    'term_and_termination': [
        'term',
        'duration',
        'terminate',
        'termination',
        'expiration'
    ],
    'governing_law': [
        'govern',
        'jurisdiction',
        'applicable law',
        'construed under'
    ]
}

# Scoring weights for different components
EVALUATION_WEIGHTS = {
    'assent': 0.15,
    'completeness': 0.15,
    'consideration': 0.15,
    'integration': 0.10,
    'risk_allocation': 0.15,
    'structure': 0.10,
    'readability': 0.10,
    'jargon': 0.10
}

# Threshold values for scoring
SCORE_THRESHOLDS = {
    'excellent': 90,
    'good': 75,
    'fair': 60,
    'poor': 45
}
