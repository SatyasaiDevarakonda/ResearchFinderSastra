"""
SASTRA Research Finder - Preprocessing Module
Builds accurate author profiles and search indices from publications data.
"""

import pandas as pd
import pickle
import re
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Any

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
EXCEL_FILE = DATA_DIR / "SASTRA_Publications_2024-25.xlsx"


def clean_text(text):
    """Clean and normalize text."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    return text


def extract_author_ids(author_ids_str):
    """Extract individual author IDs from semicolon-separated string."""
    if pd.isna(author_ids_str) or not isinstance(author_ids_str, str):
        return []
    # Split by semicolon and clean
    ids = []
    for aid in author_ids_str.split(';'):
        aid = aid.strip()
        if aid and aid.replace('.', '').isdigit():  # Valid numeric ID
            ids.append(aid)
    return ids


def extract_author_names(authors_str):
    """Extract individual author names from semicolon-separated string."""
    if pd.isna(authors_str) or not isinstance(authors_str, str):
        return []
    names = []
    for name in authors_str.split(';'):
        name = name.strip()
        if name:
            names.append(name)
    return names


def extract_countries_from_affiliations(affiliations_str):
    """Extract country names from affiliation string."""
    if pd.isna(affiliations_str) or not isinstance(affiliations_str, str):
        return []
    
    # Common country patterns in affiliations
    country_patterns = [
        r'\bIndia\b', r'\bUSA\b', r'\bUnited States\b', r'\bUK\b', r'\bUnited Kingdom\b',
        r'\bChina\b', r'\bJapan\b', r'\bGermany\b', r'\bFrance\b', r'\bCanada\b',
        r'\bAustralia\b', r'\bSingapore\b', r'\bMalaysia\b', r'\bThailand\b',
        r'\bSouth Korea\b', r'\bNetherlands\b', r'\bSweden\b', r'\bSwitzerland\b',
        r'\bItaly\b', r'\bSpain\b', r'\bBrazil\b', r'\bMexico\b', r'\bRussia\b',
        r'\bSouth Africa\b', r'\bEgypt\b', r'\bSaudi Arabia\b', r'\bUAE\b',
        r'\bNew Zealand\b', r'\bIreland\b', r'\bBelgium\b', r'\bAustria\b',
        r'\bDenmark\b', r'\bNorway\b', r'\bFinland\b', r'\bPoland\b',
        r'\bTurkey\b', r'\bIran\b', r'\bPakistan\b', r'\bBangladesh\b',
        r'\bSri Lanka\b', r'\bNepal\b', r'\bIndonesia\b', r'\bPhilippines\b',
        r'\bVietnam\b', r'\bIsrael\b', r'\bGreece\b', r'\bPortugal\b'
    ]
    
    countries = []
    for pattern in country_patterns:
        matches = re.findall(pattern, affiliations_str, re.IGNORECASE)
        for match in matches:
            # Normalize country name
            country = match.strip()
            if country.lower() in ['usa', 'united states']:
                country = 'United States'
            elif country.lower() in ['uk', 'united kingdom']:
                country = 'United Kingdom'
            else:
                country = country.title()
            
            if country not in countries:
                countries.append(country)
    
    return countries


def extract_keywords_from_text(text, min_length=3):
    """Extract meaningful keywords from text (abstract, title, keywords)."""
    if not text or pd.isna(text):
        return []
    
    text = text.lower()
    
    # Common stopwords to filter out
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'this', 'that', 'these', 'those', 'it', 'its', 'which', 'who', 'whom',
        'what', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both',
        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'also',
        'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'between', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
        'any', 'about', 'because', 'while', 'being', 'having', 'using', 'used',
        'shows', 'shown', 'show', 'based', 'however', 'therefore', 'thus',
        'paper', 'study', 'research', 'result', 'results', 'method', 'approach',
        'proposed', 'work', 'article', 'present', 'presents', 'presented',
        'obtained', 'achieved', 'performs', 'performance', 'various', 'different',
        'new', 'novel', 'several', 'many', 'first', 'second', 'third', 'one',
        'two', 'three', 'four', 'five', 'high', 'low', 'large', 'small',
        'important', 'significant', 'recently', 'mainly', 'particularly',
        'including', 'without', 'among', 'within', 'across', 'along', 'around',
        'upon', 'well', 'still', 'found', 'make', 'made', 'like', 'over',
        'such', 'even', 'most', 'use', 'uses', 'due', 'via', 'per', 'etc',
        'india', 'university', 'college', 'department', 'sastra', 'thanjavur'
    }
    
    # Extract words
    words = re.findall(r'\b[a-z][a-z0-9\-]+\b', text)
    
    # Filter and clean
    keywords = []
    for word in words:
        word = word.strip('-')
        if len(word) >= min_length and word not in stopwords:
            keywords.append(word)
    
    return keywords


def extract_bigrams_trigrams(text, min_length=3):
    """Extract meaningful bigrams and trigrams from text."""
    if not text or pd.isna(text):
        return []
    
    text = text.lower()
    words = re.findall(r'\b[a-z][a-z0-9]+\b', text)
    
    ngrams = []
    
    # Bigrams
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i+1]}"
        if len(bigram) >= min_length * 2:
            ngrams.append(bigram)
    
    # Trigrams
    for i in range(len(words) - 2):
        trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
        if len(trigram) >= min_length * 3:
            ngrams.append(trigram)
    
    return ngrams


def parse_author_keywords(keywords_str):
    """Parse author keywords from semicolon-separated string."""
    if pd.isna(keywords_str) or not isinstance(keywords_str, str):
        return []
    
    keywords = []
    for kw in keywords_str.split(';'):
        kw = kw.strip().lower()
        if kw and len(kw) >= 2:
            keywords.append(kw)
    
    return keywords


def build_author_name_id_mapping(df):
    """Build accurate mapping between author names and IDs."""
    # author_id -> set of name variants
    author_id_to_names = defaultdict(set)
    
    # name -> set of author_ids (for reverse lookup)
    name_to_author_ids = defaultdict(set)
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building name-ID mappings"):
        author_ids = extract_author_ids(row['Author(s) ID'])
        
        # Parse from "Author full names" column (most accurate)
        full_names_str = row.get('Author full names', '')
        if full_names_str and isinstance(full_names_str, str):
            # Format: "Name (ID); Name (ID); ..."
            pattern = r'([^;]+?)\s*\((\d+)\)'
            matches = re.findall(pattern, full_names_str)
            for name, aid in matches:
                name = name.strip()
                if name and aid:
                    author_id_to_names[aid].add(name)
                    # Add normalized versions
                    name_lower = name.lower()
                    name_to_author_ids[name_lower].add(aid)
                    # Also add first/last name parts
                    name_parts = re.split(r'[,\s]+', name)
                    for part in name_parts:
                        part = part.strip().lower()
                        if len(part) >= 2:
                            name_to_author_ids[part].add(aid)
        
        # Also parse from "Authors" column
        authors_str = row.get('Authors', '')
        short_names = extract_author_names(authors_str)
        
        # Try to match short names to IDs by position
        if len(short_names) == len(author_ids):
            for name, aid in zip(short_names, author_ids):
                author_id_to_names[aid].add(name)
                name_lower = name.lower()
                name_to_author_ids[name_lower].add(aid)
                # Add parts
                name_parts = re.split(r'[,\s]+', name)
                for part in name_parts:
                    part = part.strip().lower()
                    if len(part) >= 2:
                        name_to_author_ids[part].add(aid)
    
    return dict(author_id_to_names), dict(name_to_author_ids)


def build_publications_index(df):
    """Build publication records with all metadata."""
    publications = {}
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Building publications index"):
        pub_id = str(idx)
        
        # Extract all keywords
        author_kws = parse_author_keywords(row.get('Author Keywords', ''))
        index_kws = parse_author_keywords(row.get('Index Keywords', ''))
        
        abstract = clean_text(row.get('Abstract', ''))
        title = clean_text(row.get('Title', ''))
        
        # Extract keywords from abstract
        abstract_keywords = extract_keywords_from_text(abstract)
        title_keywords = extract_keywords_from_text(title)
        
        # Extract n-grams from abstract
        abstract_ngrams = extract_bigrams_trigrams(abstract)
        
        # Combine all keywords
        all_keywords = list(set(author_kws + index_kws + abstract_keywords[:50] + title_keywords))
        
        # Extract document type
        doc_type = clean_text(row.get('Document Type', 'Article'))
        if not doc_type:
            doc_type = 'Article'
        
        # Extract countries from affiliations
        affiliations = clean_text(row.get('Affiliations', ''))
        countries = extract_countries_from_affiliations(affiliations)
        
        publications[pub_id] = {
            'pub_id': pub_id,
            'title': title,
            'abstract': abstract,
            'abstract_lower': abstract.lower(),
            'year': int(row.get('Year', 0)),
            'authors': clean_text(row.get('Authors', '')),
            'author_ids': extract_author_ids(row['Author(s) ID']),
            'source': clean_text(row.get('Source title', '')),
            'citations': int(row.get('Cited by', 0)),
            'doi': clean_text(row.get('DOI', '')),
            'affiliations': affiliations,
            'author_keywords': author_kws,
            'index_keywords': index_kws,
            'all_keywords': all_keywords,
            'abstract_ngrams': abstract_ngrams[:100],  # Limit ngrams
            'document_type': doc_type,
            'countries': countries,
        }
    
    return publications


def build_author_profiles(df, publications, author_id_to_names):
    """Build comprehensive author profiles."""
    author_profiles = {}
    
    # Group publications by author
    author_pubs = defaultdict(list)
    for pub_id, pub in publications.items():
        for aid in pub['author_ids']:
            author_pubs[aid].append(pub_id)
    
    for author_id, pub_ids in tqdm(author_pubs.items(), desc="Building author profiles"):
        # Get name variants
        name_variants = list(author_id_to_names.get(author_id, set()))
        if not name_variants:
            name_variants = ['Unknown Author']
        
        # Collect all publications for this author
        author_publications = []
        total_citations = 0
        all_keywords = []
        affiliations = set()
        citation_list = []
        
        for pub_id in pub_ids:
            pub = publications[pub_id]
            author_publications.append({
                'pub_id': pub_id,
                'title': pub['title'],
                'abstract': pub['abstract'],
                'year': pub['year'],
                'authors': pub['authors'],
                'source': pub['source'],
                'citations': pub['citations'],
                'keywords': ', '.join(pub['author_keywords'][:10]),
                'document_type': pub['document_type'],
                'countries': pub['countries'],
            })
            total_citations += pub['citations']
            citation_list.append(pub['citations'])
            all_keywords.extend(pub['all_keywords'])
            if pub['affiliations']:
                affiliations.add(pub['affiliations'])
        
        # Count keyword frequencies
        keyword_counts = defaultdict(int)
        for kw in all_keywords:
            keyword_counts[kw] += 1
        
        top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:30]
        
        author_profiles[author_id] = {
            'author_id': author_id,
            'name_variants': name_variants,
            'pub_count': len(pub_ids),
            'total_citations': total_citations,
            'publications': sorted(author_publications, key=lambda x: x['year'], reverse=True),
            'pub_ids': pub_ids,
            'top_keywords': top_keywords,
            'affiliations': list(affiliations)[:5],
            'all_keywords_set': set(k for k, _ in top_keywords),
            'citation_list': citation_list,  # For histogram
        }
    
    return author_profiles


def build_keyword_index(publications):
    """Build inverted index: keyword -> list of (pub_id, score)."""
    keyword_index = defaultdict(list)
    
    for pub_id, pub in tqdm(publications.items(), desc="Building keyword index"):
        # Author keywords (highest weight)
        for kw in pub['author_keywords']:
            kw_lower = kw.lower()
            keyword_index[kw_lower].append((pub_id, 3.0))
        
        # Index keywords (high weight)
        for kw in pub['index_keywords']:
            kw_lower = kw.lower()
            keyword_index[kw_lower].append((pub_id, 2.0))
        
        # Title keywords (medium weight)
        title_kws = extract_keywords_from_text(pub['title'])
        for kw in title_kws:
            keyword_index[kw].append((pub_id, 1.5))
        
        # Abstract keywords (base weight)
        abstract_kws = extract_keywords_from_text(pub['abstract'])[:100]
        for kw in abstract_kws:
            keyword_index[kw].append((pub_id, 1.0))
        
        # N-grams from abstract (lower weight but useful for phrases)
        for ngram in pub['abstract_ngrams'][:50]:
            keyword_index[ngram].append((pub_id, 0.8))
    
    return dict(keyword_index)


def build_abstract_keywords_map(publications):
    """Build mapping of pub_id -> extracted keywords from abstract."""
    abstract_keywords = {}
    
    for pub_id, pub in publications.items():
        # Combine author keywords and abstract keywords
        kws = pub['author_keywords'] + extract_keywords_from_text(pub['abstract'])[:30]
        abstract_keywords[pub_id] = list(set(kws))[:50]
    
    return abstract_keywords


def main():
    print("=" * 60)
    print("SASTRA Research Finder - Preprocessing")
    print("=" * 60)
    
    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check for Excel file
    if not EXCEL_FILE.exists():
        # Try alternate location
        alt_path = BASE_DIR / "SASTRA_Publications_2024-25.xlsx"
        if alt_path.exists():
            import shutil
            shutil.copy(alt_path, EXCEL_FILE)
        else:
            print(f"ERROR: Excel file not found at {EXCEL_FILE}")
            print("Please place SASTRA_Publications_2024-25.xlsx in the data/ folder")
            return False
    
    print(f"\nLoading Excel file: {EXCEL_FILE}")
    df = pd.read_excel(EXCEL_FILE)
    print(f"Loaded {len(df)} publications")
    
    # Build all indices
    print("\n[1/6] Building author name-ID mappings...")
    author_id_to_names, name_to_author_ids = build_author_name_id_mapping(df)
    print(f"  Found {len(author_id_to_names)} unique author IDs")
    print(f"  Found {len(name_to_author_ids)} unique name variants")
    
    print("\n[2/6] Building publications index...")
    publications = build_publications_index(df)
    print(f"  Indexed {len(publications)} publications")
    
    print("\n[3/6] Building author profiles...")
    author_profiles = build_author_profiles(df, publications, author_id_to_names)
    print(f"  Created {len(author_profiles)} author profiles")
    
    print("\n[4/6] Building keyword index...")
    keyword_index = build_keyword_index(publications)
    print(f"  Indexed {len(keyword_index)} unique keywords/phrases")
    
    print("\n[5/6] Building abstract keywords map...")
    abstract_keywords = build_abstract_keywords_map(publications)
    print(f"  Mapped keywords for {len(abstract_keywords)} publications")
    
    # Save all data
    print("\n[6/6] Saving preprocessed data...")
    
    with open(DATA_DIR / "publications.pkl", 'wb') as f:
        pickle.dump(publications, f)
    
    with open(DATA_DIR / "author_profiles.pkl", 'wb') as f:
        pickle.dump(author_profiles, f)
    
    mappings = {
        'author_id_to_names': author_id_to_names,
        'name_to_author_ids': name_to_author_ids,
    }
    with open(DATA_DIR / "mappings.pkl", 'wb') as f:
        pickle.dump(mappings, f)
    
    with open(DATA_DIR / "keyword_index.pkl", 'wb') as f:
        pickle.dump(keyword_index, f)
    
    with open(DATA_DIR / "abstract_keywords.pkl", 'wb') as f:
        pickle.dump(abstract_keywords, f)
    
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)
    print(f"\nSaved files to {DATA_DIR}:")
    print("  - publications.pkl")
    print("  - author_profiles.pkl")
    print("  - mappings.pkl")
    print("  - keyword_index.pkl")
    print("  - abstract_keywords.pkl")
    
    # ===== PHASE 1 + PHASE 2: THEMATIC AREAS PREPROCESSING =====
    print("\n" + "="*70)
    print("THEMATIC AREAS: Pre-computing Rankings & Combinations")
    print("="*70)
    
    try:
        from thematic_areas import ThematicAreasEngine
        
        engine = ThematicAreasEngine(publications, author_profiles)
        
        # PHASE 1: Single Theme Rankings
        print("\nPhase 1: Computing single theme rankings...")
        single_rankings = engine.compute_single_theme_rankings()
        
        with open(DATA_DIR / "thematic_single_rankings.pkl", 'wb') as f:
            pickle.dump(single_rankings, f)
        
        theme_count = sum(1 for v in single_rankings.values() if len(v) > 0)
        print(f"✓ Generated rankings for {theme_count} themes")
        print("✓ Saved thematic_single_rankings.pkl")
        
        # PHASE 2: Interdisciplinary Combinations
        print("\nPhase 2: Computing interdisciplinary combinations...")
        
        APPROVED_COMBINATIONS = [
            # ============================================================
    # HEALTHCARE + TECHNOLOGY (35 combinations)
    # Rationale: AI-driven diagnostics, telemedicine, drug discovery
    # ============================================================
    
    # Core AI + Medical Imaging (8)
    ('Machine Learning', 'Medical Imaging'),
    ('Deep Learning', 'Medical Imaging'),
    ('Computer Vision', 'Medical Imaging'),
    ('Transfer Learning', 'Medical Imaging'),
    ('Explainable AI', 'Medical Imaging'),
    ('Computer Vision', 'Biomedicine'),
    ('Deep Learning', 'Biomedicine'),
    ('Machine Learning', 'Biomedicine'),
    
    # AI + Healthcare Analytics (7)
    ('Machine Learning', 'Healthcare Analytics'),
    ('Deep Learning', 'Healthcare Analytics'),
    ('NLP', 'Healthcare Analytics'),
    ('Time Series Analysis', 'Healthcare Analytics'),
    ('Explainable AI', 'Healthcare Analytics'),
    ('Machine Learning', 'Precision Medicine'),
    ('Deep Learning', 'Precision Medicine'),
    
    # AI + Drug Discovery & Bioinformatics (8)
    ('Machine Learning', 'Drug Discovery'),
    ('Deep Learning', 'Drug Discovery'),
    ('Machine Learning', 'Bioinformatics'),
    ('Deep Learning', 'Bioinformatics'),
    ('NLP', 'Bioinformatics'),
    ('Drug Discovery', 'Nanotechnology'),
    ('Bioinformatics', 'Drug Discovery'),
    ('Computational Biology', 'Drug Discovery'),
    
    # IoT + Healthcare (6)
    ('Internet of Things', 'Healthcare Analytics'),
    ('Internet of Things', 'Telemedicine & Digital Health'),
    ('Edge Computing', 'Healthcare Analytics'),
    ('Mobile Computing', 'Telemedicine & Digital Health'),
    ('Wearable Devices', 'Healthcare Analytics'),
    ('IoT', 'Medical Imaging'),
    
    # 3-Theme Healthcare Combinations (6)
    ('Machine Learning', 'Medical Imaging', 'Computer Vision'),
    ('Deep Learning', 'Medical Imaging', 'Healthcare Analytics'),
    ('Machine Learning', 'Healthcare Analytics', 'Bioinformatics'),
    ('Computer Vision', 'Medical Imaging', 'Deep Learning'),
    ('IoT', 'Healthcare Analytics', 'Edge Computing'),
    ('ML', 'Drug Discovery', 'Bioinformatics'),
    
    # ============================================================
    # SMART SYSTEMS & IOT APPLICATIONS (25 combinations)
    # Rationale: Smart cities, precision agriculture, industrial IoT
    # ============================================================
    
    # IoT + AI/ML (8)
    ('Machine Learning', 'Internet of Things'),
    ('Deep Learning', 'Internet of Things'),
    ('Edge Computing', 'Machine Learning'),
    ('Edge Computing', 'Deep Learning'),
    ('Federated Learning', 'Internet of Things'),
    ('Computer Vision', 'Internet of Things'),
    ('Time Series Analysis', 'Internet of Things'),
    ('Reinforcement Learning', 'Internet of Things'),
    
    # IoT + Wireless & Networking (5)
    ('Internet of Things', 'Wireless Communications'),
    ('IoT', 'Network Science'),
    ('Edge Computing', 'Wireless Communications'),
    ('Mobile Computing', 'IoT'),
    ('5G', 'Internet of Things'),
    
    # IoT + Environmental & Energy (5)
    ('Internet of Things', 'Environmental Engineering'),
    ('IoT', 'Renewable Energy'),
    ('IoT', 'Energy Efficiency'),
    ('IoT', 'Smart Grid'),
    ('Edge Computing', 'Renewable Energy'),
    
    # IoT + Automation & Control (4)
    ('Internet of Things', 'Robotics & Automation'),
    ('IoT', 'Control Systems'),
    ('Edge Computing', 'Robotics & Automation'),
    ('IoT', 'Autonomous Systems'),
    
    # 3-Theme IoT Combinations (3)
    ('IoT', 'Machine Learning', 'Cybersecurity'),
    ('Edge Computing', 'IoT', 'Deep Learning'),
    ('IoT', 'Renewable Energy', 'Optimization'),
    
    # ============================================================
    # CYBERSECURITY & PRIVACY (15 combinations)
    # Rationale: Digital security, data protection, secure systems
    # ============================================================
    
    # Security + AI/ML (6)
    ('Machine Learning', 'Cybersecurity'),
    ('Deep Learning', 'Cybersecurity'),
    ('Explainable AI', 'Cybersecurity'),
    ('Federated Learning', 'Privacy & Data Protection'),
    ('ML', 'Privacy & Data Protection'),
    ('NLP', 'Cybersecurity'),
    
    # Security + Cloud/IoT (5)
    ('Cybersecurity', 'Cloud Computing'),
    ('Cybersecurity', 'Internet of Things'),
    ('Privacy & Data Protection', 'Cloud Computing'),
    ('Blockchain', 'Cybersecurity'),
    ('Edge Computing', 'Cybersecurity'),
    
    # Blockchain Applications (4)
    ('Blockchain', 'Supply Chain Analytics'),
    ('Blockchain', 'Financial Analytics'),
    ('Blockchain', 'Healthcare Analytics'),
    ('Blockchain', 'IoT'),
    
    # ============================================================
    # INDUSTRY 4.0 & MANUFACTURING (20 combinations)
    # Rationale: Smart manufacturing, predictive maintenance, quality control
    # ============================================================
    
    # Robotics + AI (7)
    ('Machine Learning', 'Robotics & Automation'),
    ('Deep Learning', 'Robotics & Automation'),
    ('Computer Vision', 'Robotics & Automation'),
    ('Reinforcement Learning', 'Robotics & Automation'),
    ('Reinforcement Learning', 'Autonomous Systems'),
    ('Computer Vision', 'Autonomous Systems'),
    ('NLP', 'Human-Robot Interaction'),
    
    # Digital Twin & Predictive Maintenance (5)
    ('Digital Twin', 'Machine Learning'),
    ('Digital Twin', 'IoT'),
    ('Digital Twin', 'Predictive Maintenance'),
    ('Time Series Analysis', 'Predictive Maintenance'),
    ('ML', 'Predictive Maintenance'),
    
    # Manufacturing + Optimization (5)
    ('Operations Research', 'Supply Chain Analytics'),
    ('Optimization Algorithms', 'Supply Chain Analytics'),
    ('Machine Learning', 'Operations Research'),
    ('Supply Chain Analytics', 'IoT'),
    ('Optimization', 'Robotics & Automation'),
    
    # 3-Theme Manufacturing Combinations (3)
    ('Digital Twin', 'IoT', 'Machine Learning'),
    ('Robotics', 'Computer Vision', 'Machine Learning'),
    ('Robotics', 'Control Systems', 'Machine Learning'),
    
    # ============================================================
    # SUSTAINABLE ENERGY & ENVIRONMENT (18 combinations)
    # Rationale: Clean energy, climate action, resource optimization
    # ============================================================
    
    # AI + Renewable Energy (6)
    ('Machine Learning', 'Renewable Energy'),
    ('Deep Learning', 'Renewable Energy'),
    ('Time Series Analysis', 'Renewable Energy'),
    ('ML', 'Energy Storage'),
    ('Optimization Algorithms', 'Renewable Energy'),
    ('Computer Vision', 'Solar Energy'),
    
    # AI + Power Systems (4)
    ('Machine Learning', 'Power Systems'),
    ('Deep Learning', 'Power Systems'),
    ('Optimization', 'Smart Grid'),
    ('IoT', 'Smart Grid'),
    
    # AI + Environmental (5)
    ('Machine Learning', 'Environmental Engineering'),
    ('Computer Vision', 'Environmental Monitoring'),
    ('IoT', 'Environmental Engineering'),
    ('Deep Learning', 'Environmental Engineering'),
    ('Time Series Analysis', 'Environmental Engineering'),
    
    # 3-Theme Energy Combinations (3)
    ('ML', 'Renewable Energy', 'Optimization'),
    ('IoT', 'Smart Grid', 'Machine Learning'),
    ('Power Systems', 'Renewable Energy', 'ML'),
    
    # ============================================================
    # FINANCIAL TECHNOLOGY & BUSINESS (12 combinations)
    # Rationale: FinTech, fraud detection, market prediction
    # ============================================================
    
    # AI + Finance (6)
    ('Machine Learning', 'Financial Analytics'),
    ('Deep Learning', 'Financial Analytics'),
    ('Time Series Analysis', 'Financial Analytics'),
    ('NLP', 'Financial Analytics'),
    ('Reinforcement Learning', 'Financial Analytics'),
    ('Explainable AI', 'Financial Analytics'),
    
    # Blockchain + Finance (3)
    ('Blockchain', 'Financial Analytics'),
    ('Blockchain', 'Cybersecurity', 'Finance'),
    ('Privacy', 'Financial Analytics'),
    
    # 3-Theme Finance Combinations (3)
    ('ML', 'Financial Analytics', 'Time Series'),
    ('NLP', 'Sentiment Analysis', 'Financial Analytics'),
    ('Deep Learning', 'Financial Analytics', 'Risk Management'),
    
    # ============================================================
    # ADVANCED MATERIALS & NANOTECHNOLOGY (10 combinations)
    # Rationale: Material innovation, nanotech applications
    # ============================================================
    
    # AI + Materials (5)
    ('Machine Learning', 'Materials Science'),
    ('Deep Learning', 'Materials Science'),
    ('Machine Learning', 'Nanotechnology'),
    ('Computer Vision', 'Materials Science'),
    ('ML', 'Polymer Science'),
    
    # Nano + Other Domains (5)
    ('Nanotechnology', 'Drug Discovery'),
    ('Nanotechnology', 'Energy Storage'),
    ('Nanotechnology', 'Biomedicine'),
    ('Materials Science', 'Renewable Energy'),
    ('Polymer Science', 'Biomedicine'),
    
    # ============================================================
    # SIGNAL/SPEECH PROCESSING & COMMUNICATIONS (8 combinations)
    # Rationale: 5G, audio AI, wireless optimization
    # ============================================================
    
    ('Deep Learning', 'Speech & Audio'),
    ('NLP', 'Speech & Audio'),
    ('Machine Learning', 'Signal Processing'),
    ('Deep Learning', 'Signal Processing'),
    ('Signal Processing', 'Medical Imaging'),
    ('Wireless Communications', '5G'),
    ('ML', 'Wireless Communications'),
    ('Edge Computing', 'Wireless Communications'),
    
    # ============================================================
    # CONTROL SYSTEMS & AUTOMATION (7 combinations)
    # Rationale: Process control, adaptive systems
    # ============================================================
    
    ('Machine Learning', 'Control Systems'),
    ('Reinforcement Learning', 'Control Systems'),
    ('Optimization', 'Control Systems'),
    ('Control Systems', 'Power Systems'),
    ('Control Systems', 'Robotics & Automation'),
    ('Fuzzy Logic', 'Control Systems'),
    ('Adaptive Control', 'Robotics'),
        ]
        print(f"\n📊 Total Combinations: {len(APPROVED_COMBINATIONS)}")
        print(f"   - 2-Theme Combinations: {sum(1 for c in APPROVED_COMBINATIONS if len(c) == 2)}")
        print(f"   - 3-Theme Combinations: {sum(1 for c in APPROVED_COMBINATIONS if len(c) == 3)}")
        
        def build_team(themes, rank):
            members = []
            total_cites = 0
            for theme in themes:
                flist = single_rankings.get(theme, [])
                if rank < len(flist):
                    f = flist[rank]
                    members.append({
                        'author_id': f['author_id'],
                        'name': f['primary_name'],
                        'theme': theme,
                        'cite_score': f['total_cite_score'],
                        'paper_count': f['paper_count'],
                        'papers': f['papers'][:5]
                    })
                    total_cites += f['total_cite_score']
            if len(members) == len(themes):
                return {
                    'team_number': rank + 1,
                    'members': members,
                    'total_cite_score': total_cites,
                    'average_cite_score': total_cites / len(members),
                    'themes': themes
                }
            return None
        
        combos = {}
        for combo in APPROVED_COMBINATIONS:
            themes = list(combo)
            if all(t in single_rankings and len(single_rankings[t]) > 0 for t in themes):
                teams = []
                for r in range(10):
                    team = build_team(themes, r)
                    if team:
                        teams.append(team)
                if len(teams) >= 3:
                    key = ' + '.join(themes)
                    combos[key] = {'themes': themes, 'teams': teams, 'total_teams': len(teams)}
                    print(f"✓ {key}: {len(teams)} teams")
        
        with open(DATA_DIR / "thematic_combinations.pkl", 'wb') as f:
            pickle.dump(combos, f)
        
        print(f"\n✓ Generated {len(combos)} interdisciplinary combinations")
        print("✓ Saved thematic_combinations.pkl")
        print("="*70)
        
    except Exception as e:
        print(f"\nWarning: Could not build thematic combinations: {e}")
        print("This is optional - the main preprocessing was successful.")
    
    return True



if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ All preprocessing completed successfully!")
    else:
        print("\n❌ Preprocessing failed!")