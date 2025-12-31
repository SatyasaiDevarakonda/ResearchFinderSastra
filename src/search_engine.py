"""
SASTRA Research Finder - Search Engine Module
Highly accurate keyword, skill, and author-based search.
"""

import pickle
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Optional, Set, Tuple
import math

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"


def extract_keywords(text: str, min_length: int = 3, max_keywords: int = 20) -> List[str]:
    """Extract keywords from text for search."""
    if not text:
        return []
    
    text = text.lower()
    
    # Common stopwords
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
        'based', 'model', 'models', 'system', 'systems', 'method', 'methods',
        'approach', 'proposed', 'paper', 'study', 'research', 'result', 'results'
    }
    
    # Extract words
    words = re.findall(r'\b[a-z][a-z0-9\-]+\b', text)
    
    # Filter
    keywords = []
    seen = set()
    for word in words:
        word = word.strip('-')
        if len(word) >= min_length and word not in stopwords and word not in seen:
            keywords.append(word)
            seen.add(word)
    
    return keywords[:max_keywords]


class SearchEngine:
    """Main search engine with accurate keyword and author matching."""
    
    def __init__(self):
        """Initialize and load all preprocessed data."""
        self.publications = {}
        self.author_profiles = {}
        self.keyword_index = {}
        self.abstract_keywords = {}
        self.author_id_to_names = {}
        self.name_to_author_ids = {}
        
        self._load_data()
    
    def _load_data(self):
        """Load all preprocessed data files."""
        try:
            with open(DATA_DIR / "publications.pkl", 'rb') as f:
                self.publications = pickle.load(f)
            
            with open(DATA_DIR / "author_profiles.pkl", 'rb') as f:
                self.author_profiles = pickle.load(f)
            
            with open(DATA_DIR / "mappings.pkl", 'rb') as f:
                mappings = pickle.load(f)
                self.author_id_to_names = mappings['author_id_to_names']
                self.name_to_author_ids = mappings['name_to_author_ids']
            
            with open(DATA_DIR / "keyword_index.pkl", 'rb') as f:
                self.keyword_index = pickle.load(f)
            
            with open(DATA_DIR / "abstract_keywords.pkl", 'rb') as f:
                self.abstract_keywords = pickle.load(f)
            
            print(f"Loaded {len(self.publications)} publications")
            print(f"Loaded {len(self.author_profiles)} author profiles")
            print(f"Loaded {len(self.keyword_index)} keywords")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Data files not found. Run 'python src/preprocess.py' first.\n{e}"
            )
    
    def get_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        total_name_variants = sum(len(names) for names in self.author_id_to_names.values())
        return {
            'publications': len(self.publications),
            'author_ids': len(self.author_profiles),
            'name_variants': total_name_variants,
            'keywords': len(self.keyword_index)
        }
    
    def _normalize_keywords(self, keywords_input: str) -> List[str]:
        """Normalize and split keyword input."""
        if not keywords_input:
            return []
        
        keywords = []
        # Split by comma and clean
        for kw in keywords_input.split(','):
            kw = kw.strip().lower()
            if kw:
                keywords.append(kw)
        
        return keywords
    
    def _search_keyword_exact(self, keyword: str) -> List[Tuple[str, float]]:
        """Exact match search in keyword index."""
        return self.keyword_index.get(keyword, [])
    
    def _search_keyword_partial(self, keyword: str) -> List[Tuple[str, float]]:
        """Partial/substring match in keyword index."""
        results = []
        keyword_lower = keyword.lower()
        
        for indexed_kw, pub_list in self.keyword_index.items():
            # Check if keyword is substring of indexed keyword or vice versa
            if keyword_lower in indexed_kw or indexed_kw in keyword_lower:
                # Lower score for partial matches
                for pub_id, score in pub_list:
                    results.append((pub_id, score * 0.7))
        
        return results
    
    def _search_abstract_fulltext(self, keyword: str) -> List[Tuple[str, float]]:
        """Full-text search in abstracts."""
        results = []
        keyword_lower = keyword.lower()
        
        for pub_id, pub in self.publications.items():
            abstract_lower = pub.get('abstract_lower', pub.get('abstract', '').lower())
            title_lower = pub.get('title', '').lower()
            
            # Count occurrences in abstract
            abstract_count = abstract_lower.count(keyword_lower)
            title_count = title_lower.count(keyword_lower)
            
            if abstract_count > 0 or title_count > 0:
                # Score based on frequency and location
                score = (abstract_count * 0.5) + (title_count * 1.5)
                results.append((pub_id, score))
        
        return results
    
    def search_by_keywords(self, keywords_input: str, max_results: int = 100) -> Dict[str, Any]:
        """
        Search publications by keywords with accurate matching.
        Returns results aggregated by Author ID.
        """
        keywords = self._normalize_keywords(keywords_input)
        
        if not keywords:
            return {
                'total': 0,
                'results': [],
                'keywords_used': [],
                'total_matching_pubs': 0,
                'document_type_dist': {}
            }
        
        # Collect publication scores
        pub_scores = defaultdict(float)
        pub_matched_keywords = defaultdict(set)
        
        for keyword in keywords:
            # 1. Exact match in keyword index (highest priority)
            exact_matches = self._search_keyword_exact(keyword)
            for pub_id, score in exact_matches:
                pub_scores[pub_id] += score * 2.0
                pub_matched_keywords[pub_id].add(keyword)
            
            # 2. Partial match in keyword index
            partial_matches = self._search_keyword_partial(keyword)
            for pub_id, score in partial_matches:
                if pub_id not in pub_matched_keywords or keyword not in pub_matched_keywords[pub_id]:
                    pub_scores[pub_id] += score
                    pub_matched_keywords[pub_id].add(keyword)
            
            # 3. Full-text search in abstracts (catches remaining)
            fulltext_matches = self._search_abstract_fulltext(keyword)
            for pub_id, score in fulltext_matches:
                if pub_id not in pub_matched_keywords or keyword not in pub_matched_keywords[pub_id]:
                    pub_scores[pub_id] += score * 0.5
                    pub_matched_keywords[pub_id].add(keyword)
        
        # Aggregate by author
        author_results = defaultdict(lambda: {
            'author_id': '',
            'name_variants': [],
            'matching_papers': 0,
            'total_score': 0.0,
            'pub_ids': [],
            'matched_keywords': set()
        })
        
        matching_pub_ids = set(pub_scores.keys())
        
        # Calculate document type distribution
        doc_type_dist = defaultdict(int)
        for pub_id in matching_pub_ids:
            pub = self.publications.get(pub_id)
            if pub:
                doc_type = pub.get('document_type', 'Unknown')
                doc_type_dist[doc_type] += 1
        
        for pub_id, score in pub_scores.items():
            pub = self.publications.get(pub_id)
            if not pub:
                continue
            
            for author_id in pub['author_ids']:
                if author_id in self.author_profiles:
                    profile = self.author_profiles[author_id]
                    author_results[author_id]['author_id'] = author_id
                    author_results[author_id]['name_variants'] = profile['name_variants']
                    author_results[author_id]['matching_papers'] += 1
                    author_results[author_id]['total_score'] += score
                    author_results[author_id]['pub_ids'].append(pub_id)
                    author_results[author_id]['matched_keywords'].update(pub_matched_keywords[pub_id])
        
        # Sort by score, then by matching papers
        sorted_authors = sorted(
            author_results.values(),
            key=lambda x: (x['total_score'], x['matching_papers']),
            reverse=True
        )
        
        # Clean up for output
        results = []
        for r in sorted_authors[:max_results]:
            results.append({
                'author_id': r['author_id'],
                'name_variants': r['name_variants'],
                'matching_papers': r['matching_papers'],
                'score': round(r['total_score'], 2),
                'pub_ids': r['pub_ids'][:10],  # Limit for performance
                'matched_keywords': list(r['matched_keywords'])
            })
        
        return {
            'total': len(results),
            'results': results,
            'keywords_used': keywords,
            'total_matching_pubs': len(matching_pub_ids),
            'document_type_dist': dict(doc_type_dist)
        }
    
    def search_by_skills(self, skills: List[str], max_results: int = 100) -> Dict[str, Any]:
        """
        Search by skills (similar to keywords but with skill-specific logic).
        """
        if not skills:
            return {'total': 0, 'results': [], 'skills_used': []}
        
        # Normalize skills
        normalized_skills = [s.strip().lower() for s in skills if s.strip()]
        
        # Use keyword search with skills
        keywords_str = ', '.join(normalized_skills)
        results = self.search_by_keywords(keywords_str, max_results)
        results['skills_used'] = normalized_skills
        
        return results
    
    def search_by_author_id(self, author_id: str) -> Optional[Dict[str, Any]]:
        """Direct lookup by Author ID."""
        author_id = str(author_id).strip()
        
        if author_id in self.author_profiles:
            return self.author_profiles[author_id]
        
        # Try with/without leading zeros or formatting
        for aid in self.author_profiles:
            if aid.strip() == author_id or aid.replace('.', '') == author_id.replace('.', ''):
                return self.author_profiles[aid]
        
        return None
    
    def get_author_profile(self, author_id: str) -> Optional[Dict[str, Any]]:
        """Alias for search_by_author_id."""
        return self.search_by_author_id(author_id)
    
    def search_by_author_name(self, name: str, max_results: int = 50) -> Dict[str, Any]:
        """
        Search by author name with accurate matching.
        Returns list of matching author profiles.
        """
        if not name:
            return {'total': 0, 'results': []}
        
        name_lower = name.strip().lower()
        # Only use significant parts (length >= 3 to avoid matching common parts like 'r')
        name_parts = set(re.split(r'[,\s\.]+', name_lower))
        name_parts = {p for p in name_parts if len(p) >= 3}
        
        # Scores for each author
        author_scores = defaultdict(int)
        
        # 1. Direct/exact lookup in name_to_author_ids (highest priority)
        if name_lower in self.name_to_author_ids:
            for aid in self.name_to_author_ids[name_lower]:
                author_scores[aid] += 100
        
        # 2. Search for significant parts only
        for part in name_parts:
            if len(part) >= 4:  # Only match parts of length 4+
                if part in self.name_to_author_ids:
                    for aid in self.name_to_author_ids[part]:
                        author_scores[aid] += 30
        
        # 3. Search in author profiles' name variants - precise matching
        for author_id, profile in self.author_profiles.items():
            for variant in profile.get('name_variants', []):
                variant_lower = variant.lower()
                
                # Exact match
                if name_lower == variant_lower:
                    author_scores[author_id] += 200
                # Full name is substring (e.g., searching "chandiramouli" finds "Chandiramouli, R.")
                elif name_lower in variant_lower:
                    # Make sure it's a word boundary match, not just substring
                    # Check if the query appears as a complete word/name part
                    variant_parts = set(re.split(r'[,\s\.]+', variant_lower))
                    if name_lower in variant_parts:
                        author_scores[author_id] += 80
                    elif any(name_lower in p and len(name_lower) >= len(p) * 0.7 for p in variant_parts):
                        author_scores[author_id] += 50
                    else:
                        author_scores[author_id] += 20
                # Variant is substring of search (e.g., searching "chandiramouli ramanathan" finds "Chandiramouli, R.")
                elif variant_lower in name_lower and len(variant_lower) >= 3:
                    author_scores[author_id] += 40
                else:
                    # Check if any significant part matches a name part
                    variant_parts = set(re.split(r'[,\s\.]+', variant_lower))
                    for part in name_parts:
                        if len(part) >= 4:  # Only significant parts
                            for vpart in variant_parts:
                                if part == vpart:
                                    author_scores[author_id] += 25
                                elif part in vpart and len(part) >= len(vpart) * 0.7:
                                    author_scores[author_id] += 15
        
        # Filter out low-scoring matches
        min_score = 15
        matching_author_ids = {aid for aid, score in author_scores.items() if score >= min_score}
        
        # Build results
        results = []
        for author_id in matching_author_ids:
            if author_id in self.author_profiles:
                profile = self.author_profiles[author_id]
                results.append({
                    'author_id': author_id,
                    'name_variants': profile.get('name_variants', []),
                    'pub_count': profile.get('pub_count', 0),
                    'total_citations': profile.get('total_citations', 0),
                    'match_score': author_scores[author_id]
                })
        
        # Sort by match score, then by pub_count
        results.sort(key=lambda x: (x['match_score'], x['pub_count']), reverse=True)
        
        return {
            'total': len(results),
            'results': results[:max_results],
            'query': name
        }
    
    def get_rag_context(self, skills: List[str], max_abstracts: int = 20) -> List[Dict[str, Any]]:
        """
        Get relevant abstracts for RAG analysis.
        Returns list of publication data with full abstracts.
        """
        if not skills:
            return []
        
        # Search for relevant publications
        search_results = self.search_by_keywords(', '.join(skills), max_results=50)
        
        if not search_results.get('results'):
            return []
        
        # Collect unique publications from top authors
        seen_pubs = set()
        context = []
        
        for author_result in search_results['results'][:20]:
            for pub_id in author_result.get('pub_ids', [])[:5]:
                if pub_id in seen_pubs:
                    continue
                seen_pubs.add(pub_id)
                
                pub = self.publications.get(pub_id)
                if pub and pub.get('abstract'):
                    # Get first author ID for this pub
                    first_author_id = pub['author_ids'][0] if pub['author_ids'] else 'Unknown'
                    
                    context.append({
                        'pub_id': pub_id,
                        'title': pub['title'],
                        'abstract': pub['abstract'],
                        'authors': pub['authors'],
                        'author_id': first_author_id,
                        'year': pub['year'],
                        'keywords': pub['author_keywords'][:10],
                        'citations': pub['citations']
                    })
                
                if len(context) >= max_abstracts:
                    break
            
            if len(context) >= max_abstracts:
                break
        
        # Sort by relevance (citations and recency)
        context.sort(key=lambda x: (x['citations'], x['year']), reverse=True)
        
        return context[:max_abstracts]
    
    # ============ NEW ANALYTICS METHODS ============
    
    def get_citation_histogram_data(self, author_id: str) -> Dict[str, Any]:
        """Get citation histogram data for an author."""
        profile = self.search_by_author_id(author_id)
        if not profile:
            return {'citation_list': [], 'bins': [], 'counts': []}
        
        citation_list = profile.get('citation_list', [])
        if not citation_list:
            return {'citation_list': [], 'bins': [], 'counts': []}
        
        # Create histogram bins
        max_cite = max(citation_list) if citation_list else 0
        
        # Define bins (0-10, 10-50, 50-100, 100-500, 500+)
        bins = [0, 10, 50, 100, 500, max_cite + 1]
        bin_labels = ['0-10', '10-50', '50-100', '100-500', '500+']
        counts = [0] * 5
        
        for cite in citation_list:
            if cite < 10:
                counts[0] += 1
            elif cite < 50:
                counts[1] += 1
            elif cite < 100:
                counts[2] += 1
            elif cite < 500:
                counts[3] += 1
            else:
                counts[4] += 1
        
        return {
            'citation_list': citation_list,
            'bins': bin_labels,
            'counts': counts,
            'total_pubs': len(citation_list),
            'avg_citations': sum(citation_list) / len(citation_list) if citation_list else 0,
            'max_citations': max_cite
        }
    
    def get_national_international_collab(self, author_id: str) -> Dict[str, Any]:
        """Get national vs international collaboration data for an author."""
        profile = self.search_by_author_id(author_id)
        if not profile:
            return {'national': 0, 'international': 0, 'total': 0, 'details': []}
        
        national_count = 0
        international_count = 0
        details = []
        
        for pub in profile.get('publications', []):
            countries = pub.get('countries', [])
            
            if not countries:
                # No country info - skip
                continue
            
            # Check if all countries are India
            india_only = all(c.lower() in ['india'] for c in countries)
            has_india = any(c.lower() in ['india'] for c in countries)
            
            if india_only or (has_india and len(countries) == 1):
                national_count += 1
                collab_type = 'National'
            elif has_india and len(countries) > 1:
                international_count += 1
                collab_type = 'International'
            else:
                # No India involvement
                international_count += 1
                collab_type = 'International'
            
            details.append({
                'title': pub['title'],
                'year': pub['year'],
                'countries': countries,
                'type': collab_type
            })
        
        return {
            'national': national_count,
            'international': international_count,
            'total': national_count + international_count,
            'details': details
        }
    
    def get_country_collaboration_data(self, author_id: str) -> Dict[str, Any]:
        """Get country-wise collaboration data for an author (India vs other countries)."""
        profile = self.search_by_author_id(author_id)
        if not profile:
            return {'india_collabs': {}, 'total_pubs_with_countries': 0}
        
        # Count collaborations between India and other countries
        india_collabs = defaultdict(int)
        total_pubs_with_countries = 0
        
        for pub in profile.get('publications', []):
            countries = pub.get('countries', [])
            
            if not countries:
                continue
            
            total_pubs_with_countries += 1
            
            # Check if India is involved
            has_india = any(c.lower() in ['india'] for c in countries)
            
            if has_india:
                # Count collaborations with other countries
                for country in countries:
                    if country.lower() != 'india':
                        india_collabs[country] += 1
        
        # Sort by collaboration count
        sorted_collabs = sorted(india_collabs.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'india_collabs': dict(sorted_collabs),
            'total_pubs_with_countries': total_pubs_with_countries,
            'top_countries': sorted_collabs[:10]  # Top 10 collaborating countries
        }
    
    # ============ EXISTING ANALYTICS METHODS (UNCHANGED) ============
    
    def get_top_keywords(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get top N keywords across all publications."""
        keyword_counts = defaultdict(int)
        
        for pub in self.publications.values():
            # Count author keywords (highest weight)
            for kw in pub.get('author_keywords', []):
                keyword_counts[kw] += 3
            
            # Count index keywords
            for kw in pub.get('index_keywords', []):
                keyword_counts[kw] += 2
        
        # Sort and return top N
        top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
        return top_keywords[:limit]
    
    def get_most_active_authors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top N authors by publication count."""
        author_data = []
        
        for author_id, profile in self.author_profiles.items():
            author_data.append({
                'author_id': author_id,
                'name': ', '.join(profile.get('name_variants', ['Unknown'])[:2]),
                'pub_count': profile.get('pub_count', 0),
                'total_citations': profile.get('total_citations', 0),
                'top_keywords': [k for k, _ in profile.get('top_keywords', [])[:3]]
            })
        
        # Sort by publication count
        author_data.sort(key=lambda x: x['pub_count'], reverse=True)
        return author_data[:limit]
    
    def get_keyword_trend_over_time(self, keyword: str) -> Dict[str, Any]:
        """Get year-wise publication count for a specific keyword."""
        keyword_lower = keyword.strip().lower()
        year_counts = defaultdict(int)
        matching_pubs = []
        
        for pub_id, pub in self.publications.items():
            # Check if keyword appears in author keywords, index keywords, or abstract
            found = False
            
            # Check author keywords
            for kw in pub.get('author_keywords', []):
                if keyword_lower in kw.lower():
                    found = True
                    break
            
            # Check index keywords
            if not found:
                for kw in pub.get('index_keywords', []):
                    if keyword_lower in kw.lower():
                        found = True
                        break
            
            # Check abstract
            if not found:
                abstract_lower = pub.get('abstract_lower', pub.get('abstract', '').lower())
                if keyword_lower in abstract_lower:
                    found = True
            
            # Check title
            if not found:
                title_lower = pub.get('title', '').lower()
                if keyword_lower in title_lower:
                    found = True
            
            if found:
                year = pub.get('year', 0)
                if year > 1900:
                    year_counts[year] += 1
                    matching_pubs.append(pub_id)
        
        sorted_years = sorted(year_counts.items())
        
        return {
            'keyword': keyword,
            'years': [y for y, _ in sorted_years],
            'counts': [c for _, c in sorted_years],
            'total_publications': len(matching_pubs),
            'data': dict(sorted_years)
        }


# Singleton instance
_engine_instance = None


def get_engine() -> SearchEngine:
    """Get or create search engine singleton."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = SearchEngine()
    return _engine_instance