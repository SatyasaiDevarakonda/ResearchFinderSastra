"""
SASTRA Research Finder - Thematic Areas Module
Complete Implementation: 50 Single Domains + 150 Interdisciplinary Combinations
Research areas selected based on SASTRA's engineering, science, and technology focus
"""

import pickle
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Optional
import re

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"


class ThematicAreasEngine:
    """Engine for thematic area analysis with 50 research domains."""
    
    # 50 THEMATIC AREAS - Selected based on:
    # 1. Core CS/IT areas (AI, ML, Data Science, Networks, Security)
    # 2. Electrical & Electronics (Power, Control, Communications, Signal Processing)
    # 3. Biomedical & Healthcare (Medical Imaging, Healthcare IT, Drug Discovery)
    # 4. Materials & Chemical Engineering (Nanomaterials, Chemical Processing)
    # 5. Environmental & Sustainable Tech (Renewable Energy, Environmental Engineering)
    # 6. Emerging Technologies (Quantum Computing, Blockchain, Edge AI)
    # 7. Applied Mathematics & Operations Research
    # 8. Manufacturing & Industrial Engineering
    
    THEMATIC_AREAS = {
        # ===== CORE AI & MACHINE LEARNING (8 areas) =====
        'Machine Learning': [
            'machine learning', 'ml', 'supervised learning', 'unsupervised learning', 
            'classification', 'regression', 'clustering', 'random forest', 'svm', 
            'support vector machine', 'decision tree', 'xgboost', 'ensemble learning',
            'semi-supervised', 'active learning'
        ],
        'Deep Learning': [
            'deep learning', 'neural network', 'cnn', 'convolutional neural', 'rnn', 
            'recurrent neural', 'lstm', 'gru', 'transformer', 'attention mechanism', 
            'bert', 'gpt', 'resnet', 'vgg', 'u-net', 'unet', 'gan', 'autoencoder',
            'generative adversarial', 'variational autoencoder'
        ],
        'Reinforcement Learning': [
            'reinforcement learning', 'q-learning', 'deep q-network', 'dqn', 'policy gradient',
            'actor-critic', 'reward function', 'markov decision', 'temporal difference',
            'multi-agent learning', 'deep reinforcement'
        ],
        'Transfer Learning': [
            'transfer learning', 'domain adaptation', 'pre-trained model', 'fine-tuning',
            'knowledge transfer', 'zero-shot learning', 'few-shot learning', 'meta-learning'
        ],
        'Explainable AI': [
            'explainable ai', 'interpretable machine learning', 'xai', 'model interpretability',
            'feature importance', 'lime', 'shap', 'attention visualization', 'explainability'
        ],
        'Federated Learning': [
            'federated learning', 'distributed learning', 'privacy-preserving ml',
            'edge learning', 'decentralized learning', 'collaborative learning'
        ],
        'Computer Vision': [
            'computer vision', 'image processing', 'object detection', 'image classification', 
            'segmentation', 'face recognition', 'ocr', 'optical character', 'yolo', 
            'image analysis', 'feature detection', 'edge detection', 'semantic segmentation',
            'instance segmentation', 'image enhancement', 'super resolution'
        ],
        'Natural Language Processing': [
            'nlp', 'natural language processing', 'text mining', 'sentiment analysis', 
            'text classification', 'named entity recognition', 'ner', 'language model', 
            'word embedding', 'text analytics', 'topic modeling', 'machine translation',
            'question answering', 'text generation', 'chatbot'
        ],
        
        # ===== DATA SCIENCE & ANALYTICS (5 areas) =====
        'Data Science & Analytics': [
            'data mining', 'data analysis', 'big data', 'analytics', 'data science', 
            'predictive analytics', 'business intelligence', 'data visualization', 
            'statistical analysis', 'pattern recognition', 'exploratory data analysis'
        ],
        'Time Series Analysis': [
            'time series', 'forecasting', 'arima', 'seasonal', 'trend analysis', 
            'temporal', 'prediction', 'timeseries', 'lstm forecasting', 'prophet',
            'time series decomposition'
        ],
        'Recommendation Systems': [
            'recommendation', 'recommender system', 'collaborative filtering', 
            'content-based filtering', 'matrix factorization', 'personalization',
            'hybrid recommendation', 'context-aware recommendation'
        ],
        'Social Network Analysis': [
            'social network', 'network analysis', 'graph theory', 'community detection',
            'influence', 'social media', 'network science', 'link prediction',
            'centrality measures', 'graph mining'
        ],
        'Information Retrieval': [
            'information retrieval', 'search engine', 'ranking', 'relevance',
            'document retrieval', 'query processing', 'indexing', 'web search'
        ],
        
        # ===== CYBERSECURITY & NETWORKING (5 areas) =====
        'Cybersecurity': [
            'cybersecurity', 'network security', 'intrusion detection', 'malware', 
            'encryption', 'cryptography', 'blockchain', 'security', 'firewall', 
            'threat detection', 'vulnerability', 'penetration testing', 'security audit'
        ],
        'Privacy & Data Protection': [
            'privacy', 'data privacy', 'differential privacy', 'privacy-preserving',
            'anonymization', 'gdpr', 'data protection', 'secure multiparty computation'
        ],
        'Blockchain & Distributed Ledger': [
            'blockchain', 'distributed ledger', 'smart contract', 'cryptocurrency',
            'consensus mechanism', 'ethereum', 'bitcoin', 'decentralized application'
        ],
        'Network Science': [
            'network topology', 'network optimization', 'routing protocol', 'network performance',
            'software defined networking', 'sdn', 'network function virtualization', 'nfv'
        ],
        'Wireless Communications': [
            'wireless', '5g', '6g', 'antenna', 'rf', 'communication system', 'wireless network',
            'mobile communication', 'spectrum', 'modulation', 'mimo', 'transmission',
            'cellular network', 'wifi', 'lora'
        ],
        
        # ===== CLOUD & EDGE COMPUTING (4 areas) =====
        'Cloud Computing': [
            'cloud computing', 'distributed computing', 'virtualization', 'kubernetes', 
            'docker', 'containerization', 'microservices', 'serverless', 'cloud infrastructure',
            'cloud storage', 'cloud security', 'multi-cloud'
        ],
        'Edge Computing': [
            'edge computing', 'fog computing', 'mobile edge computing', 'mec',
            'edge ai', 'edge analytics', 'edge intelligence', 'cloudlet'
        ],
        'Internet of Things': [
            'iot', 'internet of things', 'sensor network', 'smart city', 'smart home', 
            'embedded system', 'wireless sensor', 'sensor data', 'smart device',
            'industrial iot', 'iiot', 'wearable', 'smart agriculture'
        ],
        'Mobile Computing': [
            'mobile computing', 'mobile application', 'android', 'ios', 'mobile security',
            'mobile network', 'context-aware computing', 'location-based service'
        ],
        
        # ===== BIOMEDICAL & HEALTHCARE (7 areas) =====
        'Medical Imaging': [
            'medical imaging', 'mri', 'ct scan', 'ct-scan', 'x-ray', 'xray', 'ultrasound', 
            'mammography', 'radiology', 'radiography', 'medical image', 'dicom', 
            'brain tumor', 'tumor detection', 'cancer detection', 'lesion detection',
            'pathology', 'histopathology', 'microscopy', 'biopsy', 'lung cancer', 
            'breast cancer', 'skin cancer', 'brain mri', 'chest x-ray', 'chest xray',
            'retinal imaging', 'fundus', 'ecg', 'eeg', 'medical scan', 'tomography'
        ],
        'Healthcare Analytics': [
            'healthcare', 'clinical', 'disease diagnosis', 'medical diagnosis', 
            'patient monitoring', 'electronic health', 'health informatics', 'telemedicine', 
            'medical data', 'clinical decision', 'diagnosis', 'disease classification',
            'disease prediction', 'patient care', 'clinical trial', 'health monitoring', 
            'vital signs', 'medical records', 'ehr', 'emr', 'healthcare system', 
            'hospital', 'medical treatment', 'therapy', 'prognosis', 'survival prediction'
        ],
        'Bioinformatics': [
            'bioinformatics', 'genomics', 'proteomics', 'gene expression', 'dna sequencing',
            'computational biology', 'biomedical', 'molecular biology', 'genetic', 
            'genome', 'protein', 'sequence analysis', 'rna', 'gene', 'transcriptomics',
            'metagenomics', 'systems biology'
        ],
        'Drug Discovery': [
            'drug discovery', 'pharmaceutical', 'drug design', 'molecular docking', 
            'qsar', 'cheminformatics', 'drug development', 'pharmacology', 
            'drug screening', 'compound', 'molecule', 'inhibitor', 'ligand',
            'virtual screening', 'lead optimization'
        ],
        'Biomedicine': [
            'biomedicine', 'biomedical engineering', 'biomedical', 'medical device', 
            'prosthetics', 'implant', 'tissue engineering', 'regenerative medicine',
            'biocompatibility', 'medical technology', 'clinical engineering',
            'medical instrumentation', 'biosensor', 'bioelectronics', 'biomedical science'
        ],
        'Telemedicine & Digital Health': [
            'telemedicine', 'telehealth', 'remote monitoring', 'mobile health', 'mhealth',
            'digital health', 'health app', 'wearable health', 'remote diagnosis'
        ],
        'Precision Medicine': [
            'precision medicine', 'personalized medicine', 'targeted therapy',
            'pharmacogenomics', 'biomarker', 'patient stratification'
        ],
        
        # ===== ROBOTICS & AUTOMATION (3 areas) =====
        'Robotics & Automation': [
            'robotics', 'autonomous', 'robot control', 'path planning', 'slam', 
            'navigation', 'automation', 'industrial automation', 'manipulator', 
            'mobile robot', 'motion control', 'humanoid robot', 'swarm robotics'
        ],
        'Autonomous Systems': [
            'autonomous vehicle', 'self-driving', 'autonomous navigation', 'driverless',
            'autonomous drone', 'uav', 'unmanned aerial vehicle', 'autonomous ship'
        ],
        'Human-Robot Interaction': [
            'human-robot interaction', 'hri', 'collaborative robot', 'cobot',
            'social robotics', 'assistive robotics', 'robot learning from demonstration'
        ],
        
        # ===== SIGNAL & IMAGE PROCESSING (2 areas) =====
        'Signal Processing': [
            'signal processing', 'speech recognition', 'audio processing', 'speech synthesis',
            'sound classification', 'mfcc', 'fourier transform', 'wavelet', 
            'signal analysis', 'frequency analysis', 'digital signal processing', 'dsp'
        ],
        'Speech & Audio': [
            'speech processing', 'speaker recognition', 'speech enhancement', 'voice conversion',
            'automatic speech recognition', 'asr', 'speech emotion', 'audio analysis'
        ],
        
        # ===== ELECTRICAL & ELECTRONICS (4 areas) =====
        'Power Systems': [
            'power system', 'renewable energy', 'smart grid', 'energy management', 
            'solar', 'wind energy', 'electrical network', 'power quality', 'grid', 
            'electricity', 'power generation', 'power distribution', 'microgrid'
        ],
        'Control Systems': [
            'control system', 'pid controller', 'adaptive control', 'optimal control', 
            'fuzzy control', 'nonlinear control', 'robust control', 'feedback control',
            'model predictive control', 'mpc'
        ],
        'VLSI & Embedded Systems': [
            'vlsi', 'embedded system', 'fpga', 'asic', 'system on chip', 'soc',
            'microcontroller', 'embedded software', 'real-time systems', 'hardware design'
        ],
        'Optical Systems': [
            'optical', 'photonics', 'fiber optic', 'laser', 'optical communication',
            'optical sensor', 'optical network', 'ofdm'
        ],
        
        # ===== MATERIALS & CHEMICAL (4 areas) =====
        'Materials Science': [
            'materials', 'material science', 'nanomaterial', 'composite', 'polymer', 
            'ceramic', 'metallurgy', 'material characterization', 'thin film', 'coating',
            'biomaterial', 'smart material'
        ],
        'Nanotechnology': [
            'nanotechnology', 'nanoparticle', 'nanoscale', 'quantum dot', 'carbon nanotube',
            'graphene', 'nanofabrication', 'nanocomposite', 'nanomaterial', 'nanostructure'
        ],
        'Chemical Engineering': [
            'chemical engineering', 'process optimization', 'reaction kinetics', 'catalysis',
            'bioprocess', 'separation', 'distillation', 'mass transfer', 'heat transfer',
            'chemical reactor'
        ],
        'Polymer Science': [
            'polymer', 'polymerization', 'polymer composite', 'polymer processing',
            'polymer characterization', 'biodegradable polymer', 'conducting polymer'
        ],
        
        # ===== ENVIRONMENTAL & ENERGY (3 areas) =====
        'Environmental Engineering': [
            'environmental', 'environmental engineering', 'wastewater', 'water treatment', 
            'pollution', 'waste management', 'air quality', 'sustainability', 
            'green technology', 'ecosystem', 'environmental monitoring'
        ],
        'Renewable Energy': [
            'renewable energy', 'solar energy', 'wind energy', 'biofuel', 'biomass', 
            'sustainable energy', 'energy storage', 'battery', 'fuel cell', 'photovoltaic',
            'solar cell', 'wind turbine', 'hydroelectric'
        ],
        'Energy Efficiency': [
            'energy efficiency', 'energy optimization', 'energy conservation',
            'building energy', 'hvac optimization', 'energy audit', 'green building'
        ],
        
        # ===== OPERATIONS RESEARCH & OPTIMIZATION (2 areas) =====
        'Operations Research': [
            'optimization', 'linear programming', 'scheduling', 'inventory', 'supply chain',
            'operations research', 'decision making', 'simulation', 'algorithm',
            'integer programming', 'constraint programming'
        ],
        'Optimization Algorithms': [
            'genetic algorithm', 'particle swarm', 'ant colony', 'evolutionary',
            'metaheuristic', 'simulated annealing', 'optimization algorithm',
            'differential evolution', 'harmony search', 'artificial bee colony'
        ],
        
        # ===== BUSINESS & FINANCE (2 areas) =====
        'Financial Analytics': [
            'finance', 'financial', 'stock market', 'portfolio', 'risk management', 
            'credit', 'banking', 'financial forecasting', 'trading', 'algorithmic trading',
            'fraud detection', 'credit scoring'
        ],
        'Supply Chain Analytics': [
            'supply chain', 'logistics', 'inventory management', 'demand forecasting',
            'warehouse optimization', 'distribution network', 'supply chain risk'
        ],
        
        # ===== EMERGING TECHNOLOGIES (3 areas) =====
        'Quantum Computing': [
            'quantum computing', 'quantum algorithm', 'quantum machine learning',
            'quantum cryptography', 'qubit', 'quantum entanglement', 'quantum annealing'
        ],
        'Augmented Reality': [
            'augmented reality', 'ar', 'mixed reality', 'mr', 'virtual reality', 'vr',
            'immersive', '3d visualization', 'hologram'
        ],
        'Digital Twin': [
            'digital twin', 'virtual model', 'simulation model', 'cyber-physical system',
            'industry 4.0', 'predictive maintenance'
        ],
    }
    
    def __init__(self, publications: Dict, author_profiles: Dict):
        """Initialize the Thematic Areas Engine."""
        self.publications = publications
        self.author_profiles = author_profiles
        self.thematic_cache = {}
        
    def identify_thematic_areas(self, pub: Dict[str, Any]) -> List[str]:
        """Identify thematic areas for a given publication."""
        pub_id = pub.get('pub_id', '')
        if pub_id in self.thematic_cache:
            return self.thematic_cache[pub_id]
        
        text_parts = [pub.get('title', ''), pub.get('abstract', '')]
        
        author_kw = pub.get('author_keywords', [])
        if isinstance(author_kw, list):
            text_parts.extend(author_kw)
        
        index_kw = pub.get('index_keywords', [])
        if isinstance(index_kw, list):
            text_parts.extend(index_kw)
        
        combined_text = ' '.join(str(t) for t in text_parts).lower()
        
        area_scores = defaultdict(float)
        for area_name, keywords in self.THEMATIC_AREAS.items():
            score = 0
            for keyword in keywords:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, combined_text):
                    score += 2.0
                elif keyword in combined_text:
                    score += 1.0
            
            if score > 0:
                area_scores[area_name] = score
        
        sorted_areas = sorted(area_scores.items(), key=lambda x: x[1], reverse=True)
        matching_areas = [area for area, score in sorted_areas if score >= 1.5]
        
        self.thematic_cache[pub_id] = matching_areas
        return matching_areas
    
    def get_single_theme_rankings(self) -> Dict[str, List[Dict[str, Any]]]:
        """Compute rankings of authors for each single thematic area."""
        author_theme_data = defaultdict(lambda: {
            'author_id': '', 'name_variants': [], 'total_cite_score': 0,
            'paper_count': 0, 'papers': [], 'themes': set()
        })
        
        for pub_id, pub in self.publications.items():
            themes = self.identify_thematic_areas(pub)
            cite_score = pub.get('citations', 0)
            
            author_ids = pub.get('author_ids', [])
            if isinstance(author_ids, str):
                author_ids = [aid.strip() for aid in author_ids.split(';') if aid.strip()]
            elif not isinstance(author_ids, list):
                author_ids = []
            
            for author_id in author_ids:
                if author_id in self.author_profiles:
                    profile = self.author_profiles[author_id]
                    author_data = author_theme_data[author_id]
                    
                    if not author_data['author_id']:
                        author_data['author_id'] = author_id
                        author_data['name_variants'] = profile.get('name_variants', [author_id])
                    
                    author_data['total_cite_score'] += cite_score
                    author_data['paper_count'] += 1
                    author_data['papers'].append({
                        'title': pub.get('title', ''),
                        'year': pub.get('year', ''),
                        'citations': cite_score,
                        'abstract': pub.get('abstract', '')
                    })
                    author_data['themes'].update(themes)
        
        theme_rankings = {}
        for theme_name in self.THEMATIC_AREAS.keys():
            theme_authors = []
            
            for author_id, data in author_theme_data.items():
                if theme_name in data['themes']:
                    theme_authors.append({
                        'author_id': data['author_id'],
                        'name_variants': data['name_variants'],
                        'total_cite_score': data['total_cite_score'],
                        'paper_count': data['paper_count'],
                        'papers': sorted(data['papers'], key=lambda x: x['citations'], reverse=True),
                        'primary_name': data['name_variants'][0] if data['name_variants'] else 'Unknown'
                    })
            
            theme_authors.sort(key=lambda x: x['total_cite_score'], reverse=True)
            theme_rankings[theme_name] = theme_authors[:10]
        
        return theme_rankings
    
    def compute_single_theme_rankings(self) -> Dict[str, List[Dict[str, Any]]]:
        """Alias for backward compatibility."""
        return self.get_single_theme_rankings()


# ============================================================================
# Module-level functions
# ============================================================================

def load_thematic_engine() -> ThematicAreasEngine:
    """Load the thematic engine with publications and author profiles."""
    try:
        with open(DATA_DIR / "publications.pkl", 'rb') as f:
            publications = pickle.load(f)
        
        with open(DATA_DIR / "author_profiles.pkl", 'rb') as f:
            author_profiles = pickle.load(f)
        
        return ThematicAreasEngine(publications, author_profiles)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required data files not found in {DATA_DIR}: {e}")


def load_single_theme_rankings() -> Dict[str, List[Dict[str, Any]]]:
    """Load pre-computed Phase 1 single theme rankings from cache."""
    try:
        with open(DATA_DIR / "thematic_single_rankings.pkl", 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Warning: Pre-computed rankings not found at {DATA_DIR / 'thematic_single_rankings.pkl'}")
        return {}


def load_interdisciplinary_combinations() -> Dict[str, Dict]:
    """Load pre-computed Phase 2 interdisciplinary team combinations from cache."""
    try:
        with open(DATA_DIR / "thematic_combinations.pkl", 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Warning: Pre-computed combinations not found at {DATA_DIR / 'thematic_combinations.pkl'}")
        return {}


_thematic_engine: Optional[ThematicAreasEngine] = None

def get_thematic_engine() -> ThematicAreasEngine:
    """Get or create singleton instance of ThematicAreasEngine."""
    global _thematic_engine
    if _thematic_engine is None:
        _thematic_engine = load_thematic_engine()
    return _thematic_engine


def get_theme_names() -> List[str]:
    """Get list of all available thematic area names."""
    return list(ThematicAreasEngine.THEMATIC_AREAS.keys())


def get_theme_keywords(theme_name: str) -> List[str]:
    """Get keywords for a specific theme."""
    return ThematicAreasEngine.THEMATIC_AREAS.get(theme_name, [])