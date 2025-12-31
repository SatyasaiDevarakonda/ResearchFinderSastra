"""
SASTRA Research Finder - Enhanced Mistral AI RAG Module
Ultra-accurate with improved prompting and context handling
"""

import os
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

# Try to load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Try to import Mistral
try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False
    Mistral = None


class MistralRAG:
    """Enhanced RAG system with ultra-accurate analysis."""
    
    def __init__(self):
        """Initialize Mistral API."""
        self.api_key = self._get_api_key()
        self.client = None
        self.model = None
        self._initialized = False
        
        if MISTRAL_AVAILABLE and self.api_key:
            try:
                self.client = Mistral(api_key=self.api_key)
                # Use mistral-small-latest for balanced performance
                self.model = 'mistral-small-latest'
                self._initialized = True
                print(f"✓ Mistral AI initialized with model: {self.model}")
            except Exception as e:
                print(f"✗ Mistral initialization failed: {e}")
                self._initialized = False
        else:
            if not MISTRAL_AVAILABLE:
                print("✗ Mistral AI package not installed. Run: pip install mistralai")
            if not self.api_key:
                print("✗ No API key found")
    
    def _get_api_key(self) -> str:
        """Get API key from multiple sources."""
        # Priority 1: Streamlit secrets
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and 'MISTRAL_API_KEY' in st.secrets:
                print("✓ Mistral API key loaded from Streamlit secrets")
                return st.secrets['MISTRAL_API_KEY']
        except (ImportError, FileNotFoundError, KeyError, AttributeError):
            pass
        
        # Priority 2: Environment variable
        api_key = os.getenv('MISTRAL_API_KEY', '')
        if api_key:
            print("✓ Mistral API key loaded from environment variable")
            return api_key
        
        # Priority 3: .env file
        try:
            from dotenv import load_dotenv
            load_dotenv(override=True)
            api_key = os.getenv('MISTRAL_API_KEY', '')
            if api_key:
                print("✓ Mistral API key loaded from .env file")
                return api_key
        except ImportError:
            pass
        
        print("✗ No Mistral API key found")
        return ''
    
    def is_available(self) -> bool:
        """Check if Mistral API is available."""
        return self._initialized and self.client is not None
    
    def _call_mistral(self, prompt: str, max_tokens: int = 1500, temperature: float = 0.1) -> Optional[str]:
        """Make a call to Mistral API with error handling."""
        if not self.is_available():
            return None
        
        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert research analyst specializing in academic publication analysis. Provide accurate, detailed, and specific insights based on the provided research papers. Always include Author IDs when referencing researchers or papers."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            if response and response.choices:
                return response.choices[0].message.content
            return None
        except Exception as e:
            print(f"Mistral API error: {e}")
            return None
    
    def extract_skills(self, project_title: str, context_keywords: List[str] = None) -> List[str]:
        """
        Extract highly specific skills from project title using Mistral.
        Falls back to advanced rule-based extraction if unavailable.
        """
        if not project_title:
            return []
        
        if self.is_available():
            try:
                return self._extract_skills_mistral(project_title, context_keywords)
            except Exception as e:
                print(f"Mistral skill extraction failed: {e}")
                return self._extract_skills_fallback(project_title, context_keywords)
        else:
            return self._extract_skills_fallback(project_title, context_keywords)
    
    def _extract_skills_mistral(self, project_title: str, context_keywords: List[str] = None) -> List[str]:
        """Use Mistral for precise skill extraction."""
        context_str = ""
        if context_keywords:
            context_str = f"\n\nRelated research keywords at SASTRA: {', '.join(context_keywords[:40])}"
        
        prompt = f"""Extract 10-15 highly specific technical skills, methods, and technologies from this research project title.

Project Title: "{project_title}"{context_str}

EXTRACTION RULES:
1. Be EXTREMELY specific - extract exact techniques, not broad categories
   ✓ Good: "convolutional neural networks", "U-Net architecture", "transfer learning"
   ✗ Bad: "deep learning", "AI", "machine learning"

2. Include domain-specific applications
   ✓ Good: "medical image segmentation", "brain tumor detection", "MRI analysis"
   ✗ Bad: "image processing", "detection", "analysis"

3. Extract data types and modalities if mentioned
   Examples: "MRI", "CT scan", "EEG signals", "satellite imagery", "time series data"

4. Include specific algorithms and architectures
   Examples: "ResNet", "LSTM", "random forest", "gradient boosting", "k-means clustering"

5. Identify tools, frameworks, and platforms if implied
   Examples: "TensorFlow", "PyTorch", "GPU computing", "cloud computing"

6. Capture research methodologies
   Examples: "supervised learning", "few-shot learning", "active learning", "data augmentation"

Return ONLY a comma-separated list of skills. No numbering, no explanations, no markdown.
Skills:"""
        
        response = self._call_mistral(prompt, max_tokens=250, temperature=0.05)
        
        if response:
            skills_text = response.strip()
            skills_text = re.sub(r'^[:\s"\']+|[:\s"\']+$', '', skills_text)
            
            skills = []
            for skill in skills_text.split(','):
                skill = skill.strip().lower()
                skill = re.sub(r'^\d+[\.\)]\s*', '', skill)
                if skill and 3 <= len(skill) <= 60:
                    skills.append(skill)
            
            return skills[:15]
        
        return self._extract_skills_fallback(project_title, context_keywords)
    
    def _extract_skills_fallback(self, project_title: str, context_keywords: List[str] = None) -> List[str]:
        """Advanced rule-based skill extraction."""
        if not project_title:
            return []
        
        title_lower = project_title.lower()
        
        # Comprehensive technical patterns
        tech_patterns = [
            # Deep Learning & Neural Networks
            r'\b(deep learning|neural network|cnn|convolutional neural|rnn|recurrent neural|lstm|gru|transformer|attention mechanism|self-attention)\b',
            r'\b(bert|gpt|resnet|vgg|inception|efficientnet|mobilenet|densenet|alexnet|yolo|faster rcnn|mask rcnn)\b',
            r'\b(u-net|gan|generative adversarial|vae|variational autoencoder|autoencoder|encoder-decoder)\b',
            r'\b(transfer learning|few-shot learning|zero-shot|meta-learning|continual learning|federated learning)\b',
            
            # Machine Learning
            r'\b(machine learning|supervised learning|unsupervised learning|reinforcement learning|semi-supervised)\b',
            r'\b(classification|regression|clustering|dimensionality reduction|feature extraction|feature selection)\b',
            r'\b(random forest|decision tree|gradient boosting|xgboost|lightgbm|catboost|adaboost)\b',
            r'\b(svm|support vector machine|naive bayes|k-nearest neighbor|knn|logistic regression)\b',
            r'\b(ensemble learning|bagging|boosting|stacking|voting classifier)\b',
            
            # NLP & Text Mining
            r'\b(natural language processing|nlp|text mining|text analytics|sentiment analysis|opinion mining)\b',
            r'\b(named entity recognition|ner|pos tagging|part-of-speech|dependency parsing|constituency parsing)\b',
            r'\b(word embedding|word2vec|glove|fasttext|sentence embedding|document embedding)\b',
            r'\b(language model|text generation|machine translation|summarization|question answering)\b',
            r'\b(topic modeling|lda|latent dirichlet|tf-idf|bag of words|n-gram)\b',
            
            # Computer Vision
            r'\b(computer vision|image processing|image analysis|visual recognition|scene understanding)\b',
            r'\b(object detection|object recognition|image classification|semantic segmentation|instance segmentation)\b',
            r'\b(image segmentation|edge detection|feature detection|keypoint detection|corner detection)\b',
            r'\b(face recognition|facial recognition|face detection|emotion recognition|gesture recognition)\b',
            r'\b(optical character recognition|ocr|image captioning|visual question answering)\b',
            
            # Data Science & Analytics
            r'\b(data mining|data analysis|big data|data science|predictive analytics|prescriptive analytics)\b',
            r'\b(time series analysis|forecasting|trend analysis|seasonal decomposition|arima|prophet)\b',
            r'\b(anomaly detection|outlier detection|fraud detection|intrusion detection)\b',
            r'\b(recommendation system|collaborative filtering|content-based filtering|matrix factorization)\b',
            r'\b(a/b testing|hypothesis testing|statistical analysis|exploratory data analysis|eda)\b',
            
            # Medical & Healthcare
            r'\b(medical imaging|healthcare analytics|clinical decision support|disease diagnosis|drug discovery)\b',
            r'\b(mri|ct scan|x-ray|ultrasound|ecg|eeg|emg|mammography|histopathology)\b',
            r'\b(brain tumor|cancer detection|disease classification|medical image segmentation)\b',
            r'\b(bioinformatics|genomics|proteomics|gene expression|dna sequencing)\b',
            r'\b(telemedicine|remote monitoring|wearable sensors|health informatics)\b',
            
            # IoT & Edge Computing
            r'\b(internet of things|iot|edge computing|fog computing|smart city|smart home)\b',
            r'\b(sensor network|wireless sensor|sensor fusion|sensor data)\b',
            r'\b(real-time processing|stream processing|event detection|activity recognition)\b',
            r'\b(energy efficiency|power optimization|battery management|energy harvesting)\b',
            
            # Cybersecurity
            r'\b(cybersecurity|network security|information security|data security|privacy preservation)\b',
            r'\b(intrusion detection|intrusion prevention|malware detection|phishing detection)\b',
            r'\b(encryption|cryptography|blockchain|distributed ledger|smart contract)\b',
            r'\b(access control|authentication|authorization|identity management)\b',
            
            # Optimization & Algorithms
            r'\b(optimization|genetic algorithm|evolutionary algorithm|particle swarm|ant colony)\b',
            r'\b(simulated annealing|tabu search|hill climbing|branch and bound)\b',
            r'\b(linear programming|integer programming|constraint satisfaction|multi-objective optimization)\b',
            
            # Signal & Speech Processing
            r'\b(signal processing|speech recognition|speech synthesis|audio processing|sound classification)\b',
            r'\b(feature extraction|mfcc|spectrogram|fourier transform|wavelet transform)\b',
            
            # Robotics & Autonomous Systems
            r'\b(robotics|autonomous vehicle|self-driving|path planning|motion planning|slam)\b',
            r'\b(obstacle avoidance|navigation|localization|mapping|robot control)\b',
            
            # Cloud & Distributed Computing
            r'\b(cloud computing|distributed computing|parallel processing|mapreduce|spark|hadoop)\b',
            r'\b(containerization|docker|kubernetes|microservices|serverless)\b',
            
            # Tools & Frameworks (implied)
            r'\b(tensorflow|pytorch|keras|scikit-learn|opencv|nltk|spacy|pandas|numpy)\b',
            r'\b(gpu|cuda|tpu|distributed training|model compression|quantization)\b',
        ]
        
        skills = set()
        
        # Extract all pattern matches
        for pattern in tech_patterns:
            matches = re.findall(pattern, title_lower)
            for match in matches:
                if isinstance(match, tuple):
                    skills.update(m for m in match if m)
                else:
                    skills.add(match)
        
        # Extract meaningful bigrams and trigrams
        words = re.findall(r'\b[a-z]+\b', title_lower)
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            # Check if bigram is meaningful
            if any(tech_word in bigram for tech_word in [
                'neural', 'deep', 'machine', 'learning', 'network', 'detection', 'recognition',
                'classification', 'segmentation', 'analysis', 'processing', 'prediction',
                'optimization', 'based', 'driven', 'assisted', 'enhanced', 'automated'
            ]):
                skills.add(bigram)
        
        # Add context keywords if provided
        if context_keywords:
            for kw in context_keywords[:15]:
                if len(kw) >= 3:
                    skills.add(kw.lower())
        
        # Filter and clean
        final_skills = []
        stopwords = {'the', 'and', 'for', 'with', 'using', 'based', 'via', 'through', 'from'}
        for skill in skills:
            skill = skill.strip()
            if 3 <= len(skill) <= 60 and skill not in stopwords:
                final_skills.append(skill)
        
        return final_skills[:15]
    
    def analyze(self, context: List[Dict[str, Any]], skills: List[str]) -> Dict[str, Any]:
        """
        Generate ultra-accurate research analysis using enhanced RAG.
        """
        if not self.is_available():
            return {
                'analysis': None,
                'error': 'Mistral API not configured. Add MISTRAL_API_KEY to .env or Streamlit secrets.'
            }
        
        if not context:
            return {
                'analysis': None,
                'error': 'No relevant publications found.'
            }
        
        try:
            return self._generate_analysis(context, skills)
        except Exception as e:
            return {
                'analysis': None,
                'error': f'Analysis failed: {str(e)}'
            }
    
    def _generate_analysis(self, context: List[Dict[str, Any]], skills: List[str]) -> Dict[str, Any]:
        """Generate detailed, accurate analysis."""
        
        # Build rich context from publications
        context_parts = []
        for idx, pub in enumerate(context[:20], 1):  # Use more papers for better context
            author_id = pub.get('author_id', 'Unknown')
            abstract = pub.get('abstract', '')[:1000]  # More abstract content
            keywords = pub.get('keywords', [])
            author_keywords = pub.get('author_keywords', [])
            index_keywords = pub.get('index_keywords', [])
            
            all_keywords = list(set(keywords + author_keywords + index_keywords))[:10]
            keywords_str = ', '.join(all_keywords) if all_keywords else 'Not specified'
            
            context_parts.append(f"""
[Paper {idx}]
Title: {pub.get('title', 'Untitled')}
Author ID: {author_id}
Authors: {pub.get('authors', 'N/A')}
Year: {pub.get('year', 'N/A')}
Citations: {pub.get('citations', 0)}
Keywords: {keywords_str}
Abstract: {abstract}
""")
        
        context_text = "\n".join(context_parts)
        skills_text = ", ".join(skills)
        
        prompt = f"""You are an expert research analyst at SASTRA University. Analyze the provided research publications and generate a comprehensive, highly detailed report.

USER'S REQUIRED SKILLS/INTERESTS: {skills_text}

RELEVANT PUBLICATIONS FROM SASTRA RESEARCHERS:
{context_text}

Generate a DETAILED analysis in this EXACT format:

## 1. KEY METHODS & TECHNIQUES
Provide a comprehensive list of specific methods, algorithms, and techniques found in these papers.
- Be EXTREMELY specific (e.g., "U-Net architecture with attention gates for brain tumor segmentation in MRI images")
- Include implementation details where mentioned
- Mention specific model architectures (ResNet-50, LSTM with 256 units, etc.)
- Describe preprocessing and feature extraction techniques
- List evaluation metrics used (accuracy, F1-score, IoU, etc.)
- Minimum 8-12 specific techniques with details

## 2. REPRESENTATIVE PAPERS
List the most relevant papers with Author IDs for easy lookup:
- "Complete Paper Title" (AUTHOR_ID: XXXXX) - Detailed description of methodology, contributions, and key findings
- Include year and citation count if significant
- Explain why each paper is relevant to the user's requirements
- List 8-12 most relevant papers

## 3. REQUIRED TECHNOLOGIES & TOOLS
Be comprehensive and specific:
- Programming Languages: Python 3.x, R, MATLAB (specify versions if mentioned)
- Deep Learning Frameworks: TensorFlow 2.x, PyTorch 1.x, Keras (with specific versions)
- Libraries: NumPy, pandas, scikit-learn, OpenCV, NLTK, spaCy (list all mentioned)
- Pre-trained Models: ResNet-50, VGG-16, BERT-base, etc.
- Hardware: GPU requirements (NVIDIA Tesla V100, etc.), RAM, storage
- Development Tools: Jupyter notebooks, Git, Docker
- Cloud Platforms: AWS, GCP, Azure (if mentioned)
- Datasets used in the research

## 4. RECOMMENDED RESEARCHERS (Ranked by Relevance)
Based on publication count, citation impact, and relevance to user's requirements:
1. Author ID: XXXXX - Dr. [Name if known]
   Expertise: Detailed description of their research focus
   Relevant Papers: [Count] papers in this area
   Why Recommended: Specific reasons based on their work
   
2. [Continue for 5-8 top researchers]

## 5. RESEARCH GAPS & OPPORTUNITIES
Identify areas not well covered in current SASTRA research:
- Unexplored techniques or methods
- Potential interdisciplinary opportunities
- Emerging trends in the field
- Specific improvements that could be made

## 6. NEXT STEPS FOR COLLABORATION/IMPLEMENTATION
Provide actionable, specific recommendations:
- Which papers to read first (in order of priority)
- Which researchers to contact for collaboration
- Suggested starting points for implementation
- Key challenges to be aware of
- Estimated timeline and resources needed
- Specific technical prerequisites

CRITICAL REQUIREMENTS:
- Always include Author IDs when referencing papers or researchers
- Be SPECIFIC and TECHNICAL, never generic
- Base ALL information on the provided publications
- Do NOT invent or assume information not in the context
- Use exact numbers, metrics, and technical terms from the papers
- If specific details aren't available, clearly state "Not specified in available papers"
- Provide enough detail that a researcher could follow up immediately"""

        response = self._call_mistral(prompt, max_tokens=2500, temperature=0.15)
        
        if response:
            return {
                'analysis': response,
                'error': None
            }
        else:
            return {
                'analysis': None,
                'error': 'Empty response from Mistral API'
            }
    
    def summarize_author(self, profile: Dict[str, Any]) -> str:
        """Generate detailed author research summary."""
        if not self.is_available():
            return ""
        
        try:
            pubs = profile.get('publications', [])[:15]
            keywords = profile.get('top_keywords', [])[:20]
            
            if not pubs:
                return ""
            
            pub_details = []
            for p in pubs:
                title = p.get('title', 'Untitled')
                year = p.get('year', 'N/A')
                cites = p.get('citations', 0)
                pub_details.append(f"- ({year}, {cites} citations) {title}")
            
            pub_text = "\n".join(pub_details)
            kw_text = ", ".join([f"{k} ({c})" for k, c in keywords]) if keywords else "Not available"
            
            prompt = f"""Summarize this SASTRA researcher's expertise and contributions in 3-4 detailed sentences.

Researcher: {', '.join(profile.get('name_variants', [])[:2])}
Total Publications: {profile.get('pub_count', 0)}
Total Citations: {profile.get('total_citations', 0)}

Top Research Keywords: {kw_text}

Recent Publications:
{pub_text}

Write a detailed summary that:
1. Identifies their main research areas and expertise
2. Highlights their key contributions and methodologies
3. Mentions specific techniques or domains they work in
4. Notes any significant impact (high-cited papers, trends)

Summary (3-4 sentences):"""
            
            response = self._call_mistral(prompt, max_tokens=200, temperature=0.2)
            
            if response:
                return response.strip()
        except Exception as e:
            print(f"Author summary failed: {e}")
        
        return ""


# Singleton instance
_rag_instance = None


def get_rag() -> MistralRAG:
    """Get or create RAG singleton."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = MistralRAG()
    return _rag_instance
