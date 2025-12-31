# SASTRA Research Finder - Enhanced Version with Thematic Areas

Research discovery system using **Author ID as single source of truth** with **abstract-based matching**, **Mistral AI RAG**, and **Thematic Area Analysis**.

## 🎯 Core Principles

1. **Author ID = Single Source of Truth**
   - Every author maps to a unique Author ID from `Author(s) ID` column
   - Multiple name variants (Brindha, Brindha G.R., Brindha GR) → ONE Author ID
   - ALL searches aggregate by Author ID

2. **Abstract-Based Matching**
   - ALL searches use abstracts as primary field
   - Keywords extracted from abstracts boost matching
   - Bigram and trigram extraction for better relevance

3. **Dynamic Skill Extraction (NO Hardcoding)**
   - Skills extracted from project title
   - Enhanced with abstract keywords
   - NO predefined skills like "NLP", "CV", "ML"

4. **One-Click Expansion**
   - Click Author ID → Full profile
   - All name variants
   - All papers with full abstracts

5. **Enhanced Mistral AI RAG**
   - Ultra-accurate with improved prompting
   - Temperature: 0.05-0.15 for maximum accuracy
   - Comprehensive 6-section analysis
   - Author ID references in output

6. **🆕 Thematic Areas Analysis**
   - 30+ pre-defined thematic research domains
   - Single-theme faculty rankings by citation impact
   - Interdisciplinary team formation (2-3 themes)
   - Interactive scatter plot visualization
   - Real-time team building based on expertise overlap

## 📊 Database Stats

- **5,159** Publications
- **9,561** Unique Author IDs
- **27,169** Name Variants
- **195,420** Indexed Keywords/Phrases

## 📋 Features

### Phase 1: Keyword → Abstract Matching
```
Input: "machine learning, deep learning, classification"

Output Table:
| Author ID    | Author Name Variants        | Matching Papers |
|--------------|----------------------------|-----------------|
| 37061393600  | Ramachandran, R.            | 95              |
| 54888993500  | Vairavasundaram, V.         | 97              |
```

### Phase 2: Skill-Based Search
```
Input: "Deep learning based segmentation models for MRI analysis"

Extracted Skills (Dynamic):
☑ deep learning
☑ segmentation  
☑ mri
☑ neural networks
☑ medical imaging

Output: Same table format as Phase 1
```

### Phase 3: Author/ID Lookup
- **Search by Author ID**: Direct lookup
- **Search by Author Name**: Returns all matching Author IDs with scoring

### One-Click Expansion (Both Phases)
Clicking any Author ID shows:
- All name variants
- Total papers & citations
- All publications with **FULL ABSTRACTS**
- Research keywords

### 🆕 Thematic Areas Tab
```
Features:
✓ 30+ Research Themes across domains (CS, Healthcare, Engineering, etc.)
✓ Single Theme View:
  - Top 10 faculty by citation impact
  - Interactive scatter plot (citations vs papers)
  - Click on faculty → view papers & abstracts
  - Visual bubble size indicates impact

✓ Interdisciplinary Teams (2-3 themes):
  - **User enters research description** (e.g., "deep learning in medicine")
  - System automatically identifies relevant themes
  - Smart team formation by citation ranking
  - Team 1: Highest cite scores from each theme combined
  - Team 2: 2nd highest, and so on
  - Up to 10 balanced teams
  - Each member shows: theme, papers, abstracts
  - One-click profile access

Thematic Categories:
📚 Computer Science & AI (8 themes)
🏥 Healthcare & Medicine (4 themes)
⚙️ Engineering (5 themes)
🧪 Materials & Chemistry (3 themes)
🌱 Environmental & Sustainability (2 themes)
📊 Other Domains (6 themes)
```

### RAG Analysis (Enhanced Mistral AI)
```
Input: Project description or skills

Output: Ultra-detailed 6-section analysis
## 1. KEY METHODS & TECHNIQUES
- 8-12 specific techniques with implementation details
- Model architectures (ResNet-50, LSTM, etc.)
- Preprocessing and evaluation metrics

## 2. REPRESENTATIVE PAPERS
- 8-12 most relevant papers
- "Title" (AUTHOR_ID: XXXXX) - Detailed methodology
- Year and citation count

## 3. REQUIRED TECHNOLOGIES & TOOLS
- Specific versions (TensorFlow 2.x, PyTorch 1.x)
- Libraries, frameworks, tools
- Hardware requirements
- Datasets used

## 4. RECOMMENDED RESEARCHERS (Ranked)
- Author ID: XXXXX - Detailed expertise
- Relevance explanation
- Paper count in area

## 5. RESEARCH GAPS & OPPORTUNITIES
- Unexplored techniques
- Interdisciplinary opportunities
- Emerging trends

## 6. NEXT STEPS FOR COLLABORATION
- Priority reading list
- Researchers to contact
- Implementation roadmap
- Technical prerequisites
```

**Enhanced Features:**
- Temperature: 0.05-0.15 (ultra-accurate)
- 20 papers context (vs 15 before)
- 2500 tokens output (vs 2000)
- Comprehensive prompting
- Specific technical details
- No generic responses

## 🚀 Quick Start

### Windows
```batch
run.bat
```

### Manual Installation
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run preprocessing (REQUIRED first time)
python src/preprocess.py

# Start the app
streamlit run app.py
```

### If dependencies fail
```bash
python -m pip install --upgrade pip
pip install mistralai
pip install -U sentence-transformers
pip install faiss-cpu
```

## 📁 Structure

```
sastra-research-finder/
├── app.py                 # Streamlit UI
├── requirements.txt
├── run.bat
├── .env.example
├── README.md
├── data/
│   ├── SASTRA_Publications_2024-25.xlsx
│   ├── publications.pkl      (generated)
│   ├── author_profiles.pkl   (generated)
│   ├── mappings.pkl          (generated)
│   ├── keyword_index.pkl     (generated)
│   └── abstract_keywords.pkl (generated)
└── src/
    ├── __init__.py
    ├── preprocess.py      # Data preprocessing
    ├── search_engine.py   # Search logic
    └── mistral_rag.py     # RAG analysis with Mistral AI
```

## 🔧 Enable Mistral AI (Optional)

1. Get API key: https://console.mistral.ai/api-keys/
2. Create `.env`:
   ```
   MISTRAL_API_KEY=your_key_here
   ```
3. Restart app

## 📊 Data Flow

```
User Keywords → Lowercase → Multi-layer Search
                                    ↓
                    1. Exact keyword index match
                    2. Partial keyword match
                    3. Full-text abstract search
                                    ↓
                    Aggregate Results per Author ID
                                    ↓
    | Author ID | Name Variants | Matching Papers | Score |
                                    ↓
            One-Click → Full Profile + All Abstracts
```

## 🔍 Search Accuracy

The search engine uses multi-layer matching:
1. **Author Keywords** (weight: 3.0) - Highest priority
2. **Index Keywords** (weight: 2.0) - Second priority
3. **Title Keywords** (weight: 1.5) - Moderate priority
4. **Abstract Keywords** (weight: 1.0) - Base matching
5. **N-grams** (weight: 0.8) - Phrase matching

Author name search uses intelligent scoring:
- Exact match: 200 points
- Full name substring: 80 points
- Partial match: 25-40 points
- Minimum threshold: 15 points (filters false positives)

## 🤖 Mistral AI Models

The app uses `mistral-small-latest` by default for cost-effective usage.
Available models (can be changed in `src/mistral_rag.py`):
- `mistral-tiny` - Fastest, cheapest
- `mistral-small-latest` - Balanced (default)
- `mistral-medium-latest` - Better quality
- `mistral-large-latest` - Best quality

---
Built for SASTRA University | Mistral AI Compatible
