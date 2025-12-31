# CHANGELOG - SASTRA Research Finder Enhanced Version

## Version 2.0 - Enhanced Edition (December 2024)

### 🎨 NEW FEATURE: Thematic Areas Analysis

#### Single Thematic Area View
- **30+ Pre-defined Research Themes** across multiple domains:
  - Computer Science & AI (8 themes)
  - Healthcare & Medicine (4 themes)
  - Engineering (5 themes)
  - Materials & Chemistry (3 themes)
  - Environmental & Sustainability (2 themes)
  - Other Domains (6 themes)

- **Faculty Rankings by Citation Impact**
  - Top 10 faculty per theme
  - Sorted by total citation score in descending order
  - Shows cite score next to each faculty member
  - Paper count and detailed publication list

- **Interactive Scatter Plot Visualization**
  - X-axis: Number of papers
  - Y-axis: Total citations
  - Bubble size represents citation impact
  - Hover to see faculty details
  - Click-through to faculty profiles

- **Detailed Faculty Information**
  - Author ID and name variants
  - Total citations and paper count in theme
  - Top 5 papers with abstracts
  - One-click access to full profile

#### Interdisciplinary Team Formation
- **2-Theme Combinations**
  - Automatically identifies all valid 2-theme pairs
  - Forms balanced teams by matching citation ranks
  - Team 1: Highest cite score from each theme
  - Team 2: 2nd highest, continuing up to 10 teams

- **3-Theme Combinations**
  - Advanced team formation with 3 expertise areas
  - Smart matching algorithm for balanced teams
  - Only shows combinations with sufficient faculty

- **Team Details**
  - Total team citation score
  - Average citation score per member
  - Individual member contributions
  - Theme-specific papers for each member
  - Full abstracts available per paper
  - One-click profile access for each member

### 🚀 ENHANCED: Mistral AI RAG Analysis

#### Improved Prompting System
- **System Role Definition**: Expert research analyst persona
- **Lower Temperature**: 0.05-0.15 (vs 0.1-0.2) for maximum accuracy
- **Expanded Context**: 20 papers (vs 15) with 1000-char abstracts (vs 800)
- **Longer Output**: 2500 tokens (vs 2000) for comprehensive analysis

#### Enhanced Output Structure (6 Sections vs 5)
1. **KEY METHODS & TECHNIQUES** - Now requires 8-12 specific techniques with implementation details
2. **REPRESENTATIVE PAPERS** - 8-12 papers (vs 5-8) with detailed methodology descriptions
3. **REQUIRED TECHNOLOGIES & TOOLS** - Now includes specific versions, datasets, and hardware requirements
4. **RECOMMENDED RESEARCHERS** - Ranked list with detailed expertise descriptions
5. **RESEARCH GAPS & OPPORTUNITIES** - NEW SECTION for unexplored areas
6. **NEXT STEPS FOR COLLABORATION** - Enhanced with priority reading lists and implementation roadmap

#### Improved Skill Extraction
- **Comprehensive Pattern Matching**: 100+ technical patterns across all domains
- **Multi-word Phrase Detection**: Improved bigram and trigram extraction
- **Context Integration**: Better use of related keywords
- **Fallback System**: Advanced rule-based extraction when API unavailable

#### Enhanced Author Summaries
- 3-4 detailed sentences (vs 2-3)
- Includes top 15 keywords (vs 10)
- Analyzes top 15 publications (vs 10)
- Mentions specific techniques and impact

### 🎯 Thematic Area Engine Features

#### Intelligent Theme Identification
- Multi-source text analysis (title, abstract, keywords)
- Weighted scoring system
- Caching for performance
- Configurable threshold (score > 2.0)

#### Smart Team Building Algorithm
- Citation-based ranking
- Prevents duplicate members across teams
- Ensures representation from all themes
- Handles varying faculty availability
- Balances team composition

#### Visualization System
- Plotly integration for interactive charts
- Dynamic bubble sizing based on impact
- Color gradient for citation scale
- Hover information with full details
- Responsive design

### 📊 Technical Improvements

#### New Dependencies
- `plotly==5.18.0` - Interactive visualizations

#### New Module: thematic_areas.py
- `ThematicAreasEngine` class
- 30+ pre-defined thematic area definitions
- `identify_thematic_areas()` method
- `get_single_theme_rankings()` method
- `get_combined_theme_teams()` method
- `get_scatter_plot_data()` method
- Singleton pattern for efficiency

#### Enhanced App Structure
- New tab6: Thematic Areas
- Sub-tabs for single vs combined themes
- Category-based theme organization
- Session state management for selections
- Expandable faculty cards
- Integrated profile navigation

### 🐛 Maintained Features

All existing features remain fully functional:
- Author/ID Lookup with citation histograms
- Keyword Search with multi-layer matching
- Skill-Based Search with dynamic extraction
- Original RAG Analysis (now enhanced)
- Analytics Dashboard with trends

### 📚 Documentation Updates

- Enhanced README with new features
- Updated feature descriptions
- Added thematic areas examples
- Improved quick start guide
- Added CHANGELOG.md

### 🔧 Code Quality

- Type hints maintained
- Comprehensive error handling
- Performance optimizations
- Caching strategies
- Modular design
- Singleton patterns

---

## Version 1.0 - Initial Release

### Core Features
- Author ID-based system
- Keyword search
- Skill-based search
- Mistral AI RAG
- Analytics dashboard
- Author profiles with collaboration analysis

### Data Processing
- Publication indexing
- Author profile aggregation
- Keyword extraction
- Abstract processing
- Citation tracking

---

**Note**: This enhanced version maintains 100% backward compatibility with v1.0 while adding powerful new capabilities for thematic research analysis and interdisciplinary collaboration discovery.
