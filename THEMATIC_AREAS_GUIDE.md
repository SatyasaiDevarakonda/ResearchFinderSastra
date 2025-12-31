# Thematic Areas Feature - User Guide

## Overview

The Thematic Areas tab provides a powerful way to explore research at SASTRA University organized by research domains. It enables you to:
1. View top faculty in any research theme
2. Visualize research impact with interactive charts
3. Build interdisciplinary research teams
4. Access detailed papers and abstracts

## How to Use

### 🎨 Single Thematic Areas

1. **Navigate to Tab**: Click on "🎨 Thematic Areas" tab
2. **Select Sub-tab**: Choose "📚 Single Thematic Areas"
3. **Browse Categories**: Themes are organized by domain:
   - 💻 Computer Science & AI
   - 🏥 Healthcare & Medicine
   - ⚙️ Engineering
   - 🧪 Materials & Chemistry
   - 🌱 Environmental & Sustainability
   - 📊 Other Domains

4. **Click a Theme**: Select any theme button (e.g., "Machine Learning (15)")
   - Number in parentheses shows faculty count

5. **View Results**:
   - **Scatter Plot**: Visual representation of faculty impact
     - X-axis: Number of papers
     - Y-axis: Total citations
     - Bubble size: Citation impact
     - Hover for details
   
   - **Faculty Rankings**: Expandable list of top 10 faculty
     - Click to expand and see:
       - Author ID and metrics
       - Top 5 papers in this theme
       - Full abstracts
       - "View Full Profile" button

### 🤝 Interdisciplinary Teams

1. **Navigate**: Select "🤝 Interdisciplinary Teams" sub-tab

2. **Choose Combination Type**:
   - **2 Themes**: Combines expertise from 2 research areas
   - **3 Themes**: Advanced combination of 3 areas

3. **Select Themes**: Click on any theme combination button
   - Example: "Machine Learning + Medical Imaging"
   - Example: "Deep Learning + Bioinformatics + Signal Processing"

4. **Review Teams**:
   - System automatically forms up to 10 balanced teams
   - Teams ranked by total citation impact
   - **Team Composition**:
     - Team 1: Highest cite score researchers from each theme
     - Team 2: 2nd highest cite scores
     - And so on...

5. **Explore Team Details**:
   - Total team citations
   - Average citations per member
   - Individual member details:
     - Theme expertise
     - Author ID
     - Citation metrics
     - Top 3 papers with abstracts
     - Profile access button

## Understanding the Data

### Citation Score
- **What**: Total citations for all papers in the specific theme
- **Why**: Indicates research impact and influence
- **Use**: Higher scores suggest established expertise

### Paper Count
- **What**: Number of publications in the theme
- **Why**: Shows depth of research experience
- **Use**: More papers indicate sustained research activity

### Theme Assignment
- **Method**: Automatic based on keywords, abstracts, and titles
- **Threshold**: Score > 2.0 required for theme assignment
- **Multiple**: Faculty can appear in multiple themes

## Team Formation Logic

### How Teams Are Built
1. **Ranking**: Faculty ranked by citation score in each theme
2. **Matching**: System matches ranks across themes
   - Team 1 gets #1 from Theme A + #1 from Theme B + #1 from Theme C
   - Team 2 gets #2 from Theme A + #2 from Theme B + #2 from Theme C
3. **Balancing**: Ensures no faculty appears in multiple teams
4. **Quality**: Only forms teams with members from ALL selected themes

### Why This Approach?
- **Balanced Expertise**: Similar impact levels across themes
- **Fair Distribution**: Top researchers distributed across teams
- **Quality Assurance**: Consistent team strength

## Tips for Best Results

### Finding Faculty
1. **Start Broad**: Use major themes first (Machine Learning, Medical Imaging)
2. **Go Specific**: Narrow down to niche areas (Time Series Analysis, Nanotechnology)
3. **Compare**: Check multiple related themes to find all relevant faculty

### Building Teams
1. **Complementary Skills**: Choose themes that complement each other
   - Good: "Machine Learning + Healthcare Analytics"
   - Good: "Materials Science + Nanotechnology + Chemical Engineering"

2. **Check Availability**: Look for combinations with sufficient faculty
   - 2-theme combos: Works with 3+ faculty per theme
   - 3-theme combos: Works best with 5+ faculty per theme

3. **Review Thoroughly**: Expand team details to understand expertise fit

### Using Visualizations
1. **Scatter Plot**: 
   - Look for outliers (high-impact researchers)
   - Identify clusters (similar research profiles)
   - Find rising stars (good citations, fewer papers)

2. **Bubble Size**: 
   - Larger bubbles = Higher impact
   - Useful for quick visual scanning

## Common Questions

**Q: Why doesn't a faculty appear in an obvious theme?**
A: Theme assignment requires sufficient keyword matches (score > 2.0). The faculty may work in the area but not have enough papers with matching keywords.

**Q: Can I form teams with more than 3 themes?**
A: Currently limited to 3 themes max for practical team sizes and better balance.

**Q: Why do some combinations show no teams?**
A: Insufficient faculty in one or more themes, or no overlapping ranks after excluding duplicates.

**Q: How often is the data updated?**
A: Based on the uploaded publications file. Rerun preprocessing after updating the Excel file.

**Q: Can I export team information?**
A: Use browser's print/PDF function or copy text from expanded sections.

## Technical Details

### Thematic Areas Included
30+ themes across 6 major categories covering:
- AI/ML, Computer Vision, NLP
- Medical imaging, Healthcare, Bioinformatics
- Robotics, Signal Processing, Communications
- Materials, Nanotechnology, Chemistry
- Environmental Engineering, Renewable Energy
- Operations Research, Financial Analytics, and more

### Performance
- Theme identification: Cached for fast access
- Team building: Real-time computation
- Visualizations: Interactive with Plotly
- Profile access: Instant navigation

## Need Help?

- Check the main README.md for general features
- Review CHANGELOG.md for latest updates
- Contact support for technical issues

---

**Pro Tip**: Combine the Thematic Areas tab with RAG Analysis for comprehensive research planning - use Thematic Areas to find experts, then use RAG to understand their methodologies!
