# Quick Start Guide - SASTRA Research Finder Enhanced

Get up and running in 5 minutes!

## ⚡ Windows Users (Easiest)

1. **Extract the ZIP file**
   - Right-click → Extract All
   - Choose a location (e.g., Desktop)

2. **Double-click `run.bat`**
   - That's it! The script will:
     - Check Python installation
     - Create virtual environment
     - Install dependencies
     - Start the application

3. **Open your browser**
   - Automatic: Should open at http://localhost:8501
   - Manual: Go to http://localhost:8501

## 🐧 Linux/Mac Users

```bash
# Navigate to extracted folder
cd sastra_research_finder_enhanced

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Mac/Linux
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 🔑 Enable Mistral AI (Optional but Recommended)

### For Local Development:

1. **Get API Key**:
   - Visit: https://console.mistral.ai/api-keys/
   - Sign up (free tier available)
   - Copy your API key

2. **Create `.env` file**:
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env and add your key
   MISTRAL_API_KEY=your_actual_api_key_here
   ```

3. **Restart the app**:
   - Press Ctrl+C in terminal
   - Run `streamlit run app.py` again

### For Streamlit Cloud Deployment:

1. Go to Streamlit Cloud dashboard
2. Click on your app → Settings → Secrets
3. Add:
   ```
   MISTRAL_API_KEY = "your_actual_api_key_here"
   ```
4. Save and restart

## ✅ Verify Installation

After starting the app, you should see:

1. **Main Page**: 
   - Title: "SASTRA Research Finder"
   - Stats showing publication count
   - 6 tabs visible

2. **Status Messages** (in terminal):
   - `✓ Loaded X publications`
   - `✓ Loaded X author profiles`
   - `✓ Mistral AI initialized` (if API key configured)

3. **All Tabs Working**:
   - 👤 Author/ID Lookup
   - 🔍 Keyword Search
   - 🎯 Skill-Based Search
   - 📊 RAG Analysis
   - 📈 Analytics
   - 🎨 Thematic Areas (NEW!)

## 🎯 Try the New Thematic Areas Feature

1. **Click on "🎨 Thematic Areas" tab**

2. **Select "📚 Single Thematic Areas"**

3. **Click any theme** (e.g., "Machine Learning")
   - See top 10 faculty
   - Interactive scatter plot
   - Click on faculty to see papers

4. **Try "🤝 Interdisciplinary Teams"**
   - Choose "2 Themes"
   - Click a combination (e.g., "Machine Learning + Medical Imaging")
   - See balanced research teams

## 🆘 Troubleshooting

### "Python not found"
- Install Python 3.9 or higher from python.org
- Make sure to check "Add Python to PATH" during installation

### "Module not found" errors
```bash
pip install -r requirements.txt --upgrade
```

### Port 8501 already in use
```bash
streamlit run app.py --server.port 8502
```

### Mistral AI not working
- Check API key in .env file
- Verify no extra spaces or quotes
- Restart the app after adding key

### Data files missing
```bash
# Run preprocessing
python src/preprocess.py
```

### Slow performance
- This is normal for first load (building indexes)
- Subsequent loads are much faster (uses cached .pkl files)

## 📚 Next Steps

1. **Read the Docs**:
   - `README.md` - Full feature overview
   - `THEMATIC_AREAS_GUIDE.md` - Detailed guide for new feature
   - `CHANGELOG.md` - What's new in v2.0

2. **Explore Features**:
   - Start with Author/ID Lookup
   - Try Keyword Search
   - Test RAG Analysis with Mistral AI
   - Explore Thematic Areas

3. **Customize**:
   - Add more thematic areas in `src/thematic_areas.py`
   - Adjust search weights in `src/search_engine.py`
   - Modify RAG prompts in `src/mistral_rag.py`

## 🎓 Learn More

- [Mistral AI Documentation](https://docs.mistral.ai/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Documentation](https://plotly.com/python/)

## 💡 Pro Tips

1. **Use RAG with Thematic Areas**: 
   - Find experts in Thematic Areas
   - Use RAG to analyze their methodologies

2. **Bookmark Profiles**:
   - Copy Author IDs from results
   - Use Direct Lookup for quick access

3. **Export Data**:
   - Use browser print/PDF for reports
   - Copy tables to Excel/Sheets

4. **Team Planning**:
   - Build teams in Thematic Areas
   - Verify expertise with full profiles
   - Use RAG for project proposals

---

**Ready to start? Double-click `run.bat` (Windows) or run `streamlit run app.py`!**

Need help? Check the documentation or contact support.
