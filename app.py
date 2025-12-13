"""
SASTRA Research Finder - Main Application
Fixed Profile Navigation with Mistral AI
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from search_engine import get_engine, extract_keywords  # type: ignore
from mistral_rag import get_rag  # type: ignore

st.set_page_config(
    page_title="SASTRA Research Finder",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: bold; color: #1E3A8A; text-align: center; margin-bottom: 0.3rem; }
    .subtitle { text-align: center; color: #64748B; margin-bottom: 1.5rem; }
    .stat-box { background: #f1f5f9; padding: 0.8rem; border-radius: 8px; text-align: center; }
    .stat-num { font-size: 1.5rem; font-weight: bold; color: #1e40af; }
    .stat-label { font-size: 0.8rem; color: #64748B; }
    .result-card { background: #f8fafc; padding: 1rem; border-radius: 8px; border-left: 4px solid #3B82F6; margin-bottom: 0.8rem; }
    .author-id { font-family: monospace; background: #dbeafe; padding: 0.2rem 0.5rem; border-radius: 4px; color: #1e40af; font-size: 0.85rem; }
    .author-name { font-weight: 600; color: #065f46; font-size: 1.1rem; }
    .pub-title { font-weight: 600; color: #1E40AF; }
    .abstract-box { background: #f0f9ff; padding: 0.8rem; border-radius: 6px; font-size: 0.9rem; margin-top: 0.5rem; border: 1px solid #bae6fd; }
    .skill-tag { display: inline-block; background: #dcfce7; color: #166534; padding: 0.2rem 0.6rem; border-radius: 15px; font-size: 0.85rem; margin: 0.15rem; }
    .keyword-tag { display: inline-block; background: #fef3c7; color: #92400e; padding: 0.2rem 0.5rem; border-radius: 10px; font-size: 0.8rem; margin: 0.1rem; }
    .profile-card { background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); padding: 1.5rem; border-radius: 12px; border-left: 5px solid #2563eb; }
    .profile-header { font-size: 1.3rem; font-weight: bold; color: #1e40af; }
    .results-table { background: white; border-radius: 8px; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .table-header { font-weight: bold; color: #1e40af; padding: 0.5rem; border-bottom: 2px solid #3B82F6; margin-bottom: 0.5rem; }
    .back-button { margin-bottom: 1rem; }
    .collab-badge { display: inline-block; padding: 0.3rem 0.7rem; border-radius: 12px; font-size: 0.85rem; font-weight: 600; margin: 0.2rem; }
    .national-badge { background: #dbeafe; color: #1e40af; }
    .international-badge { background: #fef3c7; color: #92400e; }
</style>
""", unsafe_allow_html=True)


def get_author_profile_safe(engine, author_id=None, author_name=None):
    """
    Safely get author profile using either ID or name.
    Returns (profile_dict, error_message)
    """
    try:
        if author_id:
            # Try search_by_author_id first
            if hasattr(engine, 'search_by_author_id'):
                profile = engine.search_by_author_id(author_id)
                if profile:
                    return profile, None
            
            # Fallback to get_author_profile
            if hasattr(engine, 'get_author_profile'):
                profile = engine.get_author_profile(author_id)
                if profile:
                    return profile, None
            
            # Try from author_profiles dict directly
            if hasattr(engine, 'author_profiles') and author_id in engine.author_profiles:
                return engine.author_profiles[author_id], None
            
            return None, f"Author ID '{author_id}' not found"
        
        elif author_name:
            # Search by name first
            if hasattr(engine, 'search_by_author_name'):
                results = engine.search_by_author_name(author_name)
                if results and results.get('results'):
                    first_author_id = results['results'][0]['author_id']
                    return get_author_profile_safe(engine, author_id=first_author_id)
            
            return None, f"No author found with name '{author_name}'"
        
        return None, "No author ID or name provided"
    
    except Exception as e:
        return None, f"Error loading profile: {str(e)}"


def display_profile(profile: dict, engine=None):
    """Display full author profile with all papers and abstracts."""
    if not profile:
        st.error("❌ Profile data is empty")
        return
    
    # Handle different profile structures
    name_variants = profile.get('name_variants', [])
    author_id = profile.get('author_id', 'Unknown')
    
    if not name_variants:
        name_variants = [profile.get('name', 'Unknown Author')]
    
    st.markdown(f"""
    <div class="profile-card">
        <div class="profile-header">👤 {', '.join(name_variants[:3])}</div>
        <p><b>Author ID:</b> <span class="author-id">{author_id}</span></p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Publications", profile.get('pub_count', profile.get('total_papers', 0)))
    c2.metric("Total Citations", profile.get('total_citations', 0))
    c3.metric("Name Variants", len(name_variants))
    
    if name_variants and len(name_variants) > 1:
        with st.expander("📝 All Name Variants"):
            for name in name_variants:
                st.write(f"• {name}")
    
    if profile.get('top_keywords'):
        st.write("**🏷️ Top Research Keywords:**")
        kws = [f"{k} ({c})" for k, c in profile['top_keywords'][:10]]
        st.write(", ".join(kws))
    
    if profile.get('affiliations'):
        with st.expander("🏛️ Affiliations"):
            for aff in profile['affiliations'][:5]:
                st.write(f"• {aff}")
    
    # NEW: Citation Histogram
    st.divider()
    st.subheader("📊 Citation Distribution")
    hist_data = engine.get_citation_histogram_data(author_id)
    
    if hist_data['citation_list']:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Publications", hist_data['total_pubs'])
        col2.metric("Average Citations", f"{hist_data['avg_citations']:.1f}")
        col3.metric("Max Citations", hist_data['max_citations'])
        
        # Create DataFrame for bar chart
        df_hist = pd.DataFrame({
            'Citation Range': hist_data['bins'],
            'Number of Publications': hist_data['counts']
        })
        st.bar_chart(df_hist.set_index('Citation Range'))
    else:
        st.info("No citation data available for this author")
    
    # NEW: National vs International Collaboration
    st.divider()
    st.subheader("🤝 Collaboration Analysis")
    collab_data = engine.get_national_international_collab(author_id)
    
    if collab_data['total'] > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🇮🇳 National Collaborations")
            st.metric("", collab_data['national'])
            percentage = (collab_data['national'] / collab_data['total'] * 100) if collab_data['total'] > 0 else 0
            st.caption(f"{percentage:.1f}% of all collaborations")
        
        with col2:
            st.markdown("### 🌍 International Collaborations")
            st.metric("", collab_data['international'])
            percentage = (collab_data['international'] / collab_data['total'] * 100) if collab_data['total'] > 0 else 0
            st.caption(f"{percentage:.1f}% of all collaborations")
        
        # Country-wise breakdown
        st.markdown("### 🗺️ Country-Wise Collaboration (India with Other Countries)")
        country_data = engine.get_country_collaboration_data(author_id)
        
        if country_data['india_collabs']:
            # Create DataFrame for visualization
            df_countries = pd.DataFrame(
                list(country_data['india_collabs'].items()),
                columns=['Country', 'Collaborations']
            ).sort_values('Collaborations', ascending=False).head(10)
            
            st.bar_chart(df_countries.set_index('Country'))
            
            # Show top countries in table
            with st.expander("📋 View All Collaborating Countries"):
                for country, count in country_data['top_countries']:
                    st.write(f"**{country}**: {count} joint publications with India")
        else:
            st.info("No international collaboration data available")
    else:
        st.info("No collaboration data available for this author")
    
    publications = profile.get('publications', [])
    st.divider()
    st.subheader(f"📚 Publications ({len(publications)})")
    
    if not publications:
        st.info("No publications found for this author")
    else:
        # Show first 20 publications
        for idx, pub in enumerate(publications[:20]):
            collab_type = 'National' if pub.get('countries') and all(c.lower() == 'india' for c in pub['countries']) else 'International' if pub.get('countries') else 'Unknown'
            badge_class = 'national-badge' if collab_type == 'National' else 'international-badge'
            
            with st.expander(f"📄 {pub.get('title', 'Untitled')} ({pub.get('year', 'N/A')})"):
                # Show collaboration type badge
                if collab_type != 'Unknown':
                    st.markdown(f'<span class="collab-badge {badge_class}">{collab_type} Collaboration</span>', unsafe_allow_html=True)
                
                st.write(f"**Authors:** {pub.get('authors', 'N/A')}")
                st.write(f"**Source:** {pub.get('source', 'N/A')}")
                st.write(f"**Citations:** {pub.get('citations', 0)}")
                st.write(f"**Document Type:** {pub.get('document_type', 'N/A')}")
                
                if pub.get('countries'):
                    st.write(f"**Countries:** {', '.join(pub['countries'])}")
                
                if pub.get('keywords'):
                    st.write(f"**Keywords:** {pub['keywords']}")
                
                abstract = pub.get('abstract', 'No abstract available')
                st.markdown(f"""
                <div class="abstract-box">
                    <b>📋 Abstract:</b><br><br>{abstract}
                </div>
                """, unsafe_allow_html=True)
        
        if len(publications) > 20:
            st.info(f"Showing first 20 of {len(publications)} publications")


def main():
    st.markdown('<div class="main-title">🔬 SASTRA Research Finder</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Author ID-Based Search with Abstract Matching & Mistral AI RAG</div>', unsafe_allow_html=True)

    # Initialize session state
    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = 'main'
    if 'viewing_author_id' not in st.session_state:
        st.session_state.viewing_author_id = None
    if 'viewing_author_name' not in st.session_state:
        st.session_state.viewing_author_name = None
    if 'last_search_results' not in st.session_state:
        st.session_state.last_search_results = None

    # Load engine
    try:
        engine = get_engine()
        rag = get_rag()
        stats = engine.get_stats()
        st.session_state.engine = engine
        st.session_state.rag_instance = rag
    except Exception as e:
        st.error(f"""
        ❌ **Database not found!**
        
        Run preprocessing first:
        ```bash
        python src/preprocess.py
        ```
        
        Error: {e}
        """)
        return

    # ========== PROFILE VIEW MODE ==========
    if st.session_state.view_mode == 'profile':
        # Back button at the top
        col1, col2, col3 = st.columns([1, 4, 1])
        with col1:
            if st.button("⬅️ Back", key="back_button", use_container_width=True):
                st.session_state.view_mode = 'main'
                st.rerun()
        
        st.divider()
        
        # Get and display profile
        profile, error = get_author_profile_safe(
            engine, 
            author_id=st.session_state.viewing_author_id,
            author_name=st.session_state.viewing_author_name
        )
        
        if error:
            st.error(f"❌ {error}")
            st.info("Click 'Back' button to return to search")
            
            # Debug info
            with st.expander("🔍 Debug Information"):
                st.write("**Session State:**")
                st.write(f"- Author ID: {st.session_state.viewing_author_id}")
                st.write(f"- Author Name: {st.session_state.viewing_author_name}")
                st.write(f"- View Mode: {st.session_state.view_mode}")
        elif profile:
            display_profile(profile, engine)
        else:
            st.error("❌ Profile not found")
        
        # Stop here - don't render tabs
        return
    
    # ========== MAIN VIEW MODE ==========
    # Publications stat - centered
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{stats["publications"]:,}</div><div class="stat-label">Publications</div></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Features Overview - Using Streamlit native components
    st.info("**🌟 Key Features of SASTRA Research Finder**")
    
    # Feature 1
    with st.container():
        st.markdown("#### 👤 Author / ID Lookup")
        st.write("Enables search by author name or Author ID and outputs a unified author profile including name variants, publications, citations, affiliations, collaboration stats, and research impact.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Feature 2
    with st.container():
        st.markdown("#### 🔍 Keyword Search")
        st.write("Searches all research abstracts using user keywords and outputs relevant authors, matched publications, keyword highlights, and document type distribution.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Feature 3
    with st.container():
        st.markdown("#### 🎯 Skill-Based Search")
        st.write("Extracts skills from a project title and outputs a ranked list of researchers whose expertise closely matches the selected or custom skills.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Feature 4
    with st.container():
        st.markdown("#### 📊 RAG Analysis (Mistral AI)")
        st.write("Uses relevant SASTRA abstracts with Mistral AI to generate grounded research summaries, key insights, research gaps, and future research directions. It also displays the most relevant authors along with their corresponding abstracts, ensuring that all AI-generated insights are transparent, traceable, and based entirely on institutional research data.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Feature 5
    with st.container():
        st.markdown("#### 📈 Analytics Dashboard")
        st.write("Aggregates institute-level research data and outputs top keywords, active researchers, collaboration trends, and topic trends over time.")
    
    st.divider()

    # Mistral status
    if rag.is_available():
        st.success("✅ Mistral AI Connected")
    else:
        st.info("ℹ️ Add MISTRAL_API_KEY to Streamlit Secrets for AI features")

    # Main tabs - REORDERED: Author/ID Lookup first
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "👤 Author/ID Lookup",
        "🔍 Keyword Search",
        "🎯 Skill-Based Search",
        "📊 RAG Analysis",
        "📈 Analytics"
    ])
    
    # ==================== TAB 1: AUTHOR/ID LOOKUP ====================
    with tab1:
        st.subheader("Direct Author Lookup")
        
        search_type = st.radio(
            "Search by:",
            ["Author Name", "Author ID"],
            horizontal=True,
            key="lookup_type_radio"
        )
        
        if search_type == "Author ID":
            author_id_input = st.text_input(
                "Enter Author ID:",
                placeholder="e.g., 57192051462",
                key="author_id_lookup_input"
            )
            
            if st.button("🔍 Lookup Author", key="lookup_by_id_btn"):
                if author_id_input.strip():
                    st.session_state.view_mode = 'profile'
                    st.session_state.viewing_author_id = author_id_input.strip()
                    st.session_state.viewing_author_name = None
                    st.rerun()
        
        else:  # Author Name
            author_name_input = st.text_input(
                "Enter Author Name:",
                placeholder="e.g., Brindha or Chandiramouli, R.",
                key="author_name_lookup_input"
            )
            
            if st.button("🔍 Search Name", key="lookup_by_name_btn"):
                if author_name_input.strip():
                    results = engine.search_by_author_name(author_name_input.strip())
                    
                    if results and results.get('results'):
                        st.success(f"Found **{results['total']}** matching authors")
                        
                        st.write("**Select an author:**")
                        for idx, r in enumerate(results['results'][:20]):
                            cols = st.columns([3, 7])
                            
                            with cols[0]:
                                st.code(r['author_id'])
                            
                            with cols[1]:
                                names = ', '.join(r['name_variants'][:2])
                                st.write(names)
                    else:
                        st.warning(f"No results for '{author_name_input}'")

    
    # ==================== TAB 2: KEYWORD SEARCH ====================
    with tab2:
        st.subheader("Keyword → Abstract Matching")
        
        keywords_input = st.text_input(
            "Enter Keywords:",
            placeholder="machine learning, deep learning, classification",
            key="phase1_keywords"
        )
        
        if st.button("🔍 Search Abstracts", key="phase1_search"):
            if not keywords_input.strip():
                st.warning("Please enter at least one keyword.")
            else:
                with st.spinner("Searching abstracts..."):
                    results = engine.search_by_keywords(keywords_input)
                    st.session_state.last_search_results = results
                
                st.success(f"Found **{results['total']}** Author IDs from **{results.get('total_matching_pubs', 0)}** matching publications")
                
                if results.get('keywords_used'):
                    st.write("**Keywords used:**")
                    kw_html = " ".join([f'<span class="keyword-tag">{k}</span>' for k in results['keywords_used'][:20]])
                    st.markdown(kw_html, unsafe_allow_html=True)
                
                # NEW: Document Type Distribution
                if results.get('document_type_dist'):
                    st.divider()
                    st.subheader("📄 Document Type Distribution")
                    
                    doc_dist = results['document_type_dist']
                    df_doc_types = pd.DataFrame(
                        list(doc_dist.items()),
                        columns=['Document Type', 'Count']
                    ).sort_values('Count', ascending=False)
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.bar_chart(df_doc_types.set_index('Document Type'))
                    
                    with col2:
                        st.write("**Distribution:**")
                        for doc_type, count in df_doc_types.values:
                            percentage = (count / results['total_matching_pubs'] * 100) if results['total_matching_pubs'] > 0 else 0
                            st.write(f"**{doc_type}**: {count} ({percentage:.1f}%)")
        
        # Display results if they exist
        if st.session_state.last_search_results and 'results' in st.session_state.last_search_results:
            results = st.session_state.last_search_results
            
            st.divider()
            st.subheader("Results by Author ID")
            st.caption("👆 Click to view full profile")
            
            # Create clickable results
            for idx, r in enumerate(results['results'][:50]):
                cols = st.columns([2, 5, 1.5, 1.5])
                
                with cols[0]:
                    st.code(r['author_id'])
                
                with cols[1]:
                    author_name = ', '.join(r['name_variants'][:2])
                    st.write(author_name)
                
                with cols[2]:
                    st.write(f"📄 {r['matching_papers']} papers")
                
                with cols[3]:
                    # Single unified button
                    if st.button("View Profile", key=f"view_profile_kw_{idx}"):
                        st.session_state.view_mode = 'profile'
                        st.session_state.viewing_author_id = r['author_id']
                        st.session_state.viewing_author_name = None
                        st.rerun()

    
    # ==================== TAB 3: SKILL-BASED SEARCH ====================
    with tab3:
        st.subheader("Skill-Based Research Discovery")
        
        project_title = st.text_input(
            "Project Title:",
            placeholder="Deep learning based segmentation models for MRI analysis",
            key="phase2_title"
        )
        
        if project_title:
            st.write("**Extracting skills from title...**")
            
            initial_kws = extract_keywords(project_title, max_keywords=5)
            preliminary_search = engine.search_by_keywords(','.join(initial_kws))

            abstract_kws = []
            for r in preliminary_search['results'][:10]:
                for pid in r.get('pub_ids', [])[:3]:
                    if hasattr(engine, 'abstract_keywords') and pid in engine.abstract_keywords:
                        abstract_kws.extend(engine.abstract_keywords[pid][:5])
            abstract_kws = list(set(abstract_kws))[:20]
            
            if rag.is_available():
                extracted_skills = rag.extract_skills(project_title, abstract_kws)
            else:
                extracted_skills = extract_keywords(project_title, min_length=3, max_keywords=10)
            
            st.write("**Suggested Skills:**")
            
            selected_skills = []
            cols = st.columns(4)
            for i, skill in enumerate(extracted_skills[:12]):
                if cols[i % 4].checkbox(skill, value=True, key=f"skill_{i}"):
                    selected_skills.append(skill)
            
            custom_skills = st.text_input(
                "Add custom skills (comma-separated):",
                placeholder="neural networks, medical imaging",
                key="custom_skills"
            )
            
            if custom_skills:
                for s in custom_skills.split(','):
                    s = s.strip().lower()
                    if s and s not in selected_skills:
                        selected_skills.append(s)
            
            if selected_skills:
                st.write("**Selected skills:**")
                skills_html = " ".join([f'<span class="skill-tag">{s}</span>' for s in selected_skills])
                st.markdown(skills_html, unsafe_allow_html=True)
            
            if st.button("🔍 Find Researchers", key="phase2_search"):
                if not selected_skills:
                    st.warning("Please select at least one skill.")
                else:
                    with st.spinner("Searching..."):
                        results = engine.search_by_skills(selected_skills)
                        st.session_state.last_search_results = results
                    
                    st.success(f"Found **{results['total']}** Author IDs")
            
            # Display skill-based results
            if st.session_state.last_search_results and 'results' in st.session_state.last_search_results:
                results = st.session_state.last_search_results
                
                st.divider()
                st.subheader("Matching Researchers")
                
                for idx, r in enumerate(results['results'][:50]):
                    cols = st.columns([2, 5, 1.5, 1.5])
                    
                    with cols[0]:
                        st.code(r['author_id'])
                    
                    with cols[1]:
                        author_name = ', '.join(r['name_variants'][:2])
                        st.write(author_name)
                    
                    with cols[2]:
                        st.write(f"📄 {r['matching_papers']} papers")
                    
                    with cols[3]:
                        if st.button("View Profile", key=f"view_profile_skill_{idx}"):
                            st.session_state.view_mode = 'profile'
                            st.session_state.viewing_author_id = r['author_id']
                            st.session_state.viewing_author_name = None
                            st.rerun()


    # ==================== TAB 4: RAG ANALYSIS ====================
    with tab4:
        st.subheader("AI-Powered Research Analysis (Mistral AI)")
        
        if not rag.is_available():
            st.warning("**Mistral API not configured.** Add MISTRAL_API_KEY to Streamlit Secrets")
        
        rag_skills_input = st.text_input(
            "Enter skills for analysis:",
            placeholder="deep learning, image segmentation, medical imaging",
            key="rag_skills"
        )
        
        if st.button("🤖 Generate Analysis", key="rag_analyze"):
            if not rag_skills_input.strip():
                st.warning("Please enter skills.")
            else:
                skills = [s.strip().lower() for s in rag_skills_input.split(',') if s.strip()]
                
                with st.spinner("Gathering context..."):
                    context = engine.get_rag_context(skills, max_abstracts=20)
                
                st.info(f"Found {len(context)} relevant abstracts")
                
                # Store context in session state for table display
                st.session_state.rag_context = context
                
                with st.spinner("Generating AI analysis..."):
                    analysis = rag.analyze(context, skills)
                
                if analysis.get('error'):
                    st.error(f"Error: {analysis['error']}")
                
                if analysis.get('analysis'):
                    st.markdown("---")
                    st.markdown(analysis['analysis'])
                    st.markdown("---")
        
        # Display authors in table format like other tabs
        if st.session_state.get('rag_context'):
            context = st.session_state.rag_context
            
            # Extract unique authors from context
            author_data = {}
            for c in context:
                author_id = c.get('author_id')
                if author_id and author_id not in author_data:
                    # Get full profile to get name variants
                    profile, _ = get_author_profile_safe(engine, author_id=author_id)
                    if profile:
                        author_data[author_id] = {
                            'author_id': author_id,
                            'name_variants': profile.get('name_variants', [c.get('authors', 'Unknown')]),
                            'matching_papers': 1
                        }
                    else:
                        author_data[author_id] = {
                            'author_id': author_id,
                            'name_variants': [c.get('authors', 'Unknown')],
                            'matching_papers': 1
                        }
                elif author_id:
                    author_data[author_id]['matching_papers'] += 1
            
            # Convert to list sorted by matching papers
            author_list = sorted(author_data.values(), key=lambda x: x['matching_papers'], reverse=True)
            
            st.subheader(f"📊 Relevant Authors ({len(author_list)})")
            st.caption("👆 Click to view full profile")
            
            # Display in table format
            for idx, r in enumerate(author_list[:50]):
                cols = st.columns([2, 5, 1.5, 1.5])
                
                with cols[0]:
                    st.code(r['author_id'])
                
                with cols[1]:
                    author_name = ', '.join(r['name_variants'][:2])
                    st.write(author_name)
                
                with cols[2]:
                    st.write(f"📄 {r['matching_papers']} papers")
                
                with cols[3]:
                    if st.button("View Profile", key=f"view_profile_rag_{idx}"):
                        st.session_state.view_mode = 'profile'
                        st.session_state.viewing_author_id = r['author_id']
                        st.session_state.viewing_author_name = None
                        st.rerun()
            
            # Source papers in expander (keep this for reference)
            with st.expander("📚 View Source Papers Details"):
                for idx, c in enumerate(context[:20], 1):
                    st.write(f"**{idx}. {c.get('title', 'Untitled')}**")
                    st.caption(f"Author ID: `{c.get('author_id', 'N/A')}` | Authors: {c.get('authors', 'N/A')}")
                    if c.get('abstract'):
                        st.markdown(f"""
                        <div class="abstract-box">
                            <b>📋 Abstract:</b><br>{c['abstract'][:500]}{'...' if len(c['abstract']) > 500 else ''}
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown("---")

    # ==================== TAB 5: ANALYTICS ====================
    with tab5:
        st.subheader("📈 Research Analytics Dashboard")
        
        # Create sub-tabs for different analytics (REMOVED PUBLICATION TRENDS AND COLLABORATIONS)
        analytics_tab1, analytics_tab2, analytics_tab3 = st.tabs([
            "🏷️ Top Keywords",
            "👥 Active Researchers",
            "📊 Keyword Trends"
        ])
        
        # ===== ANALYTICS TAB 1: Top Keywords =====
        with analytics_tab1:
            st.markdown("### 🏷️ Top Research Keywords at SASTRA")
            
            top_keywords = engine.get_top_keywords(limit=20)
            
            if top_keywords:
                # Create bar chart
                df_keywords = pd.DataFrame(top_keywords, columns=['Keyword', 'Count'])
                df_top10 = df_keywords.head(10)
                
                st.bar_chart(df_top10.set_index('Keyword'))
                
                # Show table with all 20
                st.markdown("**📋 Top 20 Keywords:**")
                
                # Display in 2 columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Rank 1-10:**")
                    for idx, (kw, count) in enumerate(top_keywords[:10], 1):
                        st.write(f"{idx}. **{kw}** - {count} occurrences")
                
                with col2:
                    st.markdown("**Rank 11-20:**")
                    for idx, (kw, count) in enumerate(top_keywords[10:20], 11):
                        st.write(f"{idx}. **{kw}** - {count} occurrences")
        
        # ===== ANALYTICS TAB 2: Most Active Authors =====
        with analytics_tab2:
            st.markdown("### 👥 Most Active Researchers")
            
            active_authors = engine.get_most_active_authors(limit=15)
            
            if active_authors:
                # Create DataFrame
                df_authors = pd.DataFrame(active_authors)
                df_top10 = df_authors.head(10)
                
                # Horizontal bar chart
                st.markdown("#### 📊 Top 10 by Publication Count")
                df_chart = df_top10[['name', 'pub_count']].set_index('name')
                st.bar_chart(df_chart)
                
                # Detailed table
                st.markdown("#### 📋 Detailed Leaderboard")
                
                for idx, author in enumerate(active_authors, 1):
                    with st.expander(f"#{idx} {author['name']} - {author['pub_count']} publications"):
                        col1, col2 = st.columns([2, 3])
                        
                        with col1:
                            st.write(f"**Author ID:** `{author['author_id']}`")
                            st.write(f"**Publications:** {author['pub_count']}")
                            st.write(f"**Citations:** {author['total_citations']}")
                        
                        with col2:
                            st.write("**Top Research Areas:**")
                            for kw in author['top_keywords']:
                                st.markdown(f'<span class="skill-tag">{kw}</span>', unsafe_allow_html=True)
                        
                        if st.button("View Full Profile", key=f"analytics_profile_{idx}"):
                            st.session_state.view_mode = 'profile'
                            st.session_state.viewing_author_id = author['author_id']
                            st.session_state.viewing_author_name = None
                            st.rerun()
        
        # ===== ANALYTICS TAB 3: Keyword Trends Over Time =====
        with analytics_tab3:
            st.markdown("### 📊 Keyword Trend Analysis")
            st.caption("Track how specific research topics have evolved over time")
            
            keyword_input = st.text_input(
                "Enter a keyword to track:",
                placeholder="e.g., machine learning, deep learning, neural networks",
                key="trend_keyword"
            )
            
            if st.button("📈 Analyze Trend", key="analyze_trend"):
                if keyword_input.strip():
                    with st.spinner("Analyzing keyword trend..."):
                        trend_data = engine.get_keyword_trend_over_time(keyword_input)
                    
                    if trend_data['years']:
                        st.success(f"Found **{trend_data['total_publications']}** publications containing '{keyword_input}'")
                        
                        # Create line chart
                        df_trend = pd.DataFrame({
                            'Year': trend_data['years'],
                            'Publications': trend_data['counts']
                        })
                        
                        st.line_chart(df_trend.set_index('Year'))
                        
                        # Statistics
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total Publications", trend_data['total_publications'])
                        col2.metric("Years Covered", len(trend_data['years']))
                        col3.metric("Peak Year", max(trend_data['data'].items(), key=lambda x: x[1])[0])
                        col4.metric("Peak Count", max(trend_data['counts']))
                        
                        # Trend analysis
                        if len(trend_data['years']) >= 3:
                            recent_avg = sum(trend_data['counts'][-3:]) / 3
                            early_avg = sum(trend_data['counts'][:3]) / 3
                            
                            if recent_avg > early_avg * 1.5:
                                st.info("📈 **Trending UP** - This topic is gaining momentum at SASTRA!")
                            elif recent_avg < early_avg * 0.5:
                                st.warning("📉 **Declining** - This topic is receiving less attention recently")
                            else:
                                st.success("➡️ **Stable** - Consistent research activity over time")
                        
                        # Year-by-year breakdown
                        with st.expander("📅 Year-by-Year Breakdown"):
                            for year, count in sorted(trend_data['data'].items(), reverse=True):
                                st.write(f"**{year}:** {count} publications")
                    else:
                        st.warning(f"No publications found containing the keyword '{keyword_input}'")
                        st.info("💡 Try different variations or related terms")
                else:
                    st.warning("Please enter a keyword")

    st.divider()
    st.caption("SASTRA Research Finder | Author ID-Based | Mistral AI")


if __name__ == "__main__":
    main()