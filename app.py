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
from thematic_areas import load_single_theme_rankings, load_interdisciplinary_combinations  # type: ignore

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
    .theme-tile { 
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
        padding: 1.2rem; 
        border-radius: 12px; 
        border: 2px solid #bae6fd;
        cursor: pointer;
        transition: all 0.3s;
        margin-bottom: 1rem;
    }
    .theme-tile:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        transform: translateY(-2px);
    }
    .tile-title { font-size: 1.1rem; font-weight: bold; color: #1e40af; margin-bottom: 0.5rem; }
    .tile-score { font-size: 0.9rem; color: #64748b; }
    .tile-count { font-size: 0.85rem; color: #94a3b8; }
</style>
""", unsafe_allow_html=True)


def get_author_profile_safe(engine, author_id=None, author_name=None):
    """
    Safely get author profile using either ID or name.
    Returns (profile_dict, error_message)
    """
    try:
        if author_id:
            if hasattr(engine, 'search_by_author_id'):
                profile = engine.search_by_author_id(author_id)
                if profile:
                    return profile, None
            
            if hasattr(engine, 'get_author_profile'):
                profile = engine.get_author_profile(author_id)
                if profile:
                    return profile, None
            
            if hasattr(engine, 'author_profiles') and author_id in engine.author_profiles:
                return engine.author_profiles[author_id], None
            
            return None, f"Author ID '{author_id}' not found"
        
        elif author_name:
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
    
    st.divider()
    st.subheader("📊 Citation Distribution")
    hist_data = engine.get_citation_histogram_data(author_id)
    
    if hist_data['citation_list']:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Publications", hist_data['total_pubs'])
        col2.metric("Average Citations", f"{hist_data['avg_citations']:.1f}")
        col3.metric("Max Citations", hist_data['max_citations'])
        
        df_hist = pd.DataFrame({
            'Citation Range': hist_data['bins'],
            'Number of Publications': hist_data['counts']
        })
        st.bar_chart(df_hist.set_index('Citation Range'))
    else:
        st.info("No citation data available for this author")
    
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
        
        st.markdown("### 🗺️ Country-Wise Collaboration (India with Other Countries)")
        country_data = engine.get_country_collaboration_data(author_id)
        
        if country_data['india_collabs']:
            df_countries = pd.DataFrame(
                list(country_data['india_collabs'].items()),
                columns=['Country', 'Collaborations']
            ).sort_values('Collaborations', ascending=False).head(10)
            
            st.bar_chart(df_countries.set_index('Country'))
            
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
        for idx, pub in enumerate(publications[:20]):
            collab_type = 'National' if pub.get('countries') and all(c.lower() == 'india' for c in pub['countries']) else 'International' if pub.get('countries') else 'Unknown'
            badge_class = 'national-badge' if collab_type == 'National' else 'international-badge'
            
            with st.expander(f"📄 {pub.get('title', 'Untitled')} ({pub.get('year', 'N/A')})"):
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

    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = 'main'
    if 'viewing_author_id' not in st.session_state:
        st.session_state.viewing_author_id = None
    if 'viewing_author_name' not in st.session_state:
        st.session_state.viewing_author_name = None
    if 'last_search_results' not in st.session_state:
        st.session_state.last_search_results = None

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

    if st.session_state.view_mode == 'profile':
        col1, col2, col3 = st.columns([1, 4, 1])
        with col1:
            if st.button("⬅️ Back", key="back_button", use_container_width=True):
                st.session_state.view_mode = 'main'
                st.rerun()
        
        st.divider()
        
        profile, error = get_author_profile_safe(
            engine, 
            author_id=st.session_state.viewing_author_id,
            author_name=st.session_state.viewing_author_name
        )
        
        if error:
            st.error(f"❌ {error}")
            st.info("Click 'Back' button to return to search")
            
            with st.expander("🔍 Debug Information"):
                st.write("**Session State:**")
                st.write(f"- Author ID: {st.session_state.viewing_author_id}")
                st.write(f"- Author Name: {st.session_state.viewing_author_name}")
                st.write(f"- View Mode: {st.session_state.view_mode}")
        elif profile:
            display_profile(profile, engine)
        else:
            st.error("❌ Profile not found")
        
        return
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{stats["publications"]:,}</div><div class="stat-label">Publications</div></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.info("**🌟 Key Features of SASTRA Research Finder**")
    
    with st.container():
        st.markdown("#### 👤 Author / ID Lookup")
        st.write("Enables search by author name or Author ID and outputs a unified author profile including name variants, publications, citations, affiliations, collaboration stats, and research impact.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("#### 🔍 Keyword Search")
        st.write("Searches all research abstracts using user keywords and outputs relevant authors, matched publications, keyword highlights, and document type distribution.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("#### 🎯 Skill-Based Search")
        st.write("Extracts skills from a project title and outputs a ranked list of researchers whose expertise closely matches the selected or custom skills.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("#### 📊 RAG Analysis (Mistral AI)")
        st.write("Uses relevant SASTRA abstracts with Mistral AI to generate grounded research summaries, key insights, research gaps, and future research directions. It also displays the most relevant authors along with their corresponding abstracts, ensuring that all AI-generated insights are transparent, traceable, and based entirely on institutional research data.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("#### 📈 Analytics Dashboard")
        st.write("Aggregates institute-level research data and outputs top keywords, active researchers, collaboration trends, and topic trends over time.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("#### 🎨 Thematic Areas")
        st.write("Analyzes research by thematic domains (50+ themes like Machine Learning, Medical Imaging, Robotics) and ranks top 10 faculty by citation impact in each area. Also identifies high-value interdisciplinary team combinations (100+ pre-computed pairs/triplets) showing faculty collaboration opportunities across themes, with teams ranked by total citation score.")
    
    st.divider()

    if rag.is_available():
        st.success("✅ Mistral AI Connected")
    else:
        st.info("ℹ️ Add MISTRAL_API_KEY to Streamlit Secrets for AI features")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "👤 Author/ID Lookup",
        "🔍 Keyword Search",
        "🎯 Skill-Based Search",
        "📊 RAG Analysis",
        "📈 Analytics",
        "🎨 Thematic Areas"
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
        
        else:
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
        
        if st.session_state.last_search_results and 'results' in st.session_state.last_search_results:
            results = st.session_state.last_search_results
            
            st.divider()
            st.subheader("Results by Author ID")
            st.caption("👆 Click to view full profile")
            
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
                
                st.session_state.rag_context = context
                
                with st.spinner("Generating AI analysis..."):
                    analysis = rag.analyze(context, skills)
                
                if analysis.get('error'):
                    st.error(f"Error: {analysis['error']}")
                
                if analysis.get('analysis'):
                    st.markdown("---")
                    st.markdown(analysis['analysis'])
                    st.markdown("---")
        
        if st.session_state.get('rag_context'):
            context = st.session_state.rag_context
            
            author_data = {}
            for c in context:
                author_id = c.get('author_id')
                if author_id and author_id not in author_data:
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
            
            author_list = sorted(author_data.values(), key=lambda x: x['matching_papers'], reverse=True)
            
            st.subheader(f"📊 Relevant Authors ({len(author_list)})")
            st.caption("👆 Click to view full profile")
            
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
        
        analytics_tab1, analytics_tab2, analytics_tab3 = st.tabs([
            "🏷️ Top Keywords",
            "👥 Active Researchers",
            "📊 Keyword Trends"
        ])
        
        with analytics_tab1:
            st.markdown("### 🏷️ Top Research Keywords at SASTRA")
            
            top_keywords = engine.get_top_keywords(limit=20)
            
            if top_keywords:
                df_keywords = pd.DataFrame(top_keywords, columns=['Keyword', 'Count'])
                df_top10 = df_keywords.head(10)
                
                st.bar_chart(df_top10.set_index('Keyword'))
                
                st.markdown("**📋 Top 20 Keywords:**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Rank 1-10:**")
                    for idx, (kw, count) in enumerate(top_keywords[:10], 1):
                        st.write(f"{idx}. **{kw}** - {count} occurrences")
                
                with col2:
                    st.markdown("**Rank 11-20:**")
                    for idx, (kw, count) in enumerate(top_keywords[10:20], 11):
                        st.write(f"{idx}. **{kw}** - {count} occurrences")
        
        with analytics_tab2:
            st.markdown("### 👥 Most Active Researchers")
            
            active_authors = engine.get_most_active_authors(limit=15)
            
            if active_authors:
                df_authors = pd.DataFrame(active_authors)
                df_top10 = df_authors.head(10)
                
                st.markdown("#### 📊 Top 10 by Publication Count")
                df_chart = df_top10[['name', 'pub_count']].set_index('name')
                st.bar_chart(df_chart)
                
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
                        
                        df_trend = pd.DataFrame({
                            'Year': trend_data['years'],
                            'Publications': trend_data['counts']
                        })
                        
                        st.line_chart(df_trend.set_index('Year'))
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total Publications", trend_data['total_publications'])
                        col2.metric("Years Covered", len(trend_data['years']))
                        col3.metric("Peak Year", max(trend_data['data'].items(), key=lambda x: x[1])[0])
                        col4.metric("Peak Count", max(trend_data['counts']))
                        
                        if len(trend_data['years']) >= 3:
                            recent_avg = sum(trend_data['counts'][-3:]) / 3
                            early_avg = sum(trend_data['counts'][:3]) / 3
                            
                            if recent_avg > early_avg * 1.5:
                                st.info("📈 **Trending UP** - This topic is gaining momentum at SASTRA!")
                            elif recent_avg < early_avg * 0.5:
                                st.warning("📉 **Declining** - This topic is receiving less attention recently")
                            else:
                                st.success("➡️ **Stable** - Consistent research activity over time")
                        
                        with st.expander("📅 Year-by-Year Breakdown"):
                            for year, count in sorted(trend_data['data'].items(), reverse=True):
                                st.write(f"**{year}:** {count} publications")
                    else:
                        st.warning(f"No publications found containing the keyword '{keyword_input}'")
                        st.info("💡 Try different variations or related terms")
                else:
                    st.warning("Please enter a keyword")

    # ==================== TAB 6: THEMATIC AREAS ====================
    with tab6:
        st.subheader("🎨 Research Thematic Areas at SASTRA")
        
        theme_tab1, theme_tab2 = st.tabs([
            "📚 Single Thematic Areas",
            "🤝 Interdisciplinary Teams"
        ])
        
        # ===== THEME TAB 1: SINGLE AREAS =====
        with theme_tab1:
            st.markdown(
    "<p style='font-size: 20px; background-color: #d1ecf1; padding: 12px; border-radius: 5px; border-left: 4px solid #0c5460;'><b style='font-weight: 700;'>Click</b> the theme and <b style='font-weight: 700;'>scroll</b> to View top 10 faculty ranked by total citation score</p>", 
    unsafe_allow_html=True
)
            
            single_rankings = load_single_theme_rankings()
            
            if not single_rankings:
                st.error("❌ Run preprocessing to generate thematic_single_rankings.pkl")
            else:
                # Sort themes by total citations (descending)
                theme_scores = []
                for theme_name, faculty_list in single_rankings.items():
                    if faculty_list:
                        total_cites = sum(f['total_cite_score'] for f in faculty_list)
                        theme_scores.append((theme_name, total_cites, len(faculty_list)))
                
                theme_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Initialize session state for selected theme
                if 'selected_single_theme' not in st.session_state:
                    st.session_state.selected_single_theme = None
                
                # Display tiles in rows of 3
                for i in range(0, len(theme_scores), 3):
                    cols = st.columns(3)
                    for j, col in enumerate(cols):
                        if i + j < len(theme_scores):
                            theme_name, total_cites, faculty_count = theme_scores[i + j]
                            
                            with col:
                                if st.button(
                                    f"**{theme_name}**\n\n📊 {total_cites:,} citations\n👥 {faculty_count} faculty",
                                    key=f"single_tile_{i}_{j}",
                                    use_container_width=True
                                ):
                                    st.session_state.selected_single_theme = theme_name
                                    st.rerun()
                
                # Display selected theme details
                if st.session_state.selected_single_theme:
                    theme_name = st.session_state.selected_single_theme
                    faculty_list = single_rankings.get(theme_name, [])
                    
                    st.divider()
                    st.markdown(f"## 🎯 {theme_name}")
                    st.success(f"Found **{len(faculty_list)}** faculty")
                    
                    # Scatter plot
                    st.markdown("### 📊 Citation Distribution")
                    
                    try:
                        import plotly.graph_objects as go
                        
                        names = [f['primary_name'] for f in faculty_list]
                        cites = [f['total_cite_score'] for f in faculty_list]
                        
                        # Better sizing: square root scaling for more visual difference
                        sizes = [max(20, min((c ** 0.6) / 3, 120)) for c in cites]
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=list(range(1, len(faculty_list) + 1)),
                            y=cites,
                            mode='markers',
                            marker=dict(
                                size=sizes,
                                color=cites,
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(title="Citations"),
                                line=dict(width=2, color='white')
                            ),
                            text=names,
                            hovertemplate='<b>%{text}</b><br>Citations: %{y:,}<br><extra></extra>',
                        ))
                        
                        fig.update_layout(
                            title=f'Top {len(faculty_list)} Faculty',
                            xaxis_title='Rank',
                            yaxis_title='Citations',
                            height=500,
                            hovermode='closest'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        pass
                    
                    st.divider()
                    st.markdown("### 👥 Top 10 Faculty")
                    
                    for idx, faculty in enumerate(faculty_list, 1):
                        with st.expander(
                            f"**#{idx}** {faculty['primary_name']} - 📊 **{faculty['total_cite_score']:,} citations** | 📄 {faculty['paper_count']} papers",
                            expanded=(idx <= 3)
                        ):
                            col1, col2 = st.columns([1, 3])
                            
                            with col1:
                                st.markdown(f"**ID:** `{faculty['author_id']}`")
                                st.metric("Citations", f"{faculty['total_cite_score']:,}")
                                st.metric("Papers", faculty['paper_count'])
                                st.metric("Avg/Paper", f"{faculty['total_cite_score'] / faculty['paper_count']:.1f}")
                                
                                if st.button("📄 View Profile", key=f"prof_{idx}_{faculty['author_id']}"):
                                    st.session_state.view_mode = 'profile'
                                    st.session_state.viewing_author_id = faculty['author_id']
                                    st.session_state.viewing_author_name = None
                                    st.rerun()
                            
                            with col2:
                                st.markdown("**📚 Top Papers:**")
                                
                                for pidx, paper in enumerate(faculty['papers'][:5], 1):
                                    st.markdown(f"**{pidx}. {paper['title']}**")
                                    st.caption(f"📅 {paper['year']} | 📊 {paper['citations']:,} citations")
                                    
                                    if paper.get('abstract'):
                                        if st.checkbox(f"📋 Show Abstract", key=f"single_abs_{idx}_{pidx}"):
                                            st.markdown(f'<div class="abstract-box">{paper["abstract"]}</div>', unsafe_allow_html=True)
                                    
                                    if pidx < 5:
                                        st.markdown("---")
        
        # ===== THEME TAB 2: INTERDISCIPLINARY =====
        with theme_tab2:
            st.markdown(
    "<p style='font-size: 20px; background-color: #d1ecf1; padding: 12px; border-radius: 5px; border-left: 4px solid #0c5460;'><b style='font-weight: 700;'>Click</b> on a combination tile and <b style='font-weight: 700;'>scroll</b> down to view pre-computed teams ranked by citation impact</p>", 
    unsafe_allow_html=True
)
            
            combos = load_interdisciplinary_combinations()
            
            if not combos:
                st.warning("Run preprocessing to generate thematic_combinations.pkl")
            else:
                # Calculate average citations for each combo
                combo_scores = []
                for combo_key, combo_data in combos.items():
                    teams = combo_data['teams']
                    if teams:
                        avg_cite = sum(t['average_cite_score'] for t in teams) / len(teams)
                        combo_scores.append((combo_key, avg_cite, len(teams)))
                
                combo_scores.sort(key=lambda x: x[1], reverse=True)
                
                if 'selected_combo' not in st.session_state:
                    st.session_state.selected_combo = None
                
                # Display combo tiles in rows of 2
                for i in range(0, len(combo_scores), 2):
                    cols = st.columns(2)
                    for j, col in enumerate(cols):
                        if i + j < len(combo_scores):
                            combo_key, avg_cite, team_count = combo_scores[i + j]
                            
                            with col:
                                if st.button(
                                    f"**{combo_key}**\n\n📊 Avg: {avg_cite:,.1f} citations/member\n👥 {team_count} teams",
                                    key=f"combo_tile_{i}_{j}",
                                    use_container_width=True
                                ):
                                    st.session_state.selected_combo = combo_key
                                    st.rerun()
                
                # Display selected combo
                if st.session_state.selected_combo:
                    combo_key = st.session_state.selected_combo
                    combo_data = combos[combo_key]
                    themes = combo_data['themes']
                    teams = combo_data['teams']
                    
                    st.divider()
                    st.markdown(f"## 👥 {combo_key}")
                    st.success(f"**{len(teams)} teams** from: **{', '.join(themes)}**")
                    
                    for team in teams:
                        tn = team['team_number']
                        ts = team['total_cite_score']
                        avg = team['average_cite_score']
                        
                        with st.expander(
                            f"**Team {tn}** - 📊 **{ts:,} citations** | Avg: {avg:,.1f}",
                            expanded=(tn <= 2)
                        ):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Members", len(team['members']))
                            with col2:
                                st.metric("Total Citations", f"{ts:,}")
                            with col3:
                                st.metric("Avg/Member", f"{avg:,.1f}")
                            
                            st.markdown("---")
                            
                            for midx, member in enumerate(team['members'], 1):
                                st.markdown(f"#### Member {midx}: {member['name']}")
                                st.caption(f"**Expertise:** {member['theme']}")
                                
                                col1, col2 = st.columns([1, 3])
                                
                                with col1:
                                    st.markdown(f"**ID:** `{member['author_id']}`")
                                    st.metric("Citations", f"{member['cite_score']:,}")
                                    st.metric("Papers", member['paper_count'])
                                    
                                    if st.button("📄 Profile", key=f"t{tn}m{midx}_{member['author_id']}"):
                                        st.session_state.view_mode = 'profile'
                                        st.session_state.viewing_author_id = member['author_id']
                                        st.session_state.viewing_author_name = None
                                        st.rerun()
                                
                                with col2:
                                    st.markdown(f"**📚 Top Papers in {member['theme']}:**")
                                    
                                    for pidx, paper in enumerate(member['papers'], 1):
                                        st.markdown(f"**{pidx}. {paper['title']}**")
                                        st.caption(f"📅 {paper['year']} | 📊 {paper['citations']:,} citations")
                                        
                                        if paper.get('abstract'):
                                            if st.checkbox(f"📋 Show Abstract", key=f"combo_abs_t{tn}m{midx}p{pidx}"):
                                                st.markdown(f'<div class="abstract-box">{paper["abstract"]}</div>', unsafe_allow_html=True)
                                        
                                        if pidx < len(member['papers']):
                                            st.markdown("---")
                                
                                if midx < len(team['members']):
                                    st.markdown("---")

    st.divider()
    st.caption("SASTRA Research Finder | Phase 1 + Phase 2")


if __name__ == "__main__":
    main()