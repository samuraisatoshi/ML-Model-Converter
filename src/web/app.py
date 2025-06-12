#!/usr/bin/env python3
"""
ML Model Converter - Web Application
====================================

A professional tool for converting machine learning models to TensorFlow Lite format.
Built with SOLID principles and modern web technologies.

Author: SamuraiSatoshi (in collaboration with Claude)
License: CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
Repository: https://github.com/samuraisatoshi/ML-Model-Converter

Copyright (c) 2024 SamuraiSatoshi
This work is licensed under Creative Commons Attribution-NonCommercial 4.0 International.
For commercial licensing, contact: samuraisatoshi@cryptoworld.8shield.net
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from web.page_modules.home import render_home
from web.page_modules.converter import render_converter
from web.page_modules.history import render_history
from web.page_modules.settings import render_settings
from web.page_modules.donations import render_donations


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="ML Model Converter",
        page_icon="🔄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Hide only specific Streamlit UI elements
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    [data-testid="stToolbar"] {display: none;}
    .stException {display: none;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        padding: 1rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    /* Style for navigation selectbox */
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 2px solid #1f77b4;
        background-color: #f8f9fa;
    }
    .stSelectbox > div > div > div {
        color: #333;
        font-weight: 500;
        padding: 8px 12px;
    }
    .stSelectbox [data-baseweb="select"] {
        border-radius: 8px;
    }
    .stSelectbox [data-baseweb="select"]:hover {
        border-color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar content
    with st.sidebar:
        st.markdown("# 🔄 ML Model Converter")
        st.markdown("---")
        
        # Navigation
        st.markdown("### 📍 Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["🏠 Home", "🔄 Converter", "📊 History", "⚙️ Settings", "💰 Donations"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### 📋 Supported Formats")
        st.markdown("- **PyTorch** (.pth, .pt)")
        st.markdown("- **TensorFlow** (.pb)")
        st.markdown("- **Keras** (.h5)")
        st.markdown("- **ONNX** (.onnx)")
        
        st.markdown("---")
        st.markdown("### 💝 Support This Project")
        st.markdown("⭐ [Star on GitHub](https://github.com)")
        st.markdown("🐛 [Report Issues](https://github.com)")
        st.markdown("💰 **Crypto donations** - See Donations page!")
        
        st.markdown("---")
        st.markdown("*Made with ❤️ for the ML community*")
    
    # Render selected page
    if page == "🏠 Home":
        render_home()
    elif page == "🔄 Converter":
        render_converter()
    elif page == "📊 History":
        render_history()
    elif page == "⚙️ Settings":
        render_settings()
    elif page == "💰 Donations":
        render_donations()


if __name__ == "__main__":
    main()