import streamlit as st


def render_home():
    """Render the home page."""
    st.markdown('<h1 class="main-header">🔄 ML Model Converter</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Welcome to the ML Model Converter!
    
    This tool provides a robust, extensible solution for converting machine learning models 
    from various frameworks to **TensorFlow Lite (TFLite)** format.
    
    ### ✨ Features
    
    - **Multi-Framework Support**: Convert from PyTorch, TensorFlow, Keras, and ONNX
    - **SOLID Architecture**: Clean, maintainable, and extensible codebase
    - **Web Interface**: Easy-to-use Streamlit interface
    - **CLI Support**: Command-line interface for automation
    - **Conversion History**: Track all your conversions
    - **Model Optimization**: Various optimization levels and quantization options
    
    ### 🚀 Quick Start
    
    1. **Upload Model**: Go to the Converter page and upload your model file
    2. **Configure**: Set your conversion preferences
    3. **Convert**: Click convert and download your TFLite model
    4. **Track**: View conversion history and manage files
    
    ### 📊 Supported Conversions
    """)
    
    # Create conversion flow diagram
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        #### PyTorch
        - `.pth` files
        - `.pt` files
        - State dict or full models
        """)
    
    with col2:
        st.markdown("""
        #### TensorFlow
        - `.pb` files
        - SavedModel format
        - Frozen graphs
        """)
    
    with col3:
        st.markdown("""
        #### Keras
        - `.h5` files
        - `.keras` files
        - Sequential & Functional models
        """)
    
    with col4:
        st.markdown("""
        #### ONNX
        - `.onnx` files
        - All ONNX versions
        - Cross-platform models
        """)
    
    st.markdown("---")
    
    # Architecture overview
    with st.expander("🏗️ Architecture Overview"):
        st.markdown("""
        ### SOLID Principles Implementation
        
        - **Single Responsibility**: Each class has one clear purpose
        - **Open/Closed**: Extensible for new model formats
        - **Liskov Substitution**: Interchangeable converter implementations
        - **Interface Segregation**: Specific interfaces for each responsibility
        - **Dependency Inversion**: High-level modules depend on abstractions
        
        ### Key Components
        
        - **Core Layer**: Interfaces, entities, and business logic
        - **Converters**: Model-specific conversion implementations
        - **Services**: Application orchestration and coordination
        - **Infrastructure**: Storage, logging, and configuration
        - **Web Interface**: This Streamlit application
        """)
    
    # Getting started
    with st.expander("📖 Getting Started Guide"):
        st.markdown("""
        ### Prerequisites
        
        - Python 3.11+
        - Required packages (automatically installed)
        
        ### Basic Usage
        
        1. **Prepare Your Model**
           - Ensure your model file is in a supported format
           - Know your model's input shape
           - Have any required model class definitions
        
        2. **Use the Converter**
           - Navigate to the Converter page
           - Upload your model file
           - Configure conversion settings
           - Start the conversion process
        
        3. **Download Results**
           - Download the converted TFLite model
           - View conversion statistics
           - Check the conversion history
        
        ### Tips for Best Results
        
        - Use quantization for smaller model sizes
        - Specify correct input shapes
        - Provide representative datasets for INT8 quantization
        - Check conversion logs for any warnings
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Built with ❤️ using Streamlit and following SOLID principles</p>
        <p>Ready to convert your models? Head to the <strong>Converter</strong> page!</p>
    </div>
    """, unsafe_allow_html=True)