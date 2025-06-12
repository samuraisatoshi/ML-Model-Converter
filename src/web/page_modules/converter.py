import streamlit as st
import tempfile
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.entities.model_info import ModelInfo
from core.entities.conversion_config import ConversionConfig
from core.enums.model_types import ModelType
from services.conversion_service import ConversionService
from services.validation_service import ValidationService
from infrastructure.storage.local_storage import LocalStorageService
from infrastructure.logging.file_logger import CombinedLogger


def render_converter():
    """Render the model converter page."""
    st.title("🔄 Model Converter")
    st.markdown("Convert your machine learning models to TensorFlow Lite format")
    
    # Initialize services
    if 'services_initialized' not in st.session_state:
        st.session_state.logger = CombinedLogger()
        st.session_state.storage = LocalStorageService()
        st.session_state.validation = ValidationService(st.session_state.logger)
        st.session_state.conversion_service = ConversionService(
            st.session_state.storage,
            st.session_state.logger,
            st.session_state.validation
        )
        st.session_state.services_initialized = True
    
    # File upload section
    st.markdown("### 📁 Upload Model")
    uploaded_file = st.file_uploader(
        "Choose a model file",
        type=['pth', 'pt', 'pb', 'h5', 'keras', 'onnx'],
        help="Supported formats: PyTorch (.pth, .pt), TensorFlow (.pb), Keras (.h5, .keras), ONNX (.onnx)"
    )
    
    if uploaded_file is not None:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = Path(tmp_file.name)
        
        # Display file info
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        file_size = temp_path.stat().st_size
        st.info(f"📊 File size: {file_size / (1024*1024):.2f} MB")
        
        # Model configuration section
        st.markdown("### ⚙️ Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Model type (auto-detected)
            try:
                model_type = ModelType.from_extension(uploaded_file.name)
                st.selectbox(
                    "Model Type", 
                    [model_type.value], 
                    disabled=True,
                    help="Automatically detected from file extension"
                )
            except ValueError as e:
                st.error(f"❌ {str(e)}")
                return
            
            # Input shape
            st.markdown("**Input Shape**")
            batch_size = st.number_input("Batch Size", value=1, min_value=1)
            
            if model_type in [ModelType.PYTORCH, ModelType.TENSORFLOW, ModelType.KERAS]:
                channels = st.number_input("Channels", value=3, min_value=1)
                height = st.number_input("Height", value=224, min_value=1)
                width = st.number_input("Width", value=224, min_value=1)
                input_shape = (batch_size, channels, height, width)
            else:
                # For ONNX, allow custom input shape
                shape_str = st.text_input("Input Shape", value="1,3,224,224", help="Comma-separated dimensions")
                try:
                    input_shape = tuple(map(int, shape_str.split(',')))
                except ValueError:
                    st.error("❌ Invalid input shape format")
                    return
        
        with col2:
            # Output configuration
            output_filename = st.text_input(
                "Output Filename", 
                value=f"{Path(uploaded_file.name).stem}_converted.tflite"
            )
            
            # Additional metadata for PyTorch models
            model_metadata = {}
            if model_type == ModelType.PYTORCH:
                with st.expander("🐍 PyTorch Model Information", expanded=False):
                    st.info("""
                    ✨ **Auto-Detection Enabled**: Our converter automatically handles most PyTorch model formats:
                    
                    **✅ Fully Supported (No extra steps needed):**
                    - Complete models saved with `torch.save(model, 'model.pth')`
                    - Models saved with `{'model': model}` dictionary format
                    - Common architectures (ResNet, VGG, MobileNet, EfficientNet, etc.)
                    - Models with architecture metadata in checkpoint
                    
                    **⚠️ May need assistance:**
                    - Custom architecture state_dict files
                    - Models with unusual layer naming
                    """)
                    
                    st.markdown("**💡 For best results:**")
                    st.markdown("- Use `torch.save(model, 'path.pth')` when saving PyTorch models")
                    st.markdown("- Include architecture info: `torch.save({'model': model, 'arch': 'resnet50'}, 'path.pth')`")
                    st.markdown("- The converter will try multiple strategies to load your model automatically")
                    
                    # Advanced option for custom model class (if needed)
                    if st.checkbox("🔧 Advanced: I have a custom model class", help="Only needed for very custom architectures"):
                        st.text_area(
                            "Model Class Code",
                            placeholder="# Only needed for custom architectures that can't be auto-detected\nclass MyCustomModel(nn.Module):\n    def __init__(self):\n        # Your model definition here\n        pass",
                            help="Paste your PyTorch model class definition here",
                            height=100
                        )
                        st.info("💡 Try the conversion first - the auto-detection works for most models!")
        
        # Conversion settings section
        st.markdown("### 🔧 Conversion Settings")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            optimization_level = st.selectbox(
                "Optimization Level",
                ["none", "default", "aggressive"],
                index=1,
                help="Higher levels may reduce model size but could affect accuracy"
            )
            
            quantization = st.checkbox(
                "Enable Quantization", 
                value=True,
                help="Reduces model size, may slightly affect accuracy"
            )
        
        with col2:
            inference_type = st.selectbox(
                "Inference Type",
                ["FLOAT32", "FLOAT16", "INT8"],
                index=0,
                help="INT8 provides smallest size but requires representative dataset"
            )
            
            allow_custom_ops = st.checkbox(
                "Allow Custom Ops",
                value=False,
                help="Enable if your model uses custom operations"
            )
        
        with col3:
            experimental_converter = st.checkbox(
                "Use Experimental Converter",
                value=True,
                help="Use TensorFlow's experimental converter (recommended)"
            )
        
        # Representative dataset for INT8 quantization
        if inference_type == "INT8":
            st.warning("⚠️ INT8 quantization requires a representative dataset. This feature is not yet implemented in the web interface.")
        
        # Convert button
        st.markdown("---")
        
        if st.button("🚀 Convert Model", type="primary", use_container_width=True):
            try:
                # Create ModelInfo
                model_info = ModelInfo(
                    file_path=temp_path,
                    model_type=model_type,
                    input_shape=input_shape,
                    model_size=file_size,
                    metadata=model_metadata
                )
                
                # Create ConversionConfig
                output_path = Path("./outputs/converted") / output_filename
                config = ConversionConfig(
                    output_path=output_path,
                    optimization_level=optimization_level,
                    quantization=quantization,
                    inference_type=inference_type,
                    allow_custom_ops=allow_custom_ops,
                    experimental_new_converter=experimental_converter
                )
                
                # Show progress
                with st.spinner("Converting model... This may take a few minutes."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("🔍 Validating input...")
                    progress_bar.progress(20)
                    
                    status_text.text("🔄 Starting conversion...")
                    progress_bar.progress(40)
                    
                    # Perform conversion
                    result = st.session_state.conversion_service.convert_model(model_info, config)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Conversion completed!")
                
                # Display results
                if result.is_successful:
                    st.success("🎉 Conversion completed successfully!")
                    
                    # Show conversion statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Original Size", f"{result.original_size_mb:.2f} MB")
                    with col2:
                        st.metric("Converted Size", f"{result.converted_size_mb:.2f} MB")
                    with col3:
                        st.metric("Size Reduction", f"{result.size_reduction_percent:.1f}%")
                    with col4:
                        st.metric("Conversion Time", f"{result.execution_time:.2f}s")
                    
                    # Download button
                    if result.output_path.exists():
                        with open(result.output_path, 'rb') as f:
                            st.download_button(
                                label="📥 Download Converted Model",
                                data=f.read(),
                                file_name=output_filename,
                                mime="application/octet-stream",
                                use_container_width=True
                            )
                    
                    # Show conversion details
                    with st.expander("📊 Conversion Details"):
                        st.json({
                            "input_model": {
                                "filename": uploaded_file.name,
                                "type": model_type.value,
                                "size_mb": result.original_size_mb,
                                "input_shape": input_shape
                            },
                            "output_model": {
                                "filename": output_filename,
                                "size_mb": result.converted_size_mb,
                                "compression_ratio": result.compression_ratio
                            },
                            "settings": {
                                "optimization_level": optimization_level,
                                "quantization": quantization,
                                "inference_type": inference_type,
                                "execution_time": result.execution_time
                            }
                        })
                
                else:
                    st.error(f"❌ Conversion failed: {result.error_message}")
                    
                    with st.expander("🔍 Error Details"):
                        st.text(result.error_message)
                        
                        # Provide specific guidance based on error type and model type
                        if "model structure" in result.error_message.lower() and model_type == ModelType.PYTORCH:
                            st.markdown("### 🐍 PyTorch Model Loading Solutions:")
                            st.markdown("""
                            **The converter couldn't automatically detect your model architecture. Try these solutions:**
                            
                            1. **🔄 Re-save your model as a complete model:**
                               ```python
                               # Instead of: torch.save(model.state_dict(), 'model.pth')
                               torch.save(model, 'model.pth')  # Save complete model
                               ```
                            
                            2. **📝 Add architecture information:**
                               ```python
                               torch.save({
                                   'model': model,
                                   'arch': 'resnet50',  # or your architecture name
                                   'state_dict': model.state_dict()
                               }, 'model.pth')
                               ```
                            
                            3. **🎯 Use a common architecture:**
                               - If your model is based on ResNet, VGG, MobileNet, etc., the converter can often detect it
                               - Consider fine-tuning from a pre-trained model instead of training from scratch
                            
                            4. **🛠️ Check the Advanced settings:**
                               - Use the "I have a custom model class" option above
                               - Provide your model class definition
                            """)
                        else:
                            st.markdown("**General troubleshooting:**")
                            st.markdown("- Check if the model file is valid and not corrupted")
                            st.markdown("- Verify input shape matches your model's expected input")
                            st.markdown("- Try different optimization settings (disable quantization, use 'none' optimization)")
                            st.markdown("- Ensure your model doesn't use unsupported operations")
                        
                        if model_type == ModelType.PYTORCH:
                            st.info("💡 **Pro tip:** The converter works best with complete PyTorch models saved using `torch.save(model, path)` rather than just the state_dict.")
                
                # Cleanup temporary file
                temp_path.unlink(missing_ok=True)
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                temp_path.unlink(missing_ok=True)
    
    else:
        # Show supported formats info
        st.info("👆 Please upload a model file to get started")
        
        with st.expander("📋 Supported Model Formats"):
            st.markdown("""
            | Framework | Extensions | Notes |
            |-----------|------------|-------|
            | PyTorch   | .pth, .pt  | State dict or complete models |
            | TensorFlow| .pb        | Frozen graphs, SavedModel |
            | Keras     | .h5, .keras| Sequential and Functional models |
            | ONNX      | .onnx      | Cross-platform models |
            """)
            
        with st.expander("💡 Conversion Tips"):
            st.markdown("""
            - **Input Shape**: Make sure to specify the correct input dimensions
            - **Quantization**: Enable for smaller models, disable for maximum accuracy
            - **Optimization**: Use 'default' for balanced performance
            - **PyTorch Models**: Complete saved models work best
            - **Large Models**: May take several minutes to convert
            """)