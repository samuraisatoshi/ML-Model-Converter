import streamlit as st
import yaml
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from infrastructure.storage.local_storage import LocalStorageService
from infrastructure.logging.file_logger import CombinedLogger


def render_settings():
    """Render the settings page."""
    st.title("⚙️ Settings")
    st.markdown("Configure your ML Model Converter preferences")
    
    # Initialize services
    if 'settings_services_initialized' not in st.session_state:
        st.session_state.settings_logger = CombinedLogger()
        st.session_state.settings_storage = LocalStorageService()
        st.session_state.settings_services_initialized = True
    
    # Default settings
    default_settings = {
        "conversion": {
            "default_optimization_level": "default",
            "default_quantization": True,
            "default_inference_type": "FLOAT32",
            "allow_custom_ops": False,
            "experimental_converter": True,
            "max_conversion_time": 600  # seconds
        },
        "storage": {
            "output_directory": "./outputs/converted",
            "temp_directory": "./outputs/temp",
            "logs_directory": "./outputs/logs",
            "auto_cleanup_temp": True,
            "max_history_entries": 1000
        },
        "interface": {
            "theme": "auto",
            "show_advanced_options": False,
            "enable_tooltips": True,
            "default_batch_size": 1,
            "default_image_size": [224, 224, 3]
        },
        "logging": {
            "log_level": "INFO",
            "console_logging": True,
            "file_logging": True,
            "max_log_size_mb": 10,
            "log_backup_count": 5
        }
    }
    
    # Load existing settings
    settings_file = Path("./config/app_config.yaml")
    if settings_file.exists():
        try:
            with open(settings_file, 'r') as f:
                current_settings = yaml.safe_load(f) or default_settings
        except Exception:
            current_settings = default_settings
    else:
        current_settings = default_settings
    
    # Settings tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔄 Conversion", "🗂️ Storage", "🎨 Interface", "📝 Logging"])
    
    with tab1:
        st.markdown("### Conversion Settings")
        st.markdown("Configure default behavior for model conversions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            default_optimization = st.selectbox(
                "Default Optimization Level",
                ["none", "default", "aggressive"],
                index=["none", "default", "aggressive"].index(
                    current_settings["conversion"]["default_optimization_level"]
                ),
                help="Default optimization level for new conversions"
            )
            
            default_quantization = st.checkbox(
                "Enable Quantization by Default",
                value=current_settings["conversion"]["default_quantization"],
                help="Whether to enable quantization by default"
            )
            
            default_inference = st.selectbox(
                "Default Inference Type",
                ["FLOAT32", "FLOAT16", "INT8"],
                index=["FLOAT32", "FLOAT16", "INT8"].index(
                    current_settings["conversion"]["default_inference_type"]
                ),
                help="Default precision for converted models"
            )
        
        with col2:
            allow_custom_ops = st.checkbox(
                "Allow Custom Operations",
                value=current_settings["conversion"]["allow_custom_ops"],
                help="Allow models with custom operations"
            )
            
            experimental_converter = st.checkbox(
                "Use Experimental Converter",
                value=current_settings["conversion"]["experimental_converter"],
                help="Use TensorFlow's experimental converter features"
            )
            
            max_conversion_time = st.number_input(
                "Max Conversion Time (seconds)",
                min_value=60,
                max_value=3600,
                value=current_settings["conversion"]["max_conversion_time"],
                help="Maximum time allowed for a single conversion"
            )
        
        # Update conversion settings
        current_settings["conversion"].update({
            "default_optimization_level": default_optimization,
            "default_quantization": default_quantization,
            "default_inference_type": default_inference,
            "allow_custom_ops": allow_custom_ops,
            "experimental_converter": experimental_converter,
            "max_conversion_time": max_conversion_time
        })
    
    with tab2:
        st.markdown("### Storage Settings")
        st.markdown("Configure file storage and management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            output_dir = st.text_input(
                "Output Directory",
                value=current_settings["storage"]["output_directory"],
                help="Directory for converted models"
            )
            
            temp_dir = st.text_input(
                "Temporary Directory",
                value=current_settings["storage"]["temp_directory"],
                help="Directory for temporary files during conversion"
            )
            
            logs_dir = st.text_input(
                "Logs Directory",
                value=current_settings["storage"]["logs_directory"],
                help="Directory for log files"
            )
        
        with col2:
            auto_cleanup = st.checkbox(
                "Auto-cleanup Temporary Files",
                value=current_settings["storage"]["auto_cleanup_temp"],
                help="Automatically clean temporary files after conversion"
            )
            
            max_history = st.number_input(
                "Max History Entries",
                min_value=100,
                max_value=10000,
                value=current_settings["storage"]["max_history_entries"],
                help="Maximum number of conversion history entries to keep"
            )
        
        # Storage statistics
        st.markdown("### Current Storage Usage")
        try:
            stats = st.session_state.settings_storage.get_storage_stats()
            if stats:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Converted Files", stats.get('total_converted_files', 0))
                with col2:
                    st.metric("Temp Files", stats.get('total_temp_files', 0))
                with col3:
                    st.metric("Log Files", stats.get('total_log_files', 0))
        except Exception as e:
            st.warning(f"Could not load storage statistics: {str(e)}")
        
        # Update storage settings
        current_settings["storage"].update({
            "output_directory": output_dir,
            "temp_directory": temp_dir,
            "logs_directory": logs_dir,
            "auto_cleanup_temp": auto_cleanup,
            "max_history_entries": max_history
        })
    
    with tab3:
        st.markdown("### Interface Settings")
        st.markdown("Customize the user interface behavior")
        
        col1, col2 = st.columns(2)
        
        with col1:
            show_advanced = st.checkbox(
                "Show Advanced Options by Default",
                value=current_settings["interface"]["show_advanced_options"],
                help="Display advanced conversion options by default"
            )
            
            enable_tooltips = st.checkbox(
                "Enable Tooltips",
                value=current_settings["interface"]["enable_tooltips"],
                help="Show helpful tooltips throughout the interface"
            )
            
            default_batch_size = st.number_input(
                "Default Batch Size",
                min_value=1,
                max_value=32,
                value=current_settings["interface"]["default_batch_size"],
                help="Default batch size for model inputs"
            )
        
        with col2:
            st.markdown("**Default Image Dimensions**")
            
            height = st.number_input(
                "Height",
                min_value=1,
                max_value=2048,
                value=current_settings["interface"]["default_image_size"][0],
                help="Default image height"
            )
            
            width = st.number_input(
                "Width",
                min_value=1,
                max_value=2048,
                value=current_settings["interface"]["default_image_size"][1],
                help="Default image width"
            )
            
            channels = st.number_input(
                "Channels",
                min_value=1,
                max_value=4,
                value=current_settings["interface"]["default_image_size"][2],
                help="Default number of channels (1=grayscale, 3=RGB, 4=RGBA)"
            )
        
        # Update interface settings
        current_settings["interface"].update({
            "show_advanced_options": show_advanced,
            "enable_tooltips": enable_tooltips,
            "default_batch_size": default_batch_size,
            "default_image_size": [height, width, channels]
        })
    
    with tab4:
        st.markdown("### Logging Settings")
        st.markdown("Configure logging behavior and verbosity")
        
        col1, col2 = st.columns(2)
        
        with col1:
            log_level = st.selectbox(
                "Log Level",
                ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                index=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"].index(
                    current_settings["logging"]["log_level"]
                ),
                help="Minimum severity level for logged messages"
            )
            
            console_logging = st.checkbox(
                "Enable Console Logging",
                value=current_settings["logging"]["console_logging"],
                help="Show log messages in the console"
            )
            
            file_logging = st.checkbox(
                "Enable File Logging",
                value=current_settings["logging"]["file_logging"],
                help="Save log messages to files"
            )
        
        with col2:
            max_log_size = st.number_input(
                "Max Log File Size (MB)",
                min_value=1,
                max_value=100,
                value=current_settings["logging"]["max_log_size_mb"],
                help="Maximum size of individual log files"
            )
            
            log_backup_count = st.number_input(
                "Log Backup Count",
                min_value=1,
                max_value=20,
                value=current_settings["logging"]["log_backup_count"],
                help="Number of backup log files to keep"
            )
        
        # Log level descriptions
        with st.expander("📖 Log Level Guide"):
            st.markdown("""
            - **DEBUG**: Detailed information for debugging
            - **INFO**: General information about program execution
            - **WARNING**: Warning messages for potentially harmful situations
            - **ERROR**: Error messages for serious problems
            - **CRITICAL**: Critical errors that may cause the program to stop
            """)
        
        # Update logging settings
        current_settings["logging"].update({
            "log_level": log_level,
            "console_logging": console_logging,
            "file_logging": file_logging,
            "max_log_size_mb": max_log_size,
            "log_backup_count": log_backup_count
        })
    
    # Save settings section
    st.markdown("---")
    st.markdown("### 💾 Save Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save Settings", type="primary", use_container_width=True):
            try:
                # Ensure config directory exists
                config_dir = Path("./config")
                config_dir.mkdir(exist_ok=True)
                
                # Save settings to YAML file
                with open(settings_file, 'w') as f:
                    yaml.dump(current_settings, f, default_flow_style=False, indent=2)
                
                st.success("✅ Settings saved successfully!")
                st.balloons()
            except Exception as e:
                st.error(f"❌ Failed to save settings: {str(e)}")
    
    with col2:
        if st.button("🔄 Reset to Defaults", use_container_width=True):
            # Confirm reset
            if st.session_state.get('confirm_reset', False):
                try:
                    with open(settings_file, 'w') as f:
                        yaml.dump(default_settings, f, default_flow_style=False, indent=2)
                    st.success("✅ Settings reset to defaults!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to reset settings: {str(e)}")
                finally:
                    st.session_state.confirm_reset = False
            else:
                st.session_state.confirm_reset = True
                st.warning("⚠️ Click again to confirm reset to defaults")
    
    with col3:
        # Export settings
        if st.button("📥 Export Settings", use_container_width=True):
            settings_json = json.dumps(current_settings, indent=2)
            st.download_button(
                "Download JSON",
                settings_json,
                file_name=f"converter_settings_{st.session_state.get('export_timestamp', 'export')}.json",
                mime="application/json",
                use_container_width=True
            )
    
    # Import settings
    st.markdown("### 📤 Import Settings")
    uploaded_settings = st.file_uploader(
        "Upload settings file",
        type=['json', 'yaml', 'yml'],
        help="Upload a previously exported settings file"
    )
    
    if uploaded_settings is not None:
        try:
            if uploaded_settings.name.endswith('.json'):
                imported_settings = json.load(uploaded_settings)
            else:
                imported_settings = yaml.safe_load(uploaded_settings)
            
            st.success("✅ Settings file loaded successfully!")
            
            if st.button("Apply Imported Settings"):
                try:
                    with open(settings_file, 'w') as f:
                        yaml.dump(imported_settings, f, default_flow_style=False, indent=2)
                    st.success("✅ Imported settings applied!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to apply imported settings: {str(e)}")
            
            # Preview imported settings
            with st.expander("👀 Preview Imported Settings"):
                st.json(imported_settings)
                
        except Exception as e:
            st.error(f"❌ Failed to parse settings file: {str(e)}")
    
    # Current settings preview
    with st.expander("👀 Current Settings Preview"):
        st.json(current_settings)