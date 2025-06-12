import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from infrastructure.storage.local_storage import LocalStorageService
from infrastructure.logging.file_logger import CombinedLogger


def render_history():
    """Render the conversion history page."""
    st.title("📊 Conversion History")
    st.markdown("Track and analyze your model conversion history")
    
    # Initialize services
    if 'history_services_initialized' not in st.session_state:
        st.session_state.history_logger = CombinedLogger()
        st.session_state.history_storage = LocalStorageService()
        st.session_state.history_services_initialized = True
    
    # Load conversion history
    try:
        history = st.session_state.history_storage.load_conversion_history()
    except Exception as e:
        st.error(f"❌ Failed to load conversion history: {str(e)}")
        return
    
    if not history:
        st.info("📝 No conversion history found. Convert some models to see them here!")
        
        # Show sample data structure
        with st.expander("📋 What you'll see here"):
            st.markdown("""
            Once you start converting models, you'll see:
            
            - **Conversion Statistics**: Success rates, processing times, file sizes
            - **Model Analytics**: Charts showing conversion trends over time
            - **Detailed History**: Searchable table of all conversions
            - **Storage Management**: File cleanup and organization tools
            """)
        return
    
    # Summary metrics
    st.markdown("### 📈 Summary Statistics")
    
    # Calculate metrics
    total_conversions = len(history)
    successful_conversions = len([h for h in history if h.is_successful])
    failed_conversions = total_conversions - successful_conversions
    success_rate = (successful_conversions / total_conversions * 100) if total_conversions > 0 else 0
    
    total_original_size = sum(h.original_size for h in history) / (1024 * 1024)  # MB
    total_converted_size = sum(h.converted_size for h in history if h.is_successful) / (1024 * 1024)  # MB
    avg_compression = (1 - total_converted_size / total_original_size) * 100 if total_original_size > 0 else 0
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Conversions", 
            total_conversions,
            help="Total number of conversion attempts"
        )
    
    with col2:
        st.metric(
            "Success Rate", 
            f"{success_rate:.1f}%",
            delta=f"{successful_conversions} successful",
            help="Percentage of successful conversions"
        )
    
    with col3:
        st.metric(
            "Total Data Processed", 
            f"{total_original_size:.1f} MB",
            help="Total size of original models processed"
        )
    
    with col4:
        st.metric(
            "Average Compression", 
            f"{avg_compression:.1f}%",
            help="Average size reduction achieved"
        )
    
    # Charts section
    st.markdown("### 📊 Analytics")
    
    tab1, tab2, tab3 = st.tabs(["Conversion Trends", "Size Analysis", "Performance"])
    
    with tab1:
        # Conversion trends over time
        df = pd.DataFrame([
            {
                'date': h.timestamp.date(),
                'time': h.timestamp,
                'status': h.status.value,
                'success': h.is_successful,
                'execution_time': h.execution_time
            }
            for h in history
        ])
        
        if len(df) > 0:
            # Daily conversion counts
            daily_counts = df.groupby(['date', 'status']).size().reset_index(name='count')
            
            if len(daily_counts) > 0:
                fig = px.bar(
                    daily_counts, 
                    x='date', 
                    y='count', 
                    color='status',
                    title="Daily Conversion Activity",
                    color_discrete_map={
                        'completed': '#2E8B57',
                        'failed': '#DC143C',
                        'pending': '#FFD700'
                    }
                )
                fig.update_layout(xaxis_title="Date", yaxis_title="Number of Conversions")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Size analysis
        if successful_conversions > 0:
            size_df = pd.DataFrame([
                {
                    'original_size_mb': h.original_size_mb,
                    'converted_size_mb': h.converted_size_mb,
                    'compression_ratio': h.compression_ratio,
                    'size_reduction_percent': h.size_reduction_percent,
                    'filename': Path(h.output_path).name
                }
                for h in history if h.is_successful
            ])
            
            # Size reduction scatter plot
            fig = px.scatter(
                size_df,
                x='original_size_mb',
                y='converted_size_mb',
                size='size_reduction_percent',
                hover_data=['filename', 'compression_ratio'],
                title="Model Size: Original vs Converted",
                labels={
                    'original_size_mb': 'Original Size (MB)',
                    'converted_size_mb': 'Converted Size (MB)'
                }
            )
            fig.add_trace(go.Scatter(
                x=[0, size_df['original_size_mb'].max()],
                y=[0, size_df['original_size_mb'].max()],
                mode='lines',
                name='No Compression',
                line=dict(dash='dash', color='red')
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            # Compression distribution
            fig2 = px.histogram(
                size_df,
                x='size_reduction_percent',
                title="Distribution of Size Reduction",
                labels={'size_reduction_percent': 'Size Reduction (%)'},
                nbins=20
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        # Performance analysis
        if len(df) > 0:
            perf_df = df[df['success'] == True].copy()
            
            if len(perf_df) > 0:
                # Execution time distribution
                fig = px.histogram(
                    perf_df,
                    x='execution_time',
                    title="Conversion Time Distribution",
                    labels={'execution_time': 'Execution Time (seconds)'},
                    nbins=15
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Time", f"{perf_df['execution_time'].mean():.2f}s")
                with col2:
                    st.metric("Fastest Conversion", f"{perf_df['execution_time'].min():.2f}s")
                with col3:
                    st.metric("Slowest Conversion", f"{perf_df['execution_time'].max():.2f}s")
    
    # Detailed history table
    st.markdown("### 📋 Detailed History")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_filter = st.selectbox(
            "Filter by Status",
            ["All", "Completed", "Failed"],
            index=0
        )
    
    with col2:
        date_range = st.selectbox(
            "Time Range",
            ["All", "Last 7 days", "Last 30 days", "Last 90 days"],
            index=0
        )
    
    with col3:
        search_term = st.text_input("Search in filenames", placeholder="Enter filename...")
    
    # Apply filters
    filtered_history = history.copy()
    
    if status_filter != "All":
        if status_filter == "Completed":
            filtered_history = [h for h in filtered_history if h.is_successful]
        elif status_filter == "Failed":
            filtered_history = [h for h in filtered_history if not h.is_successful]
    
    if date_range != "All":
        days_map = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90}
        cutoff_date = datetime.now() - timedelta(days=days_map[date_range])
        filtered_history = [h for h in filtered_history if h.timestamp >= cutoff_date]
    
    if search_term:
        filtered_history = [h for h in filtered_history if search_term.lower() in Path(h.output_path).name.lower()]
    
    # Create table data
    if filtered_history:
        table_data = []
        for h in filtered_history:
            table_data.append({
                "Timestamp": h.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "Filename": Path(h.output_path).name,
                "Status": "✅ Success" if h.is_successful else "❌ Failed",
                "Original Size": f"{h.original_size_mb:.2f} MB",
                "Converted Size": f"{h.converted_size_mb:.2f} MB" if h.is_successful else "N/A",
                "Compression": f"{h.size_reduction_percent:.1f}%" if h.is_successful else "N/A",
                "Time": f"{h.execution_time:.2f}s",
                "Error": h.error_message if h.error_message else "None"
            })
        
        df_table = pd.DataFrame(table_data)
        
        # Display table with pagination
        st.dataframe(
            df_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Status": st.column_config.TextColumn(width="small"),
                "Original Size": st.column_config.TextColumn(width="small"),
                "Converted Size": st.column_config.TextColumn(width="small"),
                "Compression": st.column_config.TextColumn(width="small"),
                "Time": st.column_config.TextColumn(width="small"),
            }
        )
        
        st.info(f"Showing {len(filtered_history)} of {total_conversions} conversions")
    else:
        st.info("No conversions match the current filters.")
    
    # Storage management
    st.markdown("### 🗂️ Storage Management")
    
    try:
        storage_stats = st.session_state.history_storage.get_storage_stats()
        
        if storage_stats:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Storage Statistics**")
                st.write(f"📁 Converted files: {storage_stats.get('total_converted_files', 0)}")
                st.write(f"🗃️ Temporary files: {storage_stats.get('total_temp_files', 0)}")
                st.write(f"📝 Log files: {storage_stats.get('total_log_files', 0)}")
            
            with col2:
                st.markdown("**Storage Usage**")
                converted_size = storage_stats.get('converted_dir_size', 0) / (1024 * 1024)
                temp_size = storage_stats.get('temp_dir_size', 0) / (1024 * 1024)
                logs_size = storage_stats.get('logs_dir_size', 0) / (1024 * 1024)
                
                st.write(f"📁 Converted: {converted_size:.2f} MB")
                st.write(f"🗃️ Temporary: {temp_size:.2f} MB")
                st.write(f"📝 Logs: {logs_size:.2f} MB")
        
        # Cleanup actions
        st.markdown("**Cleanup Actions**")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧹 Clean Temporary Files", help="Remove all temporary files"):
                if st.session_state.history_storage.cleanup_temp_files():
                    st.success("✅ Temporary files cleaned successfully")
                else:
                    st.error("❌ Failed to clean temporary files")
        
        with col2:
            if st.button("📥 Export History", help="Download conversion history as JSON"):
                history_data = [
                    {
                        "timestamp": h.timestamp.isoformat(),
                        "output_path": str(h.output_path),
                        "status": h.status.value,
                        "execution_time": h.execution_time,
                        "original_size": h.original_size,
                        "converted_size": h.converted_size,
                        "error_message": h.error_message
                    }
                    for h in history
                ]
                
                import json
                json_str = json.dumps(history_data, indent=2)
                st.download_button(
                    "Download JSON",
                    json_str,
                    file_name=f"conversion_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    except Exception as e:
        st.warning(f"⚠️ Could not load storage statistics: {str(e)}")