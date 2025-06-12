import streamlit as st
import qrcode
from io import BytesIO
from PIL import Image
import base64

def generate_qr_code(address, crypto_name):
    """Generate QR code for crypto address."""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(address)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Convert to bytes for Streamlit
    buf = BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf

def render_donations():
    """Render the donations page."""
    st.title("💰 Support ML Model Converter")
    
    # Introduction
    st.markdown("""
    Thank you for considering supporting this open-source project! Your donations help maintain 
    and improve the ML Model Converter for the entire community.
    """)
    
    # Why support section
    with st.expander("🚀 Why Support This Project?", expanded=False):
        st.markdown("""
        - **🆓 Always Free**: Keep the tool completely free for everyone
        - **🔧 Continuous Development**: Add new model formats and features  
        - **🐛 Bug Fixes**: Maintain compatibility with latest frameworks
        - **📚 Documentation**: Improve guides and tutorials
        - **🌍 Community**: Support a growing community of ML developers
        - **⚡ Performance**: Optimize conversion speed and accuracy
        - **🛠️ New Features**: Batch processing, cloud storage, advanced optimizations
        """)
    
    st.markdown("---")
    st.markdown("## 💎 Crypto Donations")
    
    # Donation addresses
    donations = {
        "Bitcoin (BTC)": {
            "address": "bc1qxcgk9xt5jzwj24r9nm7cmaju5we4tdh8d78sn2",
            "emoji": "₿",
            "color": "#f7931a"
        },
        "Solana (SOL)": {
            "address": "Fb8E2B8gcmyJucQNMhwrgB1jau2FRoKdpJ3YJZu14gTV",
            "emoji": "☀️",
            "color": "#9945ff"
        },
        "Binance Coin (BNB)": {
            "address": "0x6f0c51E1322a89a7a492aDCd0D9f472eC3641F0D",
            "emoji": "🔶",
            "color": "#f0b90b"
        },
        "Ripple (XRP)": {
            "address": "rHaxd6EhGbTz8ckSJ2QoGP6icEvRpbDFiK",
            "emoji": "💧",
            "color": "#23292f"
        }
    }
    
    # Stablecoin addresses
    stablecoins = {
        "USDT Polygon": {
            "address": "TYMYc2YV8QZLYjM4MYc7SsWsQMFLkegNsW",
            "emoji": "💵",
            "network": "Polygon"
        },
        "USDT Tron": {
            "address": "TYMYc2YV8QZLYjM4MYc7SsWsQMFLkegNsW", 
            "emoji": "💵",
            "network": "Tron"
        },
        "USDC Solana": {
            "address": "Fb8E2B8gcmyJucQNMhwrgB1jau2FRoKdpJ3YJZu14gTV",
            "emoji": "🏦",
            "network": "Solana"
        },
        "USDC Tron": {
            "address": "TYMYc2YV8QZLYjM4MYc7SsWsQMFLkegNsW",
            "emoji": "🏦", 
            "network": "Tron"
        },
        "USDC Polygon": {
            "address": "0x6f0c51E1322a89a7a492aDCd0D9f472eC3641F0D",
            "emoji": "🏦",
            "network": "Polygon"
        },
        "USDC Base": {
            "address": "0x6f0c51E1322a89a7a492aDCd0D9f472eC3641F0D",
            "emoji": "🏦",
            "network": "Base"
        }
    }
    
    # Display main cryptocurrencies
    st.markdown("### 🪙 Major Cryptocurrencies")
    
    for crypto_name, info in donations.items():
        with st.container():
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                st.markdown(f"### {info['emoji']} {crypto_name}")
            
            with col2:
                # Address display with copy button
                st.code(info['address'], language=None)
                if st.button(f"📋 Copy {crypto_name} Address", key=f"copy_{crypto_name}"):
                    st.success(f"✅ {crypto_name} address copied to clipboard!")
                    # Note: Actual clipboard copying requires JavaScript, this is just UI feedback
            
            with col3:
                # Generate and display QR code
                try:
                    qr_buffer = generate_qr_code(info['address'], crypto_name)
                    st.image(qr_buffer, width=150, caption=f"{crypto_name} QR Code")
                except Exception as e:
                    st.error(f"Could not generate QR code: {str(e)}")
            
            st.markdown("---")
    
    # Display stablecoins
    st.markdown("### 🏛️ Stablecoins (USDT & USDC)")
    
    # Group by coin type
    usdt_coins = {k: v for k, v in stablecoins.items() if 'USDT' in k}
    usdc_coins = {k: v for k, v in stablecoins.items() if 'USDC' in k}
    
    for coin_group, title in [(usdt_coins, "💵 USDT (Tether)"), (usdc_coins, "🏦 USDC (USD Coin)")]:
        st.markdown(f"#### {title}")
        
        for crypto_name, info in coin_group.items():
            with st.container():
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col1:
                    st.markdown(f"**{info['emoji']} {info['network']} Network**")
                
                with col2:
                    st.code(info['address'], language=None)
                    if st.button(f"📋 Copy {crypto_name}", key=f"copy_{crypto_name}"):
                        st.success(f"✅ {crypto_name} address copied!")
                
                with col3:
                    try:
                        qr_buffer = generate_qr_code(info['address'], crypto_name)
                        st.image(qr_buffer, width=120, caption=f"{crypto_name}")
                    except Exception as e:
                        st.error(f"QR code error: {str(e)}")
                
                st.markdown("")
    
    st.markdown("---")
    
    # Thank you section
    st.markdown("## 🙏 Thank You!")
    
    st.markdown("""
    Every donation, no matter the size, helps keep this project alive and growing. Your support enables us to:
    
    - ✅ Add support for new ML frameworks (TensorFlow, Keras, more ONNX features)
    - ✅ Improve conversion speed and accuracy
    - ✅ Create better documentation and tutorials  
    - ✅ Provide community support
    - ✅ Keep the project completely free
    - ✅ Implement advanced features like batch processing and cloud storage
    """)
    
    # Other ways to support
    with st.expander("🤝 Other Ways to Support", expanded=False):
        st.markdown("""
        Don't have crypto? No problem! Here are other ways to support:
        
        - ⭐ **Star the repository** on GitHub
        - 🐛 **Report bugs** and suggest improvements  
        - 📝 **Contribute code** or documentation
        - 💬 **Share the project** with other developers
        - 🎓 **Write tutorials** or create examples
        - 📢 **Spread the word** on social media
        - 🎯 **Use the tool** and provide feedback
        """)
    
    # Statistics (placeholder for future implementation)
    with st.expander("📊 Project Impact", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Models Converted", "🔄", help="Community conversions")
        with col2:
            st.metric("Frameworks Supported", "4+", help="PyTorch, TensorFlow, Keras, ONNX")
        with col3:
            st.metric("Community Members", "🌍", help="Growing daily")
        with col4:
            st.metric("GitHub Stars", "⭐", help="Star us on GitHub!")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h3>Made with ❤️ for the ML community</h3>
        <p><em>Building better tools together</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # JavaScript for clipboard functionality (optional enhancement)
    st.markdown("""
    <script>
    function copyToClipboard(text) {
        navigator.clipboard.writeText(text).then(function() {
            // Success callback
        });
    }
    </script>
    """, unsafe_allow_html=True)