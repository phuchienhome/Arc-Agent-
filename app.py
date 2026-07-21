import streamlit as st
from web3 import Web3
import plotly.express as px
import pandas as pd
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Arc Testnet AI Market Agent", page_icon="📈", layout="wide")

# Cấu hình
RPC_URL = os.getenv("RPC_URL", "https://rpc.testnet.arc.network")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

w3 = Web3(Web3.HTTPProvider(RPC_URL))

st.title("🤖 Arc Testnet AI Market Analysis Agent")
st.markdown("**Phân tích hoạt động kinh tế on-chain trên Arc Testnet (Circle)**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Cấu hình")
    api_key = st.text_input("Groq API Key (miễn phí)", value=GROQ_API_KEY, type="password")
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
    
    num_blocks = st.slider("Số block gần nhất để phân tích", 10, 100, 30)
    st.markdown("---")
    st.info("USDC là native gas token trên Arc. Faucet: [faucet.circle.com](https://faucet.circle.com/)")

# Hàm lấy dữ liệu on-chain
def get_arc_stats(num_blocks=30):
    if not w3.is_connected():
        return None, "Không kết nối được RPC Arc Testnet"
    
    latest_block = w3.eth.block_number
    tx_counts = []
    gas_prices = []
    block_times = []
    
    for i in range(num_blocks):
        try:
            block = w3.eth.get_block(latest_block - i)
            tx_counts.append(len(block.transactions))
            if block.baseFeePerGas:
                gas_prices.append(block.baseFeePerGas / 1e9)  # Gwei
            block_times.append(block.timestamp)
        except:
            continue
    
    avg_tx = sum(tx_counts) / len(tx_counts) if tx_counts else 0
    total_tx = sum(tx_counts)
    
    return {
        "latest_block": latest_block,
        "avg_tx_per_block": round(avg_tx, 2),
        "total_tx_analyzed": total_tx,
        "tx_counts": tx_counts,
        "avg_gas_gwei": round(sum(gas_prices)/len(gas_prices), 2) if gas_prices else 0
    }, None

# Giao diện chính
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Dữ liệu On-chain Real-time")
    if st.button("🔄 Cập nhật dữ liệu", type="primary"):
        with st.spinner("Đang lấy dữ liệu từ Arc Testnet..."):
            stats, error = get_arc_stats(num_blocks)
            if error:
                st.error(error)
            else:
                st.session_state["stats"] = stats
                st.success("Đã cập nhật dữ liệu!")

    if "stats" in st.session_state:
        stats = st.session_state["stats"]
        st.metric("Block mới nhất", stats["latest_block"])
        st.metric("TB giao dịch/block", stats["avg_tx_per_block"])
        st.metric("Tổng giao dịch phân tích", stats["total_tx_analyzed"])
        st.metric("Gas trung bình (Gwei)", stats["avg_gas_gwei"])

with col2:
    if "stats" in st.session_state:
        stats = st.session_state["stats"]
        df = pd.DataFrame({
            "Block gần nhất": list(range(len(stats["tx_counts"]))),
            "Số giao dịch": stats["tx_counts"][::-1]
        })
        fig = px.line(df, x="Block gần nhất", y="Số giao dịch", title="Xu hướng giao dịch gần đây")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Phần AI Agent
st.subheader("🧠 Phân tích Thị trường bằng AI Agent")

if st.button("🚀 Chạy Phân tích AI", type="primary", disabled=not api_key):
    if "stats" not in st.session_state:
        st.warning("Vui lòng nhấn 'Cập nhật dữ liệu' trước!")
    else:
        with st.spinner("AI Agent đang phân tích..."):
            stats = st.session_state["stats"]
            
            prompt = f"""
Bạn là chuyên gia phân tích thị trường blockchain và on-chain economy.

Dữ liệu Arc Testnet hiện tại:
- Block mới nhất: {stats['latest_block']}
- Trung bình giao dịch mỗi block: {stats['avg_tx_per_block']}
- Tổng giao dịch trong {num_blocks} block gần nhất: {stats['total_tx_analyzed']}
- Gas trung bình: {stats['avg_gas_gwei']} Gwei

Hãy phân tích:
1. Tình hình hoạt động mạng (có sôi động không?)
2. Xu hướng (tăng/giảm so với bình thường)
3. Ý nghĩa đối với "thị trường" on-chain của Arc (stablecoin finance, tokenized assets)
4. Đánh giá tổng thể (Bullish / Neutral / Bearish) + lý do
5. Gợi ý hành động cho developer hoặc trader

Trả lời ngắn gọn, chuyên nghiệp, bằng tiếng Việt.
"""

            try:
                client = Groq(api_key=api_key)
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",  # hoặc mixtral-8x7b-32768
                    messages=[
                        {"role": "system", "content": "Bạn là AI Agent phân tích thị trường blockchain chuyên nghiệp."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1500
                )
                analysis = response.choices[0].message.content
                st.session_state["analysis"] = analysis
            except Exception as e:
                st.error(f"Lỗi gọi AI: {e}")

if "analysis" in st.session_state:
    st.markdown("### 📝 Báo cáo Phân tích Thị trường từ AI Agent")
    st.markdown(st.session_state["analysis"])

st.markdown("---")
st.caption("Arc Testnet AI Market Agent • Powered by Groq + Web3.py + Streamlit")

# Chạy app: streamlit run app.py
