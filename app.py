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
import streamlit as st
from web3 import Web3
import plotly.express as px
import pandas as pd
import json
from dotenv import load_dotenv
import os

from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import tool

load_dotenv()

st.set_page_config(page_title="Arc Testnet CrewAI Market Agent", page_icon="🤖", layout="wide")

# ==================== CẤU HÌNH ====================
RPC_URL = os.getenv("RPC_URL", "https://rpc.testnet.arc.network")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

w3 = Web3(Web3.HTTPProvider(RPC_URL))

# LLM dùng Groq (miễn phí & nhanh)
llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY if GROQ_API_KEY else None
)

st.title("🤖 Arc Testnet CrewAI Multi-Agent Market Analysis")
st.markdown("**Hệ thống Multi-Agent thông minh phân tích thị trường on-chain Arc Testnet (Circle)**")

# ==================== HÀM LẤY DỮ LIỆU ====================
def get_arc_stats(num_blocks: int = 30):
    if not w3.is_connected():
        return None
    latest_block = w3.eth.block_number
    tx_counts = []
    for i in range(min(num_blocks, 100)):
        try:
            block = w3.eth.get_block(latest_block - i)
            tx_counts.append(len(block.transactions))
        except:
            continue
    avg_tx = sum(tx_counts) / len(tx_counts) if tx_counts else 0
    return {
        "latest_block": latest_block,
        "avg_tx_per_block": round(avg_tx, 2),
        "total_tx_analyzed": sum(tx_counts),
        "tx_counts": tx_counts[::-1],
        "num_blocks_analyzed": len(tx_counts)
    }

# ==================== CREWAI CUSTOM TOOL ====================
@tool("Get Arc Testnet On-Chain Data")
def get_arc_testnet_data(num_blocks: int) -> str:
    """Lấy dữ liệu on-chain thực tế từ Arc Testnet (số giao dịch, block mới nhất...)."""
    stats = get_arc_stats(num_blocks)
    if stats is None:
        return "Không kết nối được với Arc Testnet RPC."
    return json.dumps(stats, ensure_ascii=False)

# ==================== ĐỊNH NGHĨA AGENTS ====================
data_collector = Agent(
    role="Blockchain Data Collector",
    goal="Thu thập dữ liệu on-chain chính xác và đầy đủ từ Arc Testnet",
    backstory="Bạn là chuyên gia thu thập dữ liệu blockchain, luôn lấy thông tin mới nhất và đáng tin cậy.",
    tools=[get_arc_testnet_data],
    llm=llm,
    verbose=True
)

trend_analyst = Agent(
    role="Market Trend Analyst",
    goal="Phân tích xu hướng giao dịch, biến động hoạt động mạng và phát hiện pattern",
    backstory="Bạn là nhà phân tích thị trường crypto chuyên sâu, giỏi đọc dữ liệu on-chain để tìm xu hướng.",
    llm=llm,
    verbose=True
)

economic_insights = Agent(
    role="On-Chain Economy Expert",
    goal="Đánh giá ý nghĩa kinh tế của dữ liệu đối với hệ sinh thái stablecoin và tokenized assets trên Arc",
    backstory="Bạn hiểu rõ Arc là Economic OS của Circle, chuyên phân tích tác động đến stablecoin finance và on-chain markets.",
    llm=llm,
    verbose=True
)

report_writer = Agent(
    role="Professional Report Synthesizer",
    goal="Tổng hợp tất cả phân tích thành báo cáo thị trường rõ ràng, chuyên nghiệp bằng tiếng Việt",
    backstory="Bạn là chuyên gia viết báo cáo tài chính và blockchain, luôn trình bày logic, khách quan và dễ hiểu.",
    llm=llm,
    verbose=True
)

# ==================== ĐỊNH NGHĨA TASKS ====================
collect_data_task = Task(
    description="Sử dụng tool để lấy dữ liệu on-chain Arc Testnet với số block do người dùng chỉ định. Trả về dữ liệu dạng JSON rõ ràng.",
    expected_output="Dữ liệu JSON chứa latest_block, avg_tx_per_block, total_tx_analyzed và danh sách tx_counts.",
    agent=data_collector
)

analyze_trend_task = Task(
    description="Phân tích dữ liệu vừa thu thập: xu hướng giao dịch tăng/giảm, mức độ hoạt động so với bình thường, biến động gas.",
    expected_output="Phân tích xu hướng chi tiết bằng tiếng Việt.",
    agent=trend_analyst,
    context=[collect_data_task]
)

economic_analysis_task = Task(
    description="Dựa trên dữ liệu và phân tích xu hướng, đánh giá tác động đến thị trường stablecoin, tokenized assets và hoạt động kinh tế trên Arc.",
    expected_output="Đánh giá kinh tế sâu sắc bằng tiếng Việt.",
    agent=economic_insights,
    context=[collect_data_task, analyze_trend_task]
)

final_report_task = Task(
    description="Tổng hợp toàn bộ kết quả thành báo cáo thị trường Arc Testnet hoàn chỉnh bằng tiếng Việt. Bao gồm: Tóm tắt, Phân tích xu hướng, Đánh giá kinh tế, Kết luận và Gợi ý.",
    expected_output="Báo cáo thị trường chuyên nghiệp, rõ ràng, có cấu trúc bằng tiếng Việt.",
    agent=report_writer,
    context=[collect_data_task, analyze_trend_task, economic_analysis_task]
)

# ==================== CREW ====================
crew = Crew(
    agents=[data_collector, trend_analyst, economic_insights, report_writer],
    tasks=[collect_data_task, analyze_trend_task, economic_analysis_task, final_report_task],
    process=Process.sequential,
    verbose=True
)

# ==================== GIAO DIỆN STREAMLIT ====================
with st.sidebar:
    st.header("⚙️ Cấu hình CrewAI")
    api_key_input = st.text_input("Groq API Key", value=GROQ_API_KEY, type="password")
    if api_key_input:
        os.environ["GROQ_API_KEY"] = api_key_input
        llm.api_key = api_key_input  # cập nhật lại
    
    num_blocks = st.slider("Số block gần nhất phân tích", 10, 80, 30)
    st.markdown("---")
    st.info("CrewAI sẽ chạy 4 agent chuyên biệt. Thời gian xử lý khoảng 20-60 giây.")

# Nút chạy Crew
if st.button("🚀 Chạy CrewAI Multi-Agent Analysis", type="primary"):
    if not api_key_input:
        st.error("Vui lòng nhập Groq API Key!")
    else:
        with st.spinner("🤖 CrewAI đang làm việc... (Data Collector → Analyst → Insights → Report)"):
            try:
                # Chạy Crew
                result = crew.kickoff(inputs={"num_blocks": num_blocks})
                
                st.success("✅ Phân tích hoàn tất!")
                
                # Hiển thị kết quả
                st.markdown("### 📊 Kết quả từ CrewAI Multi-Agent")
                st.markdown(result)
                
            except Exception as e:
                st.error(f"Lỗi khi chạy CrewAI: {str(e)}")
                st.info("Hãy kiểm tra lại Groq API Key và kết nối mạng.")

# Hiển thị biểu đồ nếu có dữ liệu (giữ nguyên từ phiên bản cũ)
st.markdown("---")
st.subheader("📈 Biểu đồ hoạt động gần đây (dữ liệu thực tế)")

if st.button("Cập nhật biểu đồ"):
    stats = get_arc_stats(num_blocks)
    if stats:
        df = pd.DataFrame({
            "Block index (gần nhất)": list(range(len(stats["tx_counts"]))),
            "Số giao dịch": stats["tx_counts"]
        })
        fig = px.line(df, x="Block index (gần nhất)", y="Số giao dịch", 
                      title=f"Xu hướng giao dịch {num_blocks} block gần nhất trên Arc Testnet")
        st.plotly_chart(fig, use_container_width=True)
        st.json(stats)
    else:
        st.error("Không lấy được dữ liệu từ RPC.")

st.caption("Arc Testnet CrewAI Market Agent • Multi-Agent System powered by CrewAI + Groq + Web3.py")
