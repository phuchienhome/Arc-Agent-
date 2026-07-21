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

st.set_page_config(page_title="Arc Testnet CrewAI Agent", page_icon="🤖", layout="wide")

# ==================== CẤU HÌNH ====================
RPC_URL = os.getenv("RPC_URL", "https://rpc.testnet.arc.network")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

w3 = Web3(Web3.HTTPProvider(RPC_URL))

llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY if GROQ_API_KEY else None
)

st.title("🤖 Arc Testnet CrewAI Multi-Agent")
st.markdown("**Phân tích thị trường + Câu hỏi tự do + Cảnh báo tự động**")

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

# ==================== CREWAI TOOL ====================
@tool("Get Arc Testnet On-Chain Data")
def get_arc_testnet_data(num_blocks: int) -> str:
    """Lấy dữ liệu on-chain thực tế từ Arc Testnet."""
    stats = get_arc_stats(num_blocks)
    if stats is None:
        return "Không kết nối được RPC."
    return json.dumps(stats, ensure_ascii=False)

# ==================== AGENTS ====================
data_collector = Agent(
    role="Blockchain Data Collector",
    goal="Thu thập dữ liệu on-chain chính xác từ Arc Testnet",
    backstory="Chuyên gia thu thập dữ liệu blockchain.",
    tools=[get_arc_testnet_data],
    llm=llm,
    verbose=True
)

trend_analyst = Agent(
    role="Market Trend Analyst",
    goal="Phân tích xu hướng và biến động hoạt động mạng",
    backstory="Nhà phân tích on-chain chuyên sâu.",
    llm=llm,
    verbose=True
)

economic_insights = Agent(
    role="On-Chain Economy Expert",
    goal="Đánh giá tác động kinh tế đến stablecoin và tokenized assets trên Arc",
    backstory="Chuyên gia về hệ sinh thái Arc của Circle.",
    llm=llm,
    verbose=True
)

report_writer = Agent(
    role="Professional Report & Answer Synthesizer",
    goal="Trả lời câu hỏi của user và tổng hợp báo cáo thị trường chất lượng cao bằng tiếng Việt",
    backstory="Chuyên gia viết báo cáo và trả lời câu hỏi về blockchain.",
    llm=llm,
    verbose=True
)

# ==================== TASKS (có hỗ trợ câu hỏi tự do) ====================
collect_data_task = Task(
    description="Lấy dữ liệu on-chain Arc Testnet với số block được chỉ định.",
    expected_output="Dữ liệu JSON.",
    agent=data_collector
)

analyze_trend_task = Task(
    description="Phân tích xu hướng từ dữ liệu thu thập được.",
    expected_output="Phân tích xu hướng bằng tiếng Việt.",
    agent=trend_analyst,
    context=[collect_data_task]
)

economic_analysis_task = Task(
    description="Đánh giá ý nghĩa kinh tế của dữ liệu.",
    expected_output="Đánh giá kinh tế bằng tiếng Việt.",
    agent=economic_insights,
    context=[collect_data_task, analyze_trend_task]
)

final_report_task = Task(
    description="""Dựa trên toàn bộ dữ liệu và phân tích, hãy:
1. Trả lời câu hỏi sau của người dùng (nếu có): {user_question}
2. Nếu không có câu hỏi cụ thể thì đưa ra báo cáo thị trường tổng quát Arc Testnet.
3. Kết hợp cảnh báo nếu phát hiện hoạt động bất thường.
Trả lời rõ ràng, có cấu trúc bằng tiếng Việt.""",
    expected_output="Báo cáo + câu trả lời chi tiết bằng tiếng Việt.",
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

# ==================== GIAO DIỆN ====================
with st.sidebar:
    st.header("⚙️ Cấu hình")
    api_key = st.text_input("Groq API Key", value=GROQ_API_KEY, type="password")
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
    
    num_blocks = st.slider("Số block gần nhất", 10, 80, 30)
    st.markdown("---")
    st.info("CrewAI sẽ chạy 4 agent chuyên biệt")

# === NHẬP CÂU HỎI TỰ DO ===
st.subheader("💬 Câu hỏi tự do (tùy chọn)")
user_question = st.text_area(
    "Bạn muốn hỏi gì về thị trường Arc Testnet?",
    placeholder="Ví dụ: Phân tích tác động khi số giao dịch tăng mạnh? Hoặc: Cơ hội nào cho stablecoin trên Arc hiện nay?",
    height=100
)

# === NÚT CHẠY ===
if st.button("🚀 Chạy CrewAI + Phân tích + Cảnh báo", type="primary"):
    if not api_key:
        st.error("Vui lòng nhập Groq API Key!")
    else:
        # 1. Lấy dữ liệu và cảnh báo trước (nhanh)
        stats = get_arc_stats(num_blocks)
        
        if stats:
            # === CẢNH BÁO TỰ ĐỘNG ===
            st.subheader("🔔 Cảnh báo tự động")
            avg_tx = stats["avg_tx_per_block"]
            
            if avg_tx > 25:
                st.error(f"⚠️ **HOẠT ĐỘNG RẤT CAO** — Trung bình {avg_tx} giao dịch/block. Có thể đang có sự kiện lớn hoặc bot activity mạnh.")
            elif avg_tx > 15:
                st.warning(f"📈 **HOẠT ĐỘNG CAO** — Trung bình {avg_tx} giao dịch/block. Thị trường đang sôi động.")
            elif avg_tx < 5:
                st.info(f"📉 **HOẠT ĐỘNG THẤP** — Trung bình {avg_tx} giao dịch/block. Thị trường đang yên ắng.")
            else:
                st.success(f"✅ **HOẠT ĐỘNG BÌNH THƯỜNG** — Trung bình {avg_tx} giao dịch/block.")
            
            # Hiển thị metrics nhanh
            col1, col2, col3 = st.columns(3)
            col1.metric("Block mới nhất", stats["latest_block"])
            col2.metric("TB TX / Block", stats["avg_tx_per_block"])
            col3.metric("Tổng TX phân tích", stats["total_tx_analyzed"])

            # Biểu đồ
            df = pd.DataFrame({
                "Block gần nhất": list(range(len(stats["tx_counts"]))),
                "Số giao dịch": stats["tx_counts"]
            })
            fig = px.line(df, x="Block gần nhất", y="Số giao dịch", title="Xu hướng giao dịch gần đây")
            st.plotly_chart(fig, use_container_width=True)

        # 2. Chạy CrewAI
        with st.spinner("🤖 CrewAI đang phân tích sâu (có thể mất 20-60 giây)..."):
            try:
                question_text = user_question if user_question.strip() else "Không có câu hỏi cụ thể. Hãy đưa ra báo cáo thị trường tổng quát Arc Testnet."
                
                result = crew.kickoff(inputs={
                    "num_blocks": num_blocks,
                    "user_question": question_text
                })
                
                st.success("✅ Phân tích CrewAI hoàn tất!")
                st.markdown("### 📝 Kết quả từ Multi-Agent Crew")
                st.markdown(result)
                
            except Exception as e:
                st.error(f"Lỗi khi chạy CrewAI: {str(e)}")

st.caption("Arc Testnet CrewAI Agent • Multi-Agent + Custom Question + Auto Alert • Powered by CrewAI + Groq")
