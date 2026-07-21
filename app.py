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

st.set_page_config(page_title="Arc Testnet AI Agent", page_icon="📈", layout="wide")

# ==================== CẤU HÌNH ====================
RPC_URL = os.getenv("RPC_URL", "https://rpc.testnet.arc.network")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

w3 = Web3(Web3.HTTPProvider(RPC_URL))

llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY
)

st.title("🤖 Arc Testnet AI Market Agent")
st.markdown("**Multi-Agent System • Phân tích thị trường + Câu hỏi tự do + Cảnh báo tự động**")

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
        "tx_counts": tx_counts[::-1]
    }

# ==================== CREWAI TOOL ====================
@tool("Get Arc Testnet Data")
def get_arc_testnet_data(num_blocks: int) -> str:
    stats = get_arc_stats(num_blocks)
    return json.dumps(stats, ensure_ascii=False) if stats else "Không kết nối được RPC"

# ==================== AGENTS ====================
data_collector = Agent(
    role="Data Collector",
    goal="Thu thập dữ liệu on-chain chính xác từ Arc Testnet",
    backstory="Chuyên gia thu thập dữ liệu blockchain.",
    tools=[get_arc_testnet_data],
    llm=llm,
    verbose=True
)

trend_analyst = Agent(
    role="Trend Analyst",
    goal="Phân tích xu hướng hoạt động mạng",
    backstory="Nhà phân tích on-chain chuyên sâu.",
    llm=llm,
    verbose=True
)

economic_expert = Agent(
    role="Economic Expert",
    goal="Đánh giá tác động kinh tế đến stablecoin và tokenized assets",
    backstory="Chuyên gia về hệ sinh thái Arc của Circle.",
    llm=llm,
    verbose=True
)

report_agent = Agent(
    role="Report & Answer Agent",
    goal="Trả lời câu hỏi người dùng và đưa ra báo cáo thị trường chuyên nghiệp",
    backstory="Chuyên gia tổng hợp và trả lời câu hỏi về thị trường blockchain.",
    llm=llm,
    verbose=True
)

# ==================== TASKS ====================
collect_task = Task(
    description="Lấy dữ liệu on-chain với số block được chỉ định.",
    expected_output="Dữ liệu JSON.",
    agent=data_collector
)

trend_task = Task(
    description="Phân tích xu hướng từ dữ liệu.",
    expected_output="Phân tích xu hướng bằng tiếng Việt.",
    agent=trend_analyst,
    context=[collect_task]
)

economic_task = Task(
    description="Đánh giá ý nghĩa kinh tế.",
    expected_output="Đánh giá kinh tế bằng tiếng Việt.",
    agent=economic_expert,
    context=[collect_task, trend_task]
)

final_task = Task(
    description="""Dựa trên dữ liệu và phân tích:
- Trả lời câu hỏi của người dùng: {user_question}
- Nếu không có câu hỏi thì đưa báo cáo thị trường tổng quát.
- Kết hợp cảnh báo nếu có hoạt động bất thường.
Trả lời rõ ràng, có cấu trúc bằng tiếng Việt.""",
    expected_output="Báo cáo + câu trả lời chi tiết bằng tiếng Việt.",
    agent=report_agent,
    context=[collect_task, trend_task, economic_task]
)

crew = Crew(
    agents=[data_collector, trend_analyst, economic_expert, report_agent],
    tasks=[collect_task, trend_task, economic_task, final_task],
    process=Process.sequential,
    verbose=True
)

# ==================== GIAO DIỆN ====================
with st.sidebar:
    st.header("Cấu hình")
    api_key = st.text_input("Groq API Key (miễn phí)", value=GROQ_API_KEY, type="password")
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
    num_blocks = st.slider("Số block gần nhất", 10, 80, 30)

# Câu hỏi tự do
st.subheader("💬 Nhập câu hỏi tự do (tùy chọn)")
user_question = st.text_area(
    "Bạn muốn hỏi gì?",
    placeholder="Ví dụ: Phân tích khi số giao dịch tăng mạnh? Hoặc: Cơ hội cho developer trên Arc hiện nay?",
    height=80
)

# Nút chạy
if st.button("🚀 Chạy AI Agent (CrewAI)", type="primary"):
    if not api_key:
        st.error("Vui lòng nhập Groq API Key!")
    else:
        stats = get_arc_stats(num_blocks)

        # === CẢNH BÁO TỰ ĐỘNG ===
        if stats:
            st.subheader("🔔 Cảnh báo tự động")
            avg = stats["avg_tx_per_block"]
            if avg > 25:
                st.error(f"⚠️ HOẠT ĐỘNG RẤT CAO — {avg} TX/block")
            elif avg > 15:
                st.warning(f"📈 HOẠT ĐỘNG CAO — {avg} TX/block")
            elif avg < 5:
                st.info(f"📉 HOẠT ĐỘNG THẤP — {avg} TX/block")
            else:
                st.success(f"✅ HOẠT ĐỘNG BÌNH THƯỜNG — {avg} TX/block")

            # Metrics + Biểu đồ
            c1, c2, c3 = st.columns(3)
            c1.metric("Block mới nhất", stats["latest_block"])
            c2.metric("TB giao dịch/block", stats["avg_tx_per_block"])
            c3.metric("Tổng TX phân tích", stats["total_tx_analyzed"])

            df = pd.DataFrame({
                "Block gần nhất": list(range(len(stats["tx_counts"]))),
                "Số giao dịch": stats["tx_counts"]
            })
            fig = px.line(df, x="Block gần nhất", y="Số giao dịch")
            st.plotly_chart(fig, use_container_width=True)

        # === CHẠY CREWAI ===
        with st.spinner("🤖 CrewAI đang phân tích..."):
            try:
                question = user_question.strip() if user_question.strip() else "Không có câu hỏi cụ thể. Hãy đưa báo cáo thị trường tổng quát Arc Testnet."
                result = crew.kickoff(inputs={
                    "num_blocks": num_blocks,
                    "user_question": question
                })
                st.success("✅ Phân tích hoàn tất!")
                st.markdown("### 📝 Kết quả từ AI Agent")
                st.markdown(result)
            except Exception as e:
                st.error(f"Lỗi: {e}")

st.caption("Arc Testnet AI Agent • CrewAI Multi-Agent System")
