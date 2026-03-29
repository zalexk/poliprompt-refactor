import streamlit as st
import os
import yaml
import json
from pathlib import Path
from poliprompt import TextClassifier, MultiModalClassifier
import pandas as pd

st.set_page_config(page_title="PoliPrompt v3.2 Expert Station", layout="wide")

# --- 1. Session State 初始化 ---
if 'init_done' not in st.session_state:
    st.session_state.init_done = False
if 'classifier' not in st.session_state:
    st.session_state.classifier = None

# --- 2. 侧边栏：API 密钥控制 ---
with st.sidebar:
    st.title("🔑 API Credentials")
    openai_key = st.text_input("OpenAI Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    openai_url = st.text_input("Proxy URL", value=os.getenv("OPENAI_BASE_URL", "https://api2.aigcbest.top/v1"))
    dashscope_key = st.text_input("DashScope Key (Required for Multimodal)", type="password", value=os.getenv("DASHSCOPE_API_KEY", ""))
    
    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["OPENAI_BASE_URL"] = openai_url
    os.environ["DASHSCOPE_API_KEY"] = dashscope_key

st.title("🛡️ PoliPrompt v3.2 Expert Workstation")

# --- 3. 动态表单配置 ---
st.subheader("⚙️ System Configuration")

with st.form("main_config_form"):
    # 第一行：项目基础信息
    row1_col1, row1_col2, row1_col3 = st.columns([2, 1, 1])
    with row1_col1:
        ws_path = st.text_input("📁 Root Work Station Path", value="D:/PoliPrompt-main/examples/HatefulMemes")
    with row1_col2:
        proj_name = st.text_input("Project Name", value="PoliPrompt v3.2 Demo")
    with row1_col3:
        proj_ver = st.text_input("Version", value="v1")

    st.divider()

    # 第二行：模态选择与动态列映射
    row2_col1, row2_col2 = st.columns([1, 3])
    with row2_col1:
        modality = st.selectbox("🎯 Modality", ["text", "multimodal"], index=1)
        data_file = st.text_input("📄 Data Filename", value="train.jsonl" if modality=="multimodal" else "topic_data.csv")
    
    with row2_col2:
        st.markdown(f"##### 🗺️ Column Mapping ({modality})")
        map_c1, map_c2, map_c3, map_c4 = st.columns(4)
        with map_c1:
            text_col = st.text_input("Text Column", value="text")
        with map_c2:
            # 🚀 动态显示图片列
            if modality == "multimodal":
                image_col = st.text_input("Image Column", value="img")
            else:
                image_col = st.text_input("Image Column", value="None", disabled=True)
        with map_c3:
            # 🚀 动态显示图片目录
            if modality == "multimodal":
                image_dir = st.text_input("Image Dir", value=".")
            else:
                image_dir = st.text_input("Image Dir", value="None", disabled=True)
        with map_c4:
            ans_col = st.text_input("Answer Column", value="label")

    st.divider()

    # 第三行：算法与模型设置
    row3_c1, row3_c2, row3_c3 = st.columns(3)
    with row3_c1:
        st.markdown("##### 🧠 Algorithm")
        lambda_val = st.slider("Lambda (Diversity vs Similarity)", 0.0, 1.0, 0.5)
        k_shots = st.number_input("K-Shots", 1, 10, 3)
        options_raw = st.text_input("Label Options (Comma separated)", value="0, 1" if modality=="multimodal" else "politics, business, sport")
        testing_mode = st.checkbox("Testing Mode", value=False)
        testing_size = st.number_input("Testing Size", 1, 5000, 128)
        
    with row3_c2:
        st.markdown("##### 🤖 Models")
        # 根据模态自动推荐 Embedding 模型
        emb_llm = st.selectbox("Embedding LLM", ["qwen", "openai"], index=0 if modality=="multimodal" else 1)
        primary_m = st.selectbox("Primary LLM (L1)", ["gpt-4o-mini", "gpt-4-turbo"], index=0)
        secondary_m = st.selectbox("Secondary LLM (L2)", ["qwen-vl-max", "gpt-4o-mini"], index=0)
        expert_m = st.selectbox("Expert LLM (L3)", ["gpt-4o", "claude-3-5-sonnet-20240620"], index=0)

    with row3_c3:
        st.markdown("##### ⚙️ Performance & Obs")
        num_workers = st.select_slider("Parallel Workers", options=[1, 2, 4, 8, 12, 16], value=4)
        obs_enabled = st.toggle("Enable Observability (Langfuse)", value=True)
        n_exemplars = st.number_input("Exemplar Pool Size", 10, 1000, 256)

    # 第四行：Prompt 编辑
    st.markdown("##### 📝 Seed Prompt Definition")
    prompt_path = Path(ws_path) / "infiles/prompts/text_prompt.txt"
    current_prompt = ""
    if prompt_path.exists():
        current_prompt = prompt_path.read_text(encoding='utf-8')
    edited_prompt = st.text_area("Seed Prompt Content", value=current_prompt, height=200)

    # 提交按钮
    submitted = st.form_submit_button("🚀 Apply & Initialize", type="primary", use_container_width=True)

# --- 4. 逻辑处理：保存全量 YAML ---
if submitted:
    try:
        # 严格按照你要求的 YAML 结构构造字典
        options_list = [opt.strip() for opt in options_raw.split(",")]
        
        full_config = {
            "project": {
                "name": proj_name,
                "version": proj_ver,
                "modality": modality,
                "work_station": ws_path,
                "data_path": data_file,
                "image_dir": image_dir if modality == "multimodal" else ".",
                "outfiles_dir": "outfiles"
            },
            "column_mapping": {
                "text_col": text_col,
                "image_col": image_col if modality == "multimodal" else "None",
                "answer_col": ans_col
            },
            "user_settings": {
                "lambda_param": lambda_val,
                "k_shots": k_shots,
                "options": options_list,
                "testing": testing_mode,
                "testing_size": testing_size
            },
            "models": {
                "embedding_llm": emb_llm,
                "primary_llm": primary_m,
                "secondary_llm": secondary_m,
                "expert_llm": expert_m
            },
            "retrieval": {
                "n_exemplars_pool": n_exemplars
            },
            "parallel": {
                "num_workers": num_workers
            },
            "observability": {
                "enabled": obs_enabled,
                "provider": "langfuse"
            }
        }

        # 保存物理文件
        yaml_save_path = Path(ws_path).parent.parent / "config.yaml"
        with open(yaml_save_path, 'w', encoding='utf-8') as f:
            yaml.dump(full_config, f, default_flow_style=False, sort_keys=False)
        
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(edited_prompt, encoding='utf-8')

        # 初始化后台
        ClassifierClass = MultiModalClassifier if modality == "multimodal" else TextClassifier
        st.session_state.classifier = ClassifierClass(
            env_path=yaml_save_path, # 临时复用
            config_path=yaml_save_path,
            prompt_path=prompt_path
        )
        st.session_state.init_done = True
        st.success(f"✅ Config saved to {yaml_save_path} and System Initialized!")
        st.balloons()
    except Exception as e:
        st.error(f"Initialization Error: {e}")

# --- 5. 运行控制 (逻辑保持不变) ---
if st.session_state.init_done:
    st.divider()
    t1, t2, t3 = st.tabs(["🚀 Run Tasks", "📂 Logic (RAFS)", "📊 Live Results"])
    tab1, tab2, tab3 = st.tabs(["🎮 Run Control", "🧠 RAFS Logic", "📊 Analytics"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🔨 Step 1: Build Pool", use_container_width=True):
                st.session_state.classifier.create_few_shot_pool()
                st.toast("Pool Created!", icon="✅")
        with c2:
            if st.button("🧠 Step 2: Optimize", use_container_width=True):
                rules = st.session_state.classifier.optimize_task_description()
                st.session_state.enhanced_rules = rules
                st.toast("Logic Synthesized!", icon="🧠")
        with c3:
            if st.button("🚀 Step 3: Run Inference", use_container_width=True):
                st.warning("Inference Active. Check VS Code Terminal.")
                st.session_state.classifier.annotate()

    with tab2:
        if hasattr(st.session_state, 'enhanced_rules'):
            st.info(st.session_state.enhanced_rules)
        else:
            st.info("Run Phase 2 to generate logic.")

    with tab3:
        log_file = Path(ws_path) / "outfiles/observability_logs.jsonl"
        if log_file.exists():
            if st.button("🔄 Refresh Data"):
                df = pd.read_json(log_file, lines=True)
                st.metric("Total Processed", len(df))
                st.dataframe(df.tail(20), use_container_width=True)