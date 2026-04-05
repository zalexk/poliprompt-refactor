"""
PoliPrompt v3.2 — Streamlit Control Center
支持：终端进度实时显示 + Streamlit HITL 界面（rerun 轮询模式）
"""
import os
import sys
import queue
import json
import yaml
import threading
import time
import html as _html
import streamlit as st
from pathlib import Path
from dotenv import set_key, dotenv_values

st.set_page_config(
    page_title="PoliPrompt v3.2",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# 工具：stdout + stderr 实时捕获
# ──────────────────────────────────────────────
class _QueueWriter:
    def __init__(self, q: queue.Queue, original):
        self._q    = q
        self._orig = original
    def write(self, text):
        self._orig.write(text)
        if text and text.strip():
            self._q.put(text)
    def flush(self):
        self._orig.flush()

def _render_log(placeholder, lines: list):
    escaped = _html.escape("".join(lines[-200:]))
    # replace \r and \n for HTML rendering
    escaped = escaped.replace("\r", "").replace("\n", "<br>")
    placeholder.markdown(
        f"""<div style="
                height:380px; overflow-y:auto;
                background:#0e1117; color:#e0e0e0;
                font-family:'Courier New',monospace; font-size:12px;
                padding:12px 16px; border-radius:6px;
                border:1px solid #333;
                white-space:pre-wrap; word-break:break-all;">
            {escaped}
            <div id='log-bottom'></div>
        </div>
        <script>
            const el = document.getElementById('log-bottom');
            if(el) el.scrollIntoView({{behavior:'smooth'}});
        </script>""",
        unsafe_allow_html=True,
    )

def run_with_live_log(func, log_placeholder, done_cb=None):
    log_q   = queue.Queue()
    err_box = [None]
    lines   = []

    orig_out, orig_err = sys.stdout, sys.stderr
    writer     = _QueueWriter(log_q, orig_out)
    sys.stdout = writer
    sys.stderr = _QueueWriter(log_q, orig_err)

    def _run():
        try:
            func()
        except Exception as e:
            err_box[0] = e
        finally:
            sys.stdout = orig_out
            sys.stderr = orig_err

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    while t.is_alive():
        try:
            while True:
                lines.append(log_q.get_nowait())
        except queue.Empty:
            pass
        _render_log(log_placeholder, lines)
        time.sleep(0.4)

    try:
        while True:
            lines.append(log_q.get_nowait())
    except queue.Empty:
        pass
    _render_log(log_placeholder, lines)

    if done_cb:
        done_cb()
    return err_box[0] is None, err_box[0]

# ──────────────────────────────────────────────
# 配置文件加载
# ──────────────────────────────────────────────
_THIS_DIR = Path(__file__).parent

def _find_config(filename: str) -> Path | None:
    for p in [
        _THIS_DIR / "src" / "poliprompt" / "configs" / filename,
        _THIS_DIR / "poliprompt" / "configs" / filename,
        _THIS_DIR / "configs" / filename,
    ]:
        if p.exists():
            return p
    return None

@st.cache_data
def _load_llm_configs() -> dict:
    p = _find_config("llm_configs.json")
    return json.loads(p.read_text(encoding="utf-8")) if p else {}

@st.cache_data
def _load_emb_configs() -> dict:
    p = _find_config("embedding_llm_configs.json")
    return json.loads(p.read_text(encoding="utf-8")) if p else {}

def get_models(level: str, modality: str) -> list:
    cfg = _load_llm_configs()
    out = [k for k, v in cfg.items()
           if v.get("level") == level
           and (modality == "text" or v.get("multimodal", False))]
    return out if out else [f"(no {level} models found)"]

def get_default(model_name: str, key: str, fallback):
    return _load_llm_configs().get(model_name, {}).get(key, fallback)

# ──────────────────────────────────────────────
# 侧边栏：API Credentials
# ──────────────────────────────────────────────
env_file     = _THIS_DIR / ".env"
existing_env = dotenv_values(env_file) if env_file.exists() else {}

def ev(key, default=""):
    return existing_env.get(key, default)

VENDORS = [
    dict(label="OpenAI",                key_env="OPENAI_API_KEY",    url_env="OPENAI_BASE_URL",    hint="https://api.openai.com/v1"),
    dict(label="Google",                key_env="GOOGLE_API_KEY",    url_env="GOOGLE_BASE_URL",    hint="https://generativelanguage.googleapis.com/v1beta/openai"),
    dict(label="Alibaba / Qwen",        key_env="DASHSCOPE_API_KEY", url_env="DASHSCOPE_BASE_URL", hint="https://dashscope.aliyuncs.com/compatible-mode/v1"),
    dict(label="Anthropic / Claude",    key_env="ANTHROPIC_API_KEY", url_env="ANTHROPIC_BASE_URL", hint="Use your relay URL"),
    dict(label="Mistral  · text only",  key_env="MISTRAL_API_KEY",   url_env="MISTRAL_BASE_URL",   hint="https://api.mistral.ai/v1"),
]
vendor_inputs = {}

with st.sidebar:
    st.title("🔑 API Credentials")
    st.caption("Fill in only the vendors you use. **Base URL** is optional.")

    for v in VENDORS:
        with st.expander(v["label"], expanded=False):
            vendor_inputs[v["key_env"]] = st.text_input("API Key",  value=ev(v["key_env"]), type="password", key=f"k_{v['key_env']}")
            vendor_inputs[v["url_env"]] = st.text_input("Base URL", value=ev(v["url_env"]), placeholder=v["hint"],    key=f"u_{v['url_env']}")

    st.divider()
    with st.expander("📡 Langfuse (Observability)", expanded=False):
        vendor_inputs["LANGFUSE_PUBLIC_KEY"] = st.text_input("Public Key", value=ev("LANGFUSE_PUBLIC_KEY"), type="password", key="k_LF_PK")
        vendor_inputs["LANGFUSE_SECRET_KEY"] = st.text_input("Secret Key", value=ev("LANGFUSE_SECRET_KEY"), type="password", key="k_LF_SK")
        vendor_inputs["LANGFUSE_HOST"]       = st.text_input("Host",       value=ev("LANGFUSE_HOST"),       placeholder="https://cloud.langfuse.com", key="k_LF_HOST")

    if st.button("💾 Save API Keys to .env", use_container_width=True):
        env_file.parent.mkdir(parents=True, exist_ok=True)
        for k, val in vendor_inputs.items():
            if val:
                set_key(str(env_file), k, val)
        st.success(f"Saved → `{env_file}`")

    st.divider()
    st.caption("**PoliPrompt v3.2**")

# ──────────────────────────────────────────────
# 主界面
# ──────────────────────────────────────────────
st.title("🛡️ PoliPrompt v3.2 Control Center")

def _load_existing_config(path: Path) -> dict:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {} if path.exists() else {}
    except Exception:
        return {}

# ══════════════════════════════════════════════
# SECTION 1 ▸ project
# ══════════════════════════════════════════════
with st.expander("📦 **project**", expanded=True):
    col_ws, col_name, col_ver, col_mod = st.columns([3, 2, 1, 1])
    ws_raw = col_ws.text_input("work_station  *(absolute path)*",
                                value=st.session_state.get("ws_path_str", "D:/PoliPrompt-main/examples/HatefulMemes"))
    ws_path = Path(ws_raw).expanduser().absolute()
    st.session_state["ws_path_str"] = str(ws_path)

    cfg_existing = _load_existing_config(ws_path / "config.yaml")
    proj = cfg_existing.get("project", {})

    proj_name    = col_name.text_input("name",    value=proj.get("name",    "MyProject"))
    proj_version = col_ver.text_input ("version", value=proj.get("version", "v1"))
    modality     = col_mod.selectbox("modality", ["text", "multimodal"],
                                     index=0 if proj.get("modality", "multimodal") == "text" else 1)

    col_data, col_img, col_out = st.columns(3)
    data_path    = col_data.text_input("data_path",    value=proj.get("data_path",    "train.jsonl"))
    image_dir    = col_img.text_input ("image_dir",    value=proj.get("image_dir",    "."), disabled=(modality == "text"))
    outfiles_dir = col_out.text_input ("outfiles_dir", value=proj.get("outfiles_dir", "outfiles"))

# ══════════════════════════════════════════════
# SECTION 2 ▸ column_mapping
# ══════════════════════════════════════════════
with st.expander("🗺️ **column_mapping**", expanded=True):
    cols_cfg = cfg_existing.get("column_mapping", {})
    cc1, cc2, cc3 = st.columns(3)
    text_col   = cc1.text_input("text_col",   value=cols_cfg.get("text_col",   "text"))
    image_col  = cc2.text_input("image_col",  value=cols_cfg.get("image_col",  "img")
                                if modality == "multimodal" else "None",
                                disabled=(modality == "text"))
    answer_col = cc3.text_input("answer_col", value=cols_cfg.get("answer_col", "label"))

# ══════════════════════════════════════════════
# SECTION 3 ▸ user_settings
# ══════════════════════════════════════════════
with st.expander("⚙️ **user_settings**", expanded=True):
    usr = cfg_existing.get("user_settings", {})
    raw_opts = st.text_input("options  *(comma-separated)*",
                              value=", ".join(str(o) for o in usr.get("options", ["0", "1"])))
    options_list = [i.strip() for i in raw_opts.split(",") if i.strip()]
    if options_list:
        st.markdown(f"**Recognised labels →** " + " ".join(f"`{o}`" for o in options_list))

    ua, ub, uc, ud = st.columns(4)
    lambda_param = ua.slider("lambda_param", 0.0, 1.0, float(usr.get("lambda_param", 0.5)), 0.05)
    k_shots      = ub.number_input("k_shots",      1,  20,  int(usr.get("k_shots",      3)))
    testing      = uc.toggle("testing",             value=bool(usr.get("testing",      False)))
    testing_size = ud.number_input("testing_size", 8, 2048, int(usr.get("testing_size", 128)), disabled=not testing)

# ══════════════════════════════════════════════
# SECTION 4 ▸ models + hyperparams
# ══════════════════════════════════════════════
with st.expander("🤖 **models**", expanded=True):
    mdls        = cfg_existing.get("models", {})
    emb_choices = list(_load_emb_configs().keys()) or ["openai", "qwen"]
    std_models  = get_models("standard", modality)
    exp_models  = get_models("expert",   modality)

    if modality == "multimodal":
        st.caption("ℹ️ Mistral models are hidden — they do not support images.")

    cfg_path_found = _find_config("llm_configs.json")
    st.caption(f"📂 Config: `{cfg_path_found}`" if cfg_path_found else "⚠️ llm_configs.json not found")

    m1, m2, m3, m4 = st.columns(4)
    def _pick(col, label, key, choices, saved):
        idx = choices.index(saved) if saved in choices else 0
        return col.selectbox(label, choices, index=idx, key=f"mdl_{key}")

    embedding_llm = _pick(m1, "🔍 embedding_llm", "emb", emb_choices, mdls.get("embedding_llm", emb_choices[0]))
    primary_llm   = _pick(m2, "🟢 primary_llm",   "l1",  std_models,  mdls.get("primary_llm",  std_models[0]))
    secondary_llm = _pick(m3, "🟡 secondary_llm", "l2",  std_models,  mdls.get("secondary_llm", std_models[0]))
    expert_llm    = _pick(m4, "🔴 expert_llm",    "l3",  exp_models,  mdls.get("expert_llm",   exp_models[0]))

    if modality == "multimodal" and embedding_llm == "openai":
        st.warning("⚠️ OpenAI embedding is text-only. Switch to **qwen** for multimodal tasks.")

    st.markdown("---")
    st.caption("Inference hyperparameters per agent:")
    card1, card2, card3 = st.columns(3)
    with card1:
        with st.container(border=True):
            st.markdown("🟢 **Primary**")
            l1_temp   = st.slider("Temperature", 0.0, 1.0, get_default(primary_llm,   "temperature", 0.0), 0.05, key="l1_temp")
            l1_tokens = st.number_input("Max Tokens", 100, 4000, get_default(primary_llm,   "max_tokens", 1000), key="l1_tok")
    with card2:
        with st.container(border=True):
            st.markdown("🟡 **Secondary**")
            l2_temp   = st.slider("Temperature", 0.0, 1.0, get_default(secondary_llm, "temperature", 0.0), 0.05, key="l2_temp")
            l2_tokens = st.number_input("Max Tokens", 100, 4000, get_default(secondary_llm, "max_tokens", 1000), key="l2_tok")
    with card3:
        with st.container(border=True):
            st.markdown("🔴 **Expert**")
            l3_temp   = st.slider("Temperature", 0.0, 1.0, get_default(expert_llm,    "temperature", 0.0), 0.05, key="l3_temp")
            l3_tokens = st.number_input("Max Tokens", 100, 4000, get_default(expert_llm,    "max_tokens", 2000), key="l3_tok")

# ══════════════════════════════════════════════
# SECTION 5 ▸ retrieval / parallel / observability
# ══════════════════════════════════════════════
retr = cfg_existing.get("retrieval",     {})
par  = cfg_existing.get("parallel",      {})
obs  = cfg_existing.get("observability", {})

with st.expander("🔎 **retrieval**", expanded=False):
    n_exemplars_pool = st.number_input("n_exemplars_pool", 32, 2048,
                                        int(retr.get("n_exemplars_pool", 256)), step=32)

with st.expander("⚡ **parallel**", expanded=False):
    pw1, pw2, pw3 = st.columns(3)
    embedding_workers = pw1.number_input("embedding_workers", 1, 32, int(par.get("embedding_workers", 8)))
    inference_workers = pw2.number_input("inference_workers", 1, 16, int(par.get("inference_workers", 2)))
    embedding_batch   = pw3.number_input("embedding_batch_size", 1, 100, int(par.get("embedding_batch_size", 5)))

with st.expander("📡 **observability**", expanded=False):
    ow1, ow2 = st.columns(2)
    obs_enabled  = ow1.toggle("enabled",  value=bool(obs.get("enabled", True)))
    obs_provider = ow2.selectbox("provider", ["langfuse"], disabled=not obs_enabled)

# ══════════════════════════════════════════════
# SECTION 6 ▸ Seed Prompt
# ══════════════════════════════════════════════
with st.expander("📝 **Seed Prompt**", expanded=True):
    st.caption("Paste or edit your system prompt here. Saved to `work_station/prompt.txt` on initialisation.")
    prompt_abs     = ws_path / "prompt.txt"
    default_prompt = prompt_abs.read_text(encoding="utf-8") if prompt_abs.exists() else ""
    seed_prompt    = st.text_area("Classification instructions / meta rules:", value=default_prompt, height=220)

# ──────────────────────────────────────────────
# YAML 预览
# ──────────────────────────────────────────────
st.divider()

def _build_config() -> dict:
    return {
        "project":        {"name": proj_name, "version": proj_version, "modality": modality,
                           "work_station": str(ws_path), "data_path": data_path,
                           "image_dir": image_dir, "outfiles_dir": outfiles_dir},
        "column_mapping": {"text_col": text_col, "image_col": image_col, "answer_col": answer_col},
        "user_settings":  {"lambda_param": lambda_param, "k_shots": int(k_shots),
                           "options": options_list, "testing": testing, "testing_size": int(testing_size)},
        "models":         {"embedding_llm": embedding_llm, "primary_llm": primary_llm,
                           "secondary_llm": secondary_llm, "expert_llm": expert_llm},
        "retrieval":      {"n_exemplars_pool": int(n_exemplars_pool)},
        "parallel":       {"embedding_workers":    int(embedding_workers),
                           "inference_workers":    int(inference_workers),
                           "embedding_batch_size": int(embedding_batch)},
        "observability":  {"enabled": obs_enabled, "provider": obs_provider},
    }

with st.expander("👁️ Preview  `config.yaml`", expanded=False):
    st.code(yaml.dump(_build_config(), sort_keys=False, allow_unicode=True), language="yaml")

# ──────────────────────────────────────────────
# 初始化
# ──────────────────────────────────────────────
if st.button("🚀  Save Config & Initialise Classifier", type="primary", use_container_width=True):
    try:
        cfg_dict   = _build_config()
        config_out = ws_path / "config.yaml"

        ws_path.mkdir(parents=True, exist_ok=True)
        prompt_abs.write_text(seed_prompt, encoding="utf-8")
        with open(config_out, "w", encoding="utf-8") as f:
            yaml.dump(cfg_dict, f, sort_keys=False, allow_unicode=True)

        root_env = _THIS_DIR / ".env"

        ui_model_configs = {
            "l1": {"model": primary_llm,   "temperature": float(l1_temp), "max_tokens": int(l1_tokens)},
            "l2": {"model": secondary_llm, "temperature": float(l2_temp), "max_tokens": int(l2_tokens)},
            "l3": {"model": expert_llm,    "temperature": float(l3_temp), "max_tokens": int(l3_tokens)},
        }

        from poliprompt import TextClassifier, MultiModalClassifier
        ClassifierCls = MultiModalClassifier if modality == "multimodal" else TextClassifier
        clf = ClassifierCls(
            env_path=root_env, config_path=config_out, prompt_path=prompt_abs,
            ui_model_configs=ui_model_configs,
        )

        # 注入 HITL 队列
        hitl_req_q = queue.Queue()
        hitl_res_q = queue.Queue()
        clf.hitl_request_queue  = hitl_req_q
        clf.hitl_response_queue = hitl_res_q

        st.session_state["classifier"]      = clf
        st.session_state["hitl_req_q"]      = hitl_req_q
        st.session_state["hitl_res_q"]      = hitl_res_q
        st.session_state["init_done"]       = True
        st.session_state["annotate_running"] = False

        # 自动检测已有文件
        outfiles_abs = ws_path / outfiles_dir
        pool_exists  = ((outfiles_abs / "embeddings.index").exists() and
                        (outfiles_abs / "exemplar_indices.json").exists())
        opt_exists   = ((outfiles_abs / "rules.json").exists() and
                        (outfiles_abs / "enhanced_rules.txt").exists())
        st.session_state["pool_done"] = pool_exists
        st.session_state["opt_done"]  = opt_exists

        if pool_exists:
            st.info("⏩ Pool files detected — Step 1 will be skipped automatically.")
        if opt_exists:
            st.info("⏩ Rules files detected — Step 2 will be skipped automatically.")

        st.success(f"✅ config.yaml → `{config_out}`")
        st.balloons()

    except Exception as e:
        st.error(f"❌ Init error: {e}")
        st.exception(e)

# ──────────────────────────────────────────────
# 工作流步骤
# ──────────────────────────────────────────────
if st.session_state.get("init_done"):
    st.divider()
    st.subheader("🔬 Run Workflow")
    clf      = st.session_state["classifier"]
    hitl_req = st.session_state.get("hitl_req_q")
    hitl_res = st.session_state.get("hitl_res_q")

    s1, s2, s3 = st.columns(3)

    # ── Step 1 ──────────────────────────────────
    with s1:
        st.markdown("##### Step 1 · Build Pool")
        st.caption("Embed data → KMeans → FAISS index")
        if st.button("🔨  Build Pool", use_container_width=True,
                     disabled=st.session_state.get("pool_done", False)):
            log_ph = st.empty()
            ok, err = run_with_live_log(clf.create_few_shot_pool, log_ph)
            if ok:
                st.session_state["pool_done"] = True
                st.success("✅ Pool ready!")
            else:
                st.error(f"❌ {err}")

    # ── Step 2 ──────────────────────────────────
    with s2:
        st.markdown("##### Step 2 · Optimise Rules")
        st.caption("Map-Reduce logic synthesis (RAFS)")
        if st.button("🧠  Optimise", use_container_width=True,
                     disabled=not st.session_state.get("pool_done", False)
                              or st.session_state.get("opt_done", False)):
            log_ph = st.empty()
            ok, err = run_with_live_log(clf.optimize_task_description, log_ph)
            if ok:
                st.session_state["opt_done"] = True
                st.success("✅ Rules ready!")
            else:
                st.error(f"❌ {err}")

    # ── Step 3 启动按钮 ──────────────────────────
    with s3:
        st.markdown("##### Step 3 · Annotate")
        st.caption("Parallel multi-agent batch inference")
        if st.button("⚡  Annotate", use_container_width=True,
                     disabled=not st.session_state.get("opt_done", False)
                              or st.session_state.get("annotate_running", False)):

            orig_out, orig_err = sys.stdout, sys.stderr
            log_q  = queue.Queue()
            writer = _QueueWriter(log_q, orig_out)
            sys.stdout = writer
            sys.stderr = _QueueWriter(log_q, orig_err)

            err_box = [None]
            def _run_annotate():
                try:
                    clf.annotate()
                except Exception as e:
                    err_box[0] = e
                finally:
                    sys.stdout = orig_out
                    sys.stderr = orig_err

            t = threading.Thread(target=_run_annotate, daemon=True)
            t.start()

            st.session_state["annotate_thread"]  = t
            st.session_state["annotate_log_q"]   = log_q
            st.session_state["annotate_err_box"] = err_box
            st.session_state["annotate_lines"]   = []
            st.session_state["annotate_running"] = True
            st.rerun()

# ── Annotate 轮询块（每次 rerun 都会执行）──────
if st.session_state.get("annotate_running"):
    st.divider()
    st.markdown("#### 📋 Annotation Progress")
    log_ph = st.empty()

    log_q = st.session_state.get("annotate_log_q")
    lines = st.session_state.get("annotate_lines", [])

    # 收集新日志
    if log_q:
        try:
            while True:
                lines.append(log_q.get_nowait())
        except queue.Empty:
            pass
    st.session_state["annotate_lines"] = lines
    _render_log(log_ph, lines)

    # 检查线程是否结束
    t = st.session_state.get("annotate_thread")
    thread_done = t and not t.is_alive()

    if thread_done:
        st.session_state["annotate_running"] = False
        err = st.session_state.get("annotate_err_box", [None])[0]
        if err:
            st.error(f"❌ Annotation error: {err}")
        else:
            st.success("✅ Annotation complete!")
        st.rerun()
    else:
        # 检查 HITL 请求
        # 关键：检测到 HITL 后不立刻 rerun，让脚本继续往下执行渲染卡片
        if hitl_req and not st.session_state.get("hitl_pending"):
            try:
                req = hitl_req.get_nowait()
                st.session_state["hitl_pending"] = req
                # 不 rerun！让脚本继续执行到 HITL 卡片渲染部分
            except queue.Empty:
                # 没有 HITL 请求，继续轮询日志
                if not st.session_state.get("hitl_pending"):
                    time.sleep(0.5)
                    st.rerun()

# ── HITL 卡片──────────────────────────────────
if st.session_state.get("hitl_pending"):
    req  = st.session_state["hitl_pending"]
    idx  = req["idx"]
    item = req["item"]
    opts = req["options"]

    st.divider()
    st.warning(f"👤 **Human Label Required — Row {idx}**")

    with st.container(border=True):
        col_text, col_img = st.columns([2, 1])
        with col_text:
            st.markdown("**Text content:**")
            st.write(item.get("text", ""))
        with col_img:
            img_path = item.get("image_path")
            if img_path and Path(img_path).exists():
                st.image(img_path, caption=f"Row {idx}", use_container_width=True)
            else:
                st.caption("No image / image not found")

        chosen = st.radio("Select label:", opts, horizontal=True, key=f"hitl_radio_{idx}")
        reason = st.text_input("Reason (optional):", key=f"hitl_reason_{idx}")

        if st.button("✅  Confirm Label", type="primary", key=f"hitl_confirm_{idx}"):
            hitl_res.put(chosen)
            st.session_state.pop("hitl_pending", None)
            st.rerun()

# ── 结果预览──────────────────────────────────
if st.session_state.get("init_done"):
    backup_csv = ws_path / outfiles_dir / f"{proj_name}_backup.csv"
    if backup_csv.exists():
        import pandas as pd
        st.divider()
        st.subheader("📊 Results Preview")
        df_res = pd.read_csv(backup_csv)
        st.dataframe(df_res, use_container_width=True, height=320)
        st.download_button(
            "⬇️ Download CSV",
            data=df_res.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            file_name=backup_csv.name, mime="text/csv",
        )
