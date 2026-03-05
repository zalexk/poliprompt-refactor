import os
import sys
import time
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import List,Optional,TypedDict, Dict, Any #类型提示

import json
import yaml  
#一种配置文件的格式，比json更易读，负责把.yaml格式文件读成python字典
import numpy as np
from tqdm import tqdm
import pandas as pd
import faiss

from langchain_core.messages import SystemMessage, HumanMessage
from collections import Counter # 放在这里确保可用
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


from .utils import *
from .llm_contribs import create_llm, call_llm_wrapper, get_llm_embeddings, _prepare_multimodal_message, calculate_token_cost
from .reducers import create_reducer
from .selectors import create_selector
from .retrieves import select_kshots

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Set the logging level to WARNING to ignore INFO and DEBUG logs of LLM requests
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)
# 除非有warning，否则不需要打印

# Setup logger
logger = logging.getLogger(__name__)
# 给当前文件创建一个叫logger的记录员

class ClassificationResult(BaseModel):
    label: str = Field(description="Selected label from options")
    reason: str = Field(description="Reasoning for this choice")

class AgentState(TypedDict):
    pending_indices: List[int]
    current_index: Optional[int]
    multimodal_data: List[dict]

    l1_res: str               # 记录 L1 原始答案
    l2_res: str               # 记录 L2 原始答案
    l3_res_list: List[str]    # 记录 L3 两次 CoT 的答案
    
    rag_distances: List[float]
    rag_labels: List[str] 
    rag_indices: List[int]

    start_time: float 
    results: Dict[str, dict]
    final_path: str         # 记录该样本走过的路径 (e.g., "L1_3/3", "L3_Consensus", "HITL")
    token_usage: Dict[str, int] # {"prompt_tokens": 0, "completion_tokens": 0, "total_cost": 0.0} 
    temp_prediction: Optional[dict] 


class MultiModalClassifier:
    def __init__(
            self, 
            data_file: str | Path,
            # 希望传进来的变量是个字符串或者是path对象，不是的话也不会报错
            work_station: str | Path,
            env_path: str | Path,
            options: List[str],
            image_dir: Optional[str | Path] = None,
            feature_col: str="text",
            # 默认参数，实例化时没写这个参数时的默认值
            answer_col: str="label",
            image_col: str = "image",
            random_state: int=42,
            # 随机结果可重复，我把代码发给别人跑出来的结果也是一样的
            requests_per_period: int=60,
            secondf_per_period: int=60, 
        ):
        """

        """
        # The path to the environment file saving passwordf
        self.data_file = Path(os.path.expanduser(data_file))
        # os是python自带的操作系统，expanduser是展开home目录~，在不同电脑上都能展开
        # 把普通的“字符串路径”转换为一个path对象，操作更便捷
        # self.data = validate_csv_file(self.data_file)
        self.options = options
        self.image_dir = Path(os.path.expanduser(image_dir))
        self.feature_col = feature_col
        self.answer_col = answer_col
        self.image_col = image_col
        self.random_state = random_state
        self.requests_per_period = requests_per_period
        self.secondf_per_period = secondf_per_period

        # Create the working station
        self.work_station = Path(os.path.expanduser(work_station))
        ensure_workstation_directories(self.work_station)
        # 实际创建了这些目录和文件夹，下面的是把建好的文件路径储存在属性当中，以便后面的函数使用
        self.infiles_dir = self.work_station / "infiles"
        self.configs_dir = self.infiles_dir / "configs"
        self.prompts_dir = self.infiles_dir / "prompts"
        self.outfiles_dir = self.work_station / "outfiles"
        self.logs_dir = self.outfiles_dir / "logs"
        self.images_out_dir = self.outfiles_dir / "images"
        # 只给了一个work_station,自动规划“输入文件夹”，“配置文件文件夹”，“日志文件夹”等
        # /就是Path对象的作用，可以拼接路径
        
        # Load the passwordf of LLMs from environment file
        load_dotenv(Path(os.path.expanduser(env_path)), override=True)

    def create_few_shot_pool(
            self, 
            embedding_llm_name: str, 
            reduce_method: str,
            select_method: str,
            testing: bool=False,
            testing_size: int=None,
        ):

        # The path to save numpy array embeddings
        embeddings_file = self.outfiles_dir / "embeddings.index"
        # The path to save indices as json files
        exemplars_file = self.outfiles_dir / "exemplar_indices.json"

        # The path to load configurations
        logger.warning(f"You are supposed to provide 'embedding_llm_configs.json', 'reduce_configs.json', and 'select_configs.json' in the path {self.configs_dir}")
        # 提醒用户应该放置这几个config文件

        embedding_llm_config = load_config(self.configs_dir / "embedding_llm_configs.json", embedding_llm_name)
        reduce_config = load_config(self.configs_dir / "reduce_configs.json", reduce_method)
        # load_config返回一个字典，里面有要使用的具体方法的细节
        if "random_state" in reduce_config:
            reduce_config["random_state"] = self.random_state
        select_config = load_config(self.configs_dir / "select_configs.json", select_method)
        if "random_state" in select_config:
            select_config["random_state"] = self.random_state

        start_time = time.time()

        data = load_and_validate_data(self.data_file, self.feature_col, image_col=self.image_col)
            
            
        # Read dataset
        questions = read_multimodal_docs_from_dataframe(data, text_col=self.feature_col,img_col=self.image_col,img_root=self.image_dir)
        if testing:
            if testing_size is None:
                raise ValueError("testing_size cannot be None in the testing mode. It must be a valid integer.")
            if len(questions) < testing_size:
                logger.info(f"Provided testing_size ({testing_size}) is greater than the number of questions ({len(questions)}), "
                    f"we will only use the first {len(questions)} questions.")
                testing_size = len(questions)
            questions = questions[: testing_size]

        # Step 1: Embed all docs
        embeddings = get_llm_embeddings(embedding_llm_name, questions, **embedding_llm_config)
        # get_llm_embeddings有四个参数，**代表拆开所以这里填补了剩下的两个参数
        # Step 2: Dimension Reduction with UMAP
        reducer = create_reducer(reduce_method, **reduce_config)
        reduced_embeddings = reducer.reduce(embeddings)
        try:
            # np.save(embeddings_file, reduced_embeddings)
            faiss.normalize_L2(reduced_embeddings)
            dimension = reduced_embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(reduced_embeddings)
            faiss.write_index(index,str(embeddings_file))
            logger.info(f"NumPy array saved to {embeddings_file}")
        except Exception as e:
            raise Exception(f"An unexpected error occurred while saving the embeddings: {e}")

        # Step 3: Select exemplar docs
        selector = create_selector(method=select_method)
        exemplar_indices = selector.select_exemplars(reduced_embeddings, **select_config)
        exemplar_indices = [int(idx) for idx in exemplar_indices]

        # Step 4: Save exemplars to work station
        try:
            exemplars_file.write_text(json.dumps(exemplar_indices, indent=4))
            logger.info(f"List of integer indices saved to {exemplars_file}")
        except Exception as e:
            raise Exception(f"An unexpected error occurred while saving the exemplar indices: {e}")

        # Track and print the computation time
        end_time = time.time()
        elapsed = end_time - start_time
        hours, minutes, secondf = track_computation_time(elapsed)
        logger.info(f"Computation time: {int(hours):02}:{int(minutes):02}:{int(secondf):02}")
        logger.info(f"You are supposed to provid {self.answer_col} in {self.data_file} for the next steps.")


    def optimize_task_description(
            self,
            llm_name: str,
            prompt_file_name: str,
            disable_progress_bar: bool=False,
        ):
        rules_path = self.outfiles_dir / "rules.json"
        llm_configs = self.configs_dir / "llm_configs.json"
        prompt_file = self.prompts_dir / prompt_file_name

        model_config = load_config(llm_configs, llm_name)
        model = create_llm(llm_name=llm_name, model_config=model_config)

        seed_prompt = get_prompt(prompt_file=prompt_file)
        exemplars_file = self.outfiles_dir / "exemplar_indices.json"
        indices = json.loads(exemplars_file.read_text())

        df = load_and_validate_data(self.data_file, self.feature_col, self.answer_col, self.image_col)
        contexts = read_multimodal_docs_from_dataframe(df, self.feature_col, self.image_col, self.image_dir)
        answers = df[self.answer_col].tolist()
        
        exemplar_contexts = [contexts[i] for i in indices]
        exemplar_answers = [answers[i] for i in indices]

        # ==========================================================
        # PHASE 1: [MAP] - Individual Rule Extraction
        # ==========================================================
        if not os.path.isfile(rules_path):
            qa_sys_prompt = """You are a political scientist. Extract a specific classification rule 
            from this exemplar. Logic should start with 'The correct option is [LABEL]'."""
            
            extracted_rules = []
            print("--- PHASE 1 (MAP): Extracting Expert Logic from Exemplars ---")
            for ctx, ans in tqdm(zip(exemplar_contexts, exemplar_answers), total=len(indices), desc="Mapping"):
                msg_text = f"Text: {ctx['text']}\nHuman Answer: {ans}"
                messages = _prepare_multimodal_message(qa_sys_prompt.format(task=seed_prompt), msg_text, ctx['image_path'])
                response = model.invoke(messages)
                extracted_rules.append(response.content)
            rules_path.write_text(json.dumps(extracted_rules, indent=4))
        else:
            extracted_rules = json.loads(rules_path.read_text())

        # ==========================================================
        # PHASE 2: [REDUCE] - Rule Synthesis (via LCEL)
        # ==========================================================
        reduce_instruction = """You are a senior political research analyst. 
        We have extracted several specific classification rules from human-labeled exemplars.
        
        Task Context: {task}
        Extracted Individual Rules:
        {rules}
        
        Please synthesize these into a final, consolidated summary of actionable rules.
        Identify consensus, resolve contradictions, and provide a clear decision logic.
        CONCISE SUMMARY RULES:"""
        
        reduce_prompt = PromptTemplate.from_template(reduce_instruction)
        
        # 现代化的 LCEL 链，避开所有 Pydantic 报错
        reduce_chain = reduce_prompt | model | StrOutputParser()

        print("--- PHASE 2 (REDUCE): Synthesizing Global Knowledge ---")
        consolidated_rules = reduce_chain.invoke({
            "task": seed_prompt,
            "rules": "\n\n".join(extracted_rules)
        })

        # 更新实例属性并持久化，方便后续 Agent 节点调用
        self.enhanced_rules = consolidated_rules
        logger.info("Map-Reduce Optimization completed successfully.")
        
        # 最终返回总结出的规则，实现动态 Prompt 闭环
        return consolidated_rules
    

    def _prepare_agent_messages(self, idx, item, include_image=True, is_cot=False):
        # 1. 卸货：接住四元组
        examples, dists, labels, r_indices = select_kshots(
            self.df, self.feature_col, self.image_col, self.answer_col, 
            self.kshots, idx, self.indices, self.faiss_index, self.lambda_param, self.options
        )
        
        # 2. 构建 System Content
        system_content = self.current_prefix 
        if hasattr(self, 'enhanced_rules') and self.enhanced_rules:
            system_content += f"\n\n### Critical Rules from Human Exemplars:\n{self.enhanced_rules}"
        
        # 3. 初始化 User Payload
        user_payload: List[Dict[str, Any]] = []
        user_payload.append({"type": "text", "text": "### Reference Examples (Analyze both Image and Text):\n"})

        # --- 处理 Few-shot 示例 (带图) ---
        from .utils import encode_image
        for i, ex in enumerate(examples):
            user_payload.append({
                "type": "text", 
                "text": f"Example {i+1}:\nText: {ex['content']}\nAnswer: {ex['answer']}"
            })
            
            # 处理示例路径（确保路径完整）
            # 如果路径里没包含 image_dir，我们才补。
            raw_path = ex.get('image_path', '')
            full_ex_path = str(Path(self.image_dir) / raw_path) if str(self.image_dir) not in str(raw_path) else str(raw_path)
            
            if os.path.exists(full_ex_path):
                ext = Path(full_ex_path).suffix.lower()
                mime = "image/png" if ext == ".png" else "image/jpeg"
                b64_ex_img = encode_image(full_ex_path)
                user_payload.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64_ex_img}", "detail": "low"}
                })
            user_payload.append({"type": "text", "text": "---\n"})

        # 4. 处理当前任务 (Query)
        user_payload.append({"type": "text", "text": f"\n### Current Task:\nText: {item['text']}"})
        
        if include_image and item.get('image_path'):
            query_img_path = item['image_path'] # 这个在初始化时通常已经拼好了
            if os.path.exists(query_img_path):
                ext = Path(query_img_path).suffix.lower()
                mime = "image/png" if ext == ".png" else "image/jpeg"
                b64_query_img = encode_image(query_img_path)
                user_payload.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64_query_img}", "detail": "high" if is_cot else "low"}
                })

        messages = [SystemMessage(content=system_content), HumanMessage(content=user_payload)]
        
        # 返回：消息 + 观测数据
        return messages, dists, labels, r_indices
    
   
    def _l1_node(self, state: AgentState):
        idx = state["pending_indices"][0] 
        item = state["multimodal_data"][idx]
        
        # 1. 记录整行处理的开始时间
        row_start_time = time.time()
        
        # 2. 卸货：接住四元组 (消息, 距离, 标签, 原始索引)
        messages, dists, labels, r_indices = self._prepare_agent_messages(idx, item, include_image=True, is_cot=False)
        
        print(f"--- [Row {idx}] Layer 1 (Model A) Analyzing ---")
        
        # 3. 调用模型 (带 include_raw=True)
        raw_res = self.structured_llm_a.invoke(messages)
        parsed = raw_res["parsed"]
        raw_msg = raw_res["raw"]
        
        # 4. 更新 Token 和 成本
        new_usage = state["token_usage"].copy()
        if hasattr(raw_msg, "usage_metadata"):
            usage = raw_msg.usage_metadata
            new_usage["prompt_tokens"] += usage.get("input_tokens", 0)
            new_usage["completion_tokens"] += usage.get("output_tokens", 0)
            new_usage["total_cost"] += calculate_token_cost(self.llm_a_name, usage.get("input_tokens", 0), usage.get("output_tokens", 0))

        # 5. 返回所有观测指标
        return {
            "current_index": idx,
            "start_time": row_start_time,
            "l1_res": parsed.label,
            "temp_prediction": parsed.dict(), # 包含 reason，供后期字数分析
            "rag_distances": dists,
            "rag_labels": labels,
            "rag_indices": r_indices,
            "token_usage": new_usage
        }

    # --- Layer 2: Secondary Model B (1 Inference) ---
    def _l2_node(self, state: AgentState):
        idx = state["current_index"]
        item = state["multimodal_data"][idx]
        
        # 同样获取消息，但我们只取 messages 即可，dists 等已经在 L1 存过了
        messages, _, _, _ = self._prepare_agent_messages(idx, item, include_image=True, is_cot=False)
        
        print(f"--- [Row {idx}] Layer 2 (Model B) Cross-checking ---")
        try:
            raw_res = self.structured_llm_b.invoke(messages)
            parsed = raw_res["parsed"]
            raw_msg = raw_res["raw"]

            new_usage = state["token_usage"].copy()
            if hasattr(raw_msg, "usage_metadata"):
                usage = raw_msg.usage_metadata
                new_usage["prompt_tokens"] += usage.get("input_tokens", 0)
                new_usage["completion_tokens"] += usage.get("output_tokens", 0)
                new_usage["total_cost"] += calculate_token_cost(self.llm_b_name, usage.get("input_tokens", 0), usage.get("output_tokens", 0))

            return {
                "l2_res": parsed.label,
                "temp_prediction": parsed.dict(),
                "token_usage": new_usage
            }
        
        except Exception as e:
            # 4. 【核心改进】如果报错（比如安全审查 400），捕获它！
            print(f"⚠️ [Row {idx}] Layer 2 FAILED/BLOCKED. Error: {e}")
            print(f"--- [Row {idx}] Escalating to L3 due to Model B failure. ---")
            
            # 我们返回一个特殊的标志
            # 逻辑：因为 L1 的结果通常是 "0" 或 "1"，所以这个字符串肯定不相等
            # 路由器 _l2_router 看到不相等，就会自动送它去 L3
            return {
                "l2_res": "MODEL_B_FAIL_OR_BLOCKED", 
                "final_path": "L2_Safety_Bypass" # 记一笔，方便后期分析
            }
    
    # --- Layer 3: Advanced Model (2 Inferences with CoT) ---
    def _l3_node(self, state: AgentState):
        idx = state["current_index"]
        item = state["multimodal_data"][idx]
        
        # 专家层开启 is_cot=True
        messages, _, _, _ = self._prepare_agent_messages(idx, item, include_image=True, is_cot=True)
        
        l3_results = []
        new_usage = state["token_usage"].copy()
        last_parsed = None
        
        print(f"--- [Row {idx}] Layer 3 (Advanced Model) Resolving (2 times CoT) ---")
        
        for i in range(2):
            raw_res = self.structured_llm_adv.invoke(messages)
            last_parsed = raw_res["parsed"]
            l3_results.append(last_parsed.label)
            
            raw_msg = raw_res["raw"]
            if hasattr(raw_msg, "usage_metadata"):
                usage = raw_msg.usage_metadata
                new_usage["prompt_tokens"] += usage.get("input_tokens", 0)
                new_usage["completion_tokens"] += usage.get("output_tokens", 0)
                new_usage["total_cost"] += calculate_token_cost(self.llm_adv_name, usage.get("input_tokens", 0), usage.get("output_tokens", 0))

        return {
            "l3_res_list": l3_results,
            "temp_prediction": last_parsed.dict() if last_parsed else None,
            "token_usage": new_usage
        }
    
    def _human_label_node(self, state: AgentState):
        idx = state["current_index"]
        item = state["multimodal_data"][idx]
        
        print(f"\n--- [Row {idx}] HUMAN INTERVENTION REQUIRED ---")
        print(f"Logic: AI nodes failed to reach a high-confidence consensus.")
        print(f"Text: {item['text']}")
        
        # 1. 显示图片
        from PIL import Image
        from IPython.display import display
        img = Image.open(item['image_path'])
        display(img)

        # 2. 获取人工标签
        options_hint = "/".join(self.options)
        user_input = input(f"Please provide the definitive label ({options_hint}): ")

        # 3. Active Learning 闭环：更新内存 DF（确保 RAG 能搜到新标签）
        self.df.at[idx, self.answer_col] = user_input
        
        # 4. 如果不在精英池，追加进去并同步到磁盘
        if idx not in self.indices:
            self.indices.append(idx)
            with open(self.outfiles_dir / "exemplar_indices.json", "w") as f:
                json.dump(self.indices, f, indent=4)
            print(f"✅ [Row {idx}] added to Exemplar Pool.")

        # 5. 只返回人工决策结果，标记路径为 HITL_Manual
        return {
            "temp_prediction": {"label": user_input, "reason": "Human-in-the-loop validation"},
            "final_path": "HITL_Manual"
        }
   
    def _auto_update_node(self, state: AgentState):
        idx = state["current_index"]
        l1 = state.get("l1_res")
        l2 = state.get("l2_res")
        l3_list = state.get("l3_res_list", [])
        pred = state.get("temp_prediction", {})
        
        # --- A. 确定最终标签和判定路径 ---
        if state.get("final_path") == "HITL_Manual":
            final_label = pred.get("label")
            path_taken = "HITL_Manual"
        elif l2 == "SAFETY_BLOCKED": # <--- 【新增】识别安全拦截路径
            final_label = l3_list[0] if l3_list else l1
            path_taken = "L2_Safety_Bypass_to_L3"
        elif l2 is not None and l1 == l2:
            final_label = l2
            path_taken = "L2_Heterogeneous_Match"
        elif l3_list and len(l3_list) > 0:
            if len(set(l3_list)) == 1:
                final_label = l3_list[0]
                path_taken = "L3_Expert_Consensus"
            else:
                final_label = l3_list[0]
                path_taken = "L3_Expert_Inconsistent"
        else:
            final_label = l1 if l1 else None
            path_taken = "L1_Initial_Exit"

        # --- B. 计算观测指标 ---
        # 1. 异构共识分 (0-3)
        consensus_score = 0
        if l2 == "SAFETY_BLOCKED":
            consensus_score = -1 # <--- 【新增】用 -1 代表“由于安全策略导致的共识失效”
        elif l1 == l2: consensus_score = 3
        elif l3_list and len(set(l3_list)) == 1: consensus_score = 2
        elif l3_list: consensus_score = 1
        
        
        # 3. 整合 JSONL 对象 (全维度观察)
        obs_log = {
            "row_id": idx,
            "final_label": final_label,
            "path": path_taken,
            "consensus": consensus_score, 
            "is_safety_blocked": (l2 == "SAFETY_BLOCKED"),
            "rag": {
                "indices": state["rag_indices"],
                "avg_dist": sum(state["rag_distances"])/len(state["rag_distances"]) if state["rag_distances"] else 0
            },
            "performance": {
                "latency": time.time() - state["start_time"],
                "cost_usd": state["token_usage"]["total_cost"], # 钱存在这里
                "reason": pred.get("reason", ""),
                "reason_len": len(pred.get("reason", ""))
            }
        }

        # 实时写入 JSONL 
        with open(self.outfiles_dir / "observability_logs.jsonl", "a", encoding='utf-8') as f:
            f.write(json.dumps(obs_log, ensure_ascii=False) + "\n")

        # --- B. 更新主结果账本 (CSV 专供 - 极简版) ---
        new_results = state["results"].copy()
        new_results[str(idx)] = {
            "label": final_label,
            "path": path_taken  # 只存这两个，CSV 会非常干净
        }


        # --- D. 打印进度并清零状态 (关键！) ---
        print(f"--- [Row {idx}] Finalized via {path_taken}. Progress: {len(new_results)} / {len(state['multimodal_data'])} ---")

        return {
            "results": new_results,
            "pending_indices": state["pending_indices"][1:], # 移除当前行，移动物流
            "current_index": None,
            "l1_res": None,
            "l2_res": None,
            "l3_res_list": [],
            "rag_distances": [],
            "rag_labels": [],
            "rag_indices": [],
            "temp_prediction": None,
            "final_path": ""
        }
    

    def _l1_router(self, state: AgentState):
        return "go_to_l2" 
    
    def _l2_router(self, state: AgentState):
        label_a = state.get("l1_res") 
        label_b = state.get("l2_res")
        
        if label_a == label_b:
            return "auto_update" # 一致就结束
        return "go_to_l3"        # 不一致就升级
    
    def _l3_router(self, state: AgentState):
        l3 = state.get("l3_res_list", [])
        if len(set(l3)) == 1 and len(l3) == 2:
            return "auto_update" # 2/2 一致
        return "human_label"     # 1/1 分歧  
    
    
    def _build_graph(self,memory):
        builder = StateGraph(AgentState)
        
        # 1. 添加所有功能节点
        builder.add_node("predict_l1", self._l1_node)
        builder.add_node("predict_l2", self._l2_node)
        builder.add_node("predict_l3", self._l3_node)
        builder.add_node("auto_update", self._auto_update_node)
        builder.add_node("human_label", self._human_label_node)

        # 2. 设置起点
        builder.add_edge(START, "predict_l1")
        
        # 3. L1 到 L2 的路口 (因为你现在 L1 只跑1次，永远去 L2，其实这里可以简化)
        builder.add_conditional_edges("predict_l1", self._l1_router, {
            "auto_update": "auto_update", # 保留着为了以后扩展
            "go_to_l2": "predict_l2",
            "go_to_l3": "predict_l3"
        })
        
        # 4. L2 的路口
        builder.add_conditional_edges("predict_l2", self._l2_router, {
            "auto_update": "auto_update",
            "go_to_l3": "predict_l3"
        })
        
        # 5. L3 的路口
        builder.add_conditional_edges("predict_l3", self._l3_router, {
            "auto_update": "auto_update",
            "human_label": "human_label"
        })

        # ==========================================
        # 6. 【核心修复】人工标注完，必须去财务室盖章记账！
        # ==========================================
        builder.add_edge("human_label", "auto_update")

        # 7. 处理循环逻辑：只有在财务室（auto_update）结账后，才看有没有下一行
        def _check_next_step(state: AgentState):
            if state["pending_indices"]:
                return "next"
            return "end"

        # 所有的终点都汇聚在 auto_update，由它来决定是循环还是结束
        builder.add_conditional_edges("auto_update", _check_next_step, {
            "next": "predict_l1",
            "end": END
        })

        # 8. 编译并加装“暂停器” (用于 HITL)
        return builder.compile(
            checkpointer=memory, 
            interrupt_before=["human_label"]
        )
    
    def annotate(
            self, 
            llm_a_name: str,
            llm_b_name: str,
            llm_adv_name: str,
            prompt_file_name: str,
            kshots: int=0,
            lambda_param: float=1.0,
            testing: bool=False,
            testing_size: int=None,
        ):
        # --- 0. 路径与持久化准备 ---
        dataset_name = self.data_file.stem
        self.output_file = self.outfiles_dir / f"{dataset_name}_agent_results.csv"
        self.backup_file = self.outfiles_dir / f"{dataset_name}_backup_v3.csv"
        db_path = self.outfiles_dir / "poliprompt_checkpoints.db"

        # --- A. 初始化环境 ---
        self.llm_a_name = llm_a_name
        self.llm_b_name = llm_b_name
        self.llm_adv_name = llm_adv_name
        llm_configs = self.configs_dir / "llm_configs.json"
        self.faiss_index = faiss.read_index(str(self.outfiles_dir / "embeddings.index"))
        self.indices = json.loads((self.outfiles_dir / "exemplar_indices.json").read_text())
        self.kshots = kshots
        self.lambda_param = lambda_param

        # --- B. 初始化三个结构化模型 (带 Qwen 兼容逻辑) ---
        # def _init_structured_llm(name):
        #     config = load_config(llm_configs, name)
        #     model = create_llm(llm_name=name, model_config=config)
        #     # 自动适配 Qwen 或 OpenAI
        #     method = None if "qwen" in name.lower() else "function_calling"
            
        #     structured = model.with_structured_output(
        #         ClassificationResult, 
        #         method=method, 
        #         include_raw=True
        #     )
        #     return structured.with_retry(stop_after_attempt=3)
        def _init_structured_llm(name):
            config = load_config(llm_configs, name)
            model = create_llm(llm_name=name, model_config=config)
            
            # 准备参数字典
            structured_kwargs = {"include_raw": True}
            
            # 如果不是 Qwen，我们才显式指定使用 function_calling
            # 如果是 Qwen，就不传 method 键，让 LangChain 自动选择最兼容的模式
            if "qwen" not in name.lower():
                structured_kwargs["method"] = "function_calling"
            
            # 使用 ** 将字典解构为参数
            structured = model.with_structured_output(
                ClassificationResult, 
                **structured_kwargs
            )
            return structured.with_retry(stop_after_attempt=3)
        
        self.structured_llm_a = _init_structured_llm(llm_a_name)
        self.structured_llm_b = _init_structured_llm(llm_b_name)
        self.structured_llm_adv = _init_structured_llm(llm_adv_name)

        # --- C. 数据准备 (核心重构：图书馆模式) ---
        # 1. 永远加载全量数据，确保 RAG 能搜到所有的精英
        self.df = load_and_validate_data(self.data_file, self.feature_col, self.answer_col, self.image_col)
        multimodal_data = read_multimodal_docs_from_dataframe(self.df, self.feature_col, self.image_col, self.image_dir)
        
        # 2. 确定我们要处理哪些行 (Task Queue)
        all_indices = self.df.index.tolist()
        if testing and testing_size:
            # 只切待办列表，不切 df 本身！
            pending = all_indices[:testing_size]
        else:
            pending = all_indices
            
        print(f"--- 🚀 Agentic System Ready ---")
        print(f"Library Size: {len(self.df)} | Task Queue: {len(pending)}")

        self.current_prefix = get_prompt(prompt_file=self.prompts_dir / prompt_file_name)

        # --- D. 构建与执行 Graph (SqliteSaver) ---
        import sqlite3
        from langgraph.checkpoint.sqlite import SqliteSaver
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        memory = SqliteSaver(conn)

        if not hasattr(self, "agent_app"):
            self.agent_app = self._build_graph(memory)

        # 这里的 thread_id 建议包含 lambda 信息，方便你做 A/B Test
        config = {
            "configurable": {"thread_id": f"job_{dataset_name}_L{str(lambda_param).replace('.','')}"},
            "recursion_limit": 3000
        }
        
        initial_state = {
            "pending_indices": pending, # 使用刚才切好的待办单
            "current_index": None,
            "multimodal_data": multimodal_data, # 传入全量数据供检索
            "results": {}, 
            "l1_res": None, "l2_res": None, "l3_res_list": [],
            "rag_distances": [], "rag_labels":[], "rag_indices":[],
            "start_time": 0.0,
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_cost": 0.0},
            "temp_prediction": None, "final_path": ""
        }

        current_state = self.agent_app.get_state(config)
        input_data = initial_state if not current_state.values else None

        # --- 执行循环 ---
        try:
            count = 0
            for event in self.agent_app.stream(input_data, config):
                if "auto_update" in event:
                    count += 1
                    # 每 10 个样本自动同步一次 CSV 供肉眼查看
                    if count % 10 == 0:
                        snap = self.agent_app.get_state(config)
                        res_dict = snap.values.get("results", {})
                        temp_df = self.df.loc[pending].copy() # 只导出待办部分的
                        for s_idx, res in res_dict.items():
                            i = int(s_idx)
                            temp_df.at[i, "predicted_label"] = res["label"]
                            temp_df.at[i, "inference_path"] = res["path"]
                        temp_df.to_csv(self.backup_file)

                if "__interrupt__" in event:
                    print(f"\n[PAUSED] Row {self.agent_app.get_state(config).values.get('current_index')} needs HITL.")
                    return 

            # --- E. 结果最终收割 ---
            final_snapshot = self.agent_app.get_state(config)
            final_results_dict = final_snapshot.values.get("results", {})
            
            output_df = self.df.loc[pending].copy() 
            for str_idx, res in final_results_dict.items():
                idx = int(str_idx)
                if idx in output_df.index:
                    # 只填入 CSV 需要的字段
                    output_df.at[idx, "predicted_label"] = res["label"]
                    output_df.at[idx, "inference_path"] = res["path"]
            
            output_df.to_csv(self.output_file, index=False)
            print(f"--- ✅ Mission Accomplished! Results saved to {self.output_file} ---")

        except Exception as e:
            print(f"Error during execution: {e}")