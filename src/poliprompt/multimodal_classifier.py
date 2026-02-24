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

    l1_results: List[str]   # Model A 的 3 个结果
    l2_result: str          # Model B 的 1 个结果
    l3_results: List[str]   # Advanced 的 2 个结果 (CoT)

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
    
    # def _prepare_agent_messages(self, idx, item, include_image=True, is_cot=False):
    #     """
    #     全能型消息拼装：
    #     1. 自动根据 RAG 选出 k-shots
    #     2. 动态拼装 System Prompt (基础 + MapReduce 规则)
    #     3. 动态拼装 User Content (文字 + 图片)
    #     """
    #     # --- 1. 获取 Few-shot 示例 (复用 select_kshots) ---
    #     # 注意：这里需要导入或者确保 select_kshots 在作用域内
    #     examples = select_kshots(
    #         self.df, self.feature_col, self.answer_col, self.kshots, 
    #         idx, self.indices, self.faiss_index, self.lambda_param, self.options
    #     )
        
    #     # --- 2. 动态构建 System Prompt ---
    #     # 这里的 self.current_prefix 是你在 annotate 开头读取的那个原始 prompt
    #     system_content = self.current_prefix 
        
    #     # 重点：如果之前跑过 optimize 任务，这里直接把生成的规则加上去，实现动态闭环
    #     if hasattr(self, 'enhanced_rules') and self.enhanced_rules:
    #         system_content += f"\n\n### Critical Rules from Human Exemplars:\n{self.enhanced_rules}"
            
    #     system_content += "\nClassify the following political content accurately."

    #     # --- 3. 动态构建 User Content ---
    #     user_text = "### Reference Examples (Few-shot):\n"
    #     for i, ex in enumerate(examples):
    #         user_text += f"Example {i+1}:\nText: {ex['content']}\nAnswer: {ex['answer']}\n---\n"
        
    #     user_text += f"\n### Current Task:\nText: {item['text']}\n"
        
    #     if is_cot:
    #         user_text += "\nRequirement: Think step by step. Analyze the visual cues and textual context before choosing the label."
        
    #     user_text += f"\nFinal Step: Select one label from the options: [{', '.join(self.options)}]."

    #     # 组装消息列表
    #     # 显式声明类型为 Any，Pylance 就会闭嘴，且不需要加 ignore
    #     user_payload: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
        
    #     # 处理图片 (多模态开关)
    #     if include_image and item.get('image_path'):
    #         # 这里你可以根据 logic 传入 low 或 high
    #         # 建议：L1 用 low 省钱，L3 用 high 保准
    #         detail_level = "high" if is_cot else "low" 
            
    #         from .utils import encode_image # 确保路径正确
    #         b64_img = encode_image(item['image_path'])
    #         user_payload.append({
    #             "type": "image_url",
    #             "image_url": {
    #                 "url": f"data:image/jpeg;base64,{b64_img}",
    #                 "detail": detail_level  # 顺手把这个优化也加上
    #             }
    #         })

    #     return [
    #         SystemMessage(content=system_content),
    #         HumanMessage(content=user_payload)
    #     ]

    def _prepare_agent_messages(self, idx, item, include_image=True, is_cot=False):
        # 1. 获取示例
        examples = select_kshots(self.df, self.feature_col, self.answer_col, self.kshots, 
                                idx, self.indices, self.faiss_index, self.lambda_param, self.options)
        
        system_content = self.current_prefix 
        if hasattr(self, 'enhanced_rules') and self.enhanced_rules:
            system_content += f"\n\n### Critical Rules:\n{self.enhanced_rules}"
        
        # 2. 构建 User Content 列表 (List[Dict[str, Any]])
        user_payload: List[Dict[str, Any]] = []
        user_payload.append({"type": "text", "text": "### Reference Examples (Analyze both Image and Text):\n"})

        # --- 【核心改动：把示例的图片也塞进去】 ---
        for i, ex in enumerate(examples):
            # 添加示例文字
            user_payload.append({
                "type": "text", 
                "text": f"Example {i+1}:\nText: {ex['content']}\nAnswer: {ex['answer']}"
            })
            # 添加示例图片 (如果有路径)
            if ex.get('image_path'):
                b64_ex_img = encode_image(ex['image_path'])
                user_payload.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_ex_img}",
                        "detail": "low"  # 示例图用 low 模式，省钱！
                    }
                })
            user_payload.append({"type": "text", "text": "---\n"})

        # 3. 添加当前任务
        task_text = f"\n### Current Task:\nText: {item['text']}\n"
        if is_cot:
            task_text += "Requirement: Think step by step. Analyze the visual cues and textual context before choosing the label."
        task_text += f"\nFinal Step: Select one label from the options: [{', '.join(self.options)}]."
        
        user_payload.append({"type": "text", "text": task_text})
        
        # 4. 添加当前样本图片
        if include_image and item.get('image_path'):
            detail_level = "high" if is_cot else "low" 
            b64_img = encode_image(item['image_path'])
            user_payload.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_img}",
                    "detail": detail_level
                }
            })

        return [
            SystemMessage(content=system_content),
            HumanMessage(content=user_payload)
        ]
    
    # --- Layer 1: Base Model A (3 Inferences) ---
    # def _l1_node(self, state: AgentState):
    #     idx = state["pending_indices"][0] 
    #     item = state["multimodal_data"][idx]
        
    #     messages = self._prepare_agent_messages(idx, item, include_image=True, is_cot=False)
        
    #     results = []
    #     l1_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        
    #     print(f"--- [Row {idx}] Layer 1 (Base Model A) Analyzing (3 times) ---")
        
    #     for i in range(3):
    #         # 重要：现在的 response 是一个字典 {"parsed": ..., "raw": ...}
    #         raw_res = self.structured_llm_a.invoke(messages)
            
    #         # 提取标签
    #         parsed = raw_res["parsed"]
    #         results.append(parsed.label)
            
    #         # 提取 Token
    #         raw_msg = raw_res["raw"]
    #         if hasattr(raw_msg, "usage_metadata"):
    #             usage = raw_msg.usage_metadata
    #             l1_usage["prompt_tokens"] += usage.get("input_tokens", 0)
    #             l1_usage["completion_tokens"] += usage.get("output_tokens", 0)

    #     new_usage = state["token_usage"].copy()
    #     new_usage["prompt_tokens"] += l1_usage["prompt_tokens"]
    #     new_usage["completion_tokens"] += l1_usage["completion_tokens"]
    #     new_usage["total_cost"] += calculate_token_cost(self.llm_a_name, l1_usage["prompt_tokens"], l1_usage["completion_tokens"])

    #     return {
    #         "current_index": idx,
    #         "l1_results": results,
    #         "token_usage": new_usage
    #     }

    def _l1_node(self, state: AgentState):
        idx = state["pending_indices"][0] 
        item = state["multimodal_data"][idx]
        messages = self._prepare_agent_messages(idx, item, include_image=True, is_cot=False)
        
        print(f"--- [Row {idx}] Layer 1 (Model A) Analyzing (1 time) ---")
        raw_res = self.structured_llm_a.invoke(messages)
        
        # 更新 Token 消耗
        new_usage = state["token_usage"].copy()
        if hasattr(raw_res["raw"], "usage_metadata"):
            usage = raw_res["raw"].usage_metadata
            new_usage["prompt_tokens"] += usage.get("input_tokens", 0)
            new_usage["completion_tokens"] += usage.get("output_tokens", 0)
            new_usage["total_cost"] += calculate_token_cost(self.llm_a_name, usage.get("input_tokens", 0), usage.get("output_tokens", 0))

        return {
            "current_index": idx,
            "l1_results": [raw_res["parsed"].label], # 存入列表，保持格式统一
            "token_usage": new_usage
        }

    # --- Layer 2: Secondary Model B (1 Inference) ---
    def _l2_node(self, state: AgentState):
        idx = state["current_index"] # 注意这里用 current_index，因为 L1 已经设置过了
        item = state["multimodal_data"][idx]
        
        messages = self._prepare_agent_messages(idx, item, include_image=True, is_cot=False)
        
        print(f"--- [Row {idx}] Layer 2 (Model B) Cross-checking ---")
        raw_res = self.structured_llm_b.invoke(messages)
        
        parsed = raw_res["parsed"]
        raw_msg = raw_res["raw"]

        new_usage = state["token_usage"].copy()
        if hasattr(raw_msg, "usage_metadata"):
            usage = raw_msg.usage_metadata
            p_tokens = usage.get("input_tokens", 0)
            c_tokens = usage.get("output_tokens", 0)
            new_usage["prompt_tokens"] += p_tokens
            new_usage["completion_tokens"] += c_tokens
            new_usage["total_cost"] += calculate_token_cost(self.llm_b_name, p_tokens, c_tokens)

        return {
            "l2_result": parsed.label,
            "token_usage": new_usage
        }
    # --- Layer 3: Advanced Model (2 Inferences with CoT) ---
    def _l3_node(self, state: AgentState):
        idx = state["current_index"]
        item = state["multimodal_data"][idx]
        
        messages = self._prepare_agent_messages(idx, item, include_image=True, is_cot=True)
        
        results = []
        l3_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        
        print(f"--- [Row {idx}] Layer 3 (Advanced Model) Resolving (2 times CoT) ---")
        
        for i in range(2):
            raw_res = self.structured_llm_adv.invoke(messages)
            
            parsed = raw_res["parsed"]
            results.append(parsed.label)
            
            raw_msg = raw_res["raw"]
            if hasattr(raw_msg, "usage_metadata"):
                usage = raw_msg.usage_metadata
                l3_usage["prompt_tokens"] += usage.get("input_tokens", 0)
                l3_usage["completion_tokens"] += usage.get("output_tokens", 0)

        new_usage = state["token_usage"].copy()
        new_usage["prompt_tokens"] += l3_usage["prompt_tokens"]
        new_usage["completion_tokens"] += l3_usage["completion_tokens"]
        new_usage["total_cost"] += calculate_token_cost(self.llm_adv_name, l3_usage["prompt_tokens"], l3_usage["completion_tokens"])

        return {
            "l3_results": results,
            "token_usage": new_usage
        }
    
    # def _auto_update_node(self, state: AgentState):
    #     idx = state["current_index"]
    #     l1 = state.get("l1_results", [])
    #     l2 = state.get("l2_result")
    #     l3 = state.get("l3_results", [])
        
    #     final_label = None
    #     path_taken = ""

    #     # 1. 如果 L1 三次全中 (3/3)
    #     if len(set(l1)) == 1 and len(l1) > 0:
    #         final_label = l1[0]
    #         path_taken = "L1_Consensus(3/3)"
            
    #     # 2. 如果 L1 是 2/1 分歧，看 L2 (B模型) 是不是同意那个“2”
    #     elif l2:
    #         l1_majority = Counter(l1).most_common(1)[0][0] # 选出 A 里的众数
    #         if l2 == l1_majority:
    #             final_label = l2
    #             path_taken = "L2_CrossMatch"
    #         else:
    #             # 如果 B 也不同意 A 的众数，这时候应该已经去 L3 了
    #             # 这个 elif 只是为了最终盖章
    #             pass
        
    #     # 3. 如果走到了 L3 (高级模型) 且 2/2 一致
    #     if not final_label and len(set(l3)) == 1 and len(l3) == 2:
    #         final_label = l3[0]
    #         path_taken = "L3_AdvancedConsensus"
            
    #     # 4. 兜底方案 (如果前面都没定论，比如 1/1/1 分歧且 L3 也打架)
    #     if not final_label:
    #         final_label = l3[0] if l3 else (l2 if l2 else l1[0])
    #         path_taken = "HITL"

    #     print(f"--- [Row {idx}] Finalizing: Label={final_label}, Path={path_taken} ---")

    #     # 1. 更新结果账本
    #     new_results = state["results"].copy()
    #     new_results[str(idx)] = {
    #         "label": final_label,
    #         "path": path_taken,
    #         "cost": state["token_usage"]["total_cost"] # 记录累计到这一行的成本
    #     }

    #     # 2. 从待办清单移除
    #     new_pending = state["pending_indices"][1:]

    #     print(f"Progress: {len(new_results)} / {len(state['multimodal_data'])} rows completed.")
        
    #     # 3. 重置临时状态，为下一行做准备
    #     return {
    #         "results": new_results,
    #         "pending_indices": new_pending,
    #         "l1_results": [],
    #         "l2_result": None,
    #         "l3_results": [],
    #         "temp_prediction": None,
    #         "final_path": path_taken # 存入状态以便导出
    #     }

    def _auto_update_node(self, state: AgentState):
        idx = state["current_index"]
        l1 = state.get("l1_results", [])
        l2 = state.get("l2_result")
        l3 = state.get("l3_results", [])
        
        from collections import Counter
        
        final_label = None
        path_taken = ""

        # --- v3.0 路径判定优先级逻辑 (从最深节点开始判) ---
        
        # 情况 A: 走到了 L3 (高级专家)
        if l3 and len(l3) > 0:
            if len(set(l3)) == 1:
                final_label = l3[0]
                path_taken = "L3_Expert_Consensus"
            else:
                final_label = l3[0] # 依然以 L3 第一次结果为准，但标记为 HITL 边缘
                path_taken = "L3_Expert_Inconsistent" # 实际上这种情况本应进 HITL，这里做个记录
        
        # 情况 B: 走到了 L2 (异构仲裁成功)
        elif l2 is not None:
            # 在 v3 逻辑中，如果能走到 auto_update 且有 l2 且没 l3，说明 A == B 成功
            final_label = l2
            path_taken = "L2_Heterogeneous_Match"
            
        # 情况 C: 兜底 (理论上 v3 至少会走完 L2)
        else:
            final_label = l1[0] if l1 else None
            path_taken = "L1_Initial_Exit"

        # 增加一个 HITL 的标记 (如果 state 里有这个标记)
        if state.get("final_path") == "HITL_Manual":
            # 如果是人工进来的，直接保留人工路径
            pass # human_label_node 已经处理过了

        print(f"--- [Row {idx}] Finalizing: Label={final_label}, Path={path_taken} ---")

        # 1. 更新结果账本
        new_results = state["results"].copy()
        new_results[str(idx)] = {
            "label": final_label,
            "path": path_taken,
            "cost": state["token_usage"]["total_cost"]
        }

        # 2. 从待办清单移除
        new_pending = state["pending_indices"][1:]
        print(f"Progress: {len(new_results)} / {len(state['multimodal_data'])} rows completed.")
        
        return {
            "results": new_results,
            "pending_indices": new_pending,
            "l1_results": [],
            "l2_result": None,
            "l3_results": [],
            "temp_prediction": None,
            "final_path": path_taken
        }
    
    def _human_label_node(self, state: AgentState):
        idx = state["current_index"]
        item = state["multimodal_data"][idx]
        
        print(f"\n--- [Row {idx}] HUMAN INTERVENTION REQUIRED ---")
        print(f"Logic: L1, L2, L3 all failed to reach a definitive consensus.")
        print(f"Text: {item['text']}")
        
        # 显示图片 (保持你之前的优秀习惯)
        from PIL import Image
        from IPython.display import display
        img = Image.open(item['image_path'])
        display(img)

        options_hint = "/".join(self.options)
        user_input = input(f"Please provide the definitive label ({options_hint}): ")

        # --- 核心新增：Active Learning 闭环 ---
        
        # 1. 把序号加入“精英名单”
        if idx not in self.indices:
            self.indices.append(idx)
            # 持久化序号 JSON
            exemplars_file = self.outfiles_dir / "exemplar_indices.json"
            with open(exemplars_file, "w") as f:
                json.dump(self.indices, f, indent=4)
        
        # 2. 【关键！】把人类给出的答案写回内存中的 DataFrame
        # 这样下一个样本进行 RAG 检索到这一行时，就能拿到 user_input 了
        self.df.at[idx, self.answer_col] = user_input
        
        print(f"✅ [Row {idx}] added to pool with label '{user_input}'. Next samples will use this as a reference.")

        # --- (记录结果到 state 的逻辑保持不变) ---
        new_results = state["results"].copy()
        new_results[str(idx)] = {
            "label": user_input,
            "path": "HITL_Manual",
            "cost": state["token_usage"]["total_cost"]
        }
        
        new_pending = state["pending_indices"][1:]

        return {
            "results": new_results, 
            "pending_indices": new_pending,
            "l1_results": [],
            "l2_result": None,
            "l3_results": [],
            "temp_prediction": None
        }
    
    # def _l1_router(self, state: AgentState):
    #     l1 = state["l1_results"]
    #     unique_count = len(set(l1))
        
    #     if unique_count == 1:
    #         return "auto_update"   # 3/3 一致
    #     if unique_count == 3:
    #         return "go_to_l3"      # 1/1/1 全分歧
    #     return "go_to_l2"

    def _l1_router(self, state: AgentState):
        # L1 跑完直接去 L2，不要判断了
        return "go_to_l2" 
    
    # def _l2_router(self, state: AgentState):
    #     # 这里的 a_majority 是 Model A 跑三次中出现两次的那个标签
    #     a_majority = Counter(state["l1_results"]).most_common(1)[0][0]
    #     if state["l2_result"] == a_majority:
    #         return "auto_update"   # B 赞同 A 的众数
    #     return "go_to_l3"          # B 反对 A 的众数，升级到 L3
    
    def _l2_router(self, state: AgentState):
        # 在这里进行“异构对冲”
        label_a = state["l1_results"][0]
        label_b = state["l2_result"]
        
        if label_a == label_b:
            return "auto_update" # 异构达成共识，直接结束
        return "go_to_l3"        # 异构吵架，升级到专家
    
    def _l3_router(self, state: AgentState):
        l3 = state["l3_results"]
        if len(set(l3)) == 1:      # Advanced 2/2 一致
            return "auto_update"
        return "human_label"   
    
    
    def _build_graph(self):
        builder = StateGraph(AgentState)
        
        # 1. 添加所有功能节点
        builder.add_node("predict_l1", self._l1_node)
        builder.add_node("predict_l2", self._l2_node)
        builder.add_node("predict_l3", self._l3_node)
        builder.add_node("auto_update", self._auto_update_node)
        builder.add_node("human_label", self._human_label_node)

        # 2. 设置起点
        builder.add_edge(START, "predict_l1")
        
        # 3. 设置核心路由：L1 之后的路口
        builder.add_conditional_edges("predict_l1", self._l1_router, {
            "auto_update": "auto_update",
            "go_to_l2": "predict_l2",
            "go_to_l3": "predict_l3"
        })
        
        # 4. 设置 L2 之后的路口
        builder.add_conditional_edges("predict_l2", self._l2_router, {
            "auto_update": "auto_update",
            "go_to_l3": "predict_l3"
        })
        
        # 5. 设置 L3 之后的路口
        builder.add_conditional_edges("predict_l3", self._l3_router, {
            "auto_update": "auto_update",
            "human_label": "human_label"
        })

        # 6. 处理循环逻辑：处理完当前行，是否有下一行？
        def _check_next_step(state: AgentState):
            if state["pending_indices"]:
                return "next"
            return "end"

        builder.add_conditional_edges("auto_update", _check_next_step, {
            "next": "predict_l1",
            "end": END
        })
        builder.add_conditional_edges("human_label", _check_next_step, {
            "next": "predict_l1",
            "end": END
        })

        # 7. 编译并加装“暂停器” (用于 HITL)
        return builder.compile(
            checkpointer=MemorySaver(), 
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
        # --- A. 初始化环境与路径 ---
        self.llm_a_name = llm_a_name
        self.llm_b_name = llm_b_name
        self.llm_adv_name = llm_adv_name
        
        dataset_name = self.data_file.stem
        llm_configs = self.configs_dir / "llm_configs.json"
        index_file = str(self.outfiles_dir / "embeddings.index")
        self.faiss_index = faiss.read_index(index_file) # 存入 self 供 Node 调用
        exemplars_file = self.outfiles_dir / "exemplar_indices.json"
        self.indices = json.loads(exemplars_file.read_text())
        self.kshots = kshots
        self.lambda_param = lambda_param

        # --- B. 初始化三个结构化模型 ---
        def _init_structured_llm(name):
            config = load_config(llm_configs, name)
            model = create_llm(llm_name=name, model_config=config)
            return model.with_structured_output(
                ClassificationResult, 
                method="function_calling", 
                include_raw=True
            )
        
        self.structured_llm_a = _init_structured_llm(llm_a_name)
        self.structured_llm_b = _init_structured_llm(llm_b_name)
        self.structured_llm_adv = _init_structured_llm(llm_adv_name)

        # --- C. 数据准备 ---
        self.df = load_and_validate_data(self.data_file, self.feature_col, self.answer_col, self.image_col)
        multimodal_data = read_multimodal_docs_from_dataframe(self.df, self.feature_col, self.image_col, self.image_dir)
        
        if testing:
            testing_size = min(testing_size, len(self.df)) if testing_size else len(self.df)
            # 1. 这里的 df 被切短了
            self.df = self.df[:testing_size].reset_index(drop=True) 
            multimodal_data = multimodal_data[:testing_size]
            
            # 2. 必须过滤 self.indices，确保里面的索引都在 testing_size 范围内
            # 否则 select_kshots 会去访问不存在的行
            self.indices = [idx for idx in self.indices if idx < testing_size]
            print(f"DEBUG: Testing mode active. Exemplar pool filtered to {len(self.indices)} items.")
            
            # 3. 还有一个隐患：如果过滤完之后池子空了，kshots 要设为 0
            if not self.indices:
                self.kshots = 0
                print("WARNING: No exemplars found within testing_size. Switching to zero-shot.")

        # 读取基础 Prompt (Seed Prompt)
        self.current_prefix = get_prompt(prompt_file=self.prompts_dir / prompt_file_name)

        # --- D. 构建与执行 Graph ---
        all_indices = self.df.index.tolist()
        initial_state = {
            "pending_indices": all_indices,
            "current_index": None,
            "results": {}, # 结果将是一个字典：{idx: {"label":..., "path":..., "cost":...}}
            "multimodal_data": multimodal_data,
            "l1_results": [], "l2_result": None, "l3_results": [],
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_cost": 0.0},
            "temp_prediction": None,
            "final_path": ""
        }

        if not hasattr(self, "agent_app"):
            self.agent_app = self._build_graph()

        config = {
            "configurable": {"thread_id": f"job_{dataset_name}"},
            "recursion_limit": 3000 # 增加这一行，设为 3000 确保能跑完 512 个样本
        }
        
        # 检查是否有 Checkpoint 可以恢复
        current_state = self.agent_app.get_state(config)
        input_data = initial_state if not current_state.values else None

        print(f"--- Starting Agentic Annotation (Total: {len(all_indices)}) ---")
        
        try:
            count = 0
            for event in self.agent_app.stream(input_data, config):
                if "auto_update" in event or "human_label" in event:
                    count += 1
                    # 每 10 个样本强制写一次文件，保命！
                    if count % 10 == 0:
                        snapshot = self.agent_app.get_state(config)
                        current_results = snapshot.values.get("results", {})
                        temp_df = self.df.copy()
                        for s_idx, res in current_results.items():
                            temp_df.at[int(s_idx), "predicted_label"] = res["label"]
                            temp_df.at[int(s_idx), "inference_path"] = res["path"]
                        temp_df.to_csv("../examples/TopicExperiment/outfiles/backup_v3.csv", index=False)
                        print(f"--- [Auto-Save] Progress synced at {count} rows ---")

                if "__interrupt__" in event:
                    print(f"\n[PAUSED] Row {self.agent_app.get_state(config).values.get('current_index')} needs HITL.")
                    return 

            # --- E. 结果导出与评估准备 ---
            final_snapshot = self.agent_app.get_state(config)
            final_results_dict = final_snapshot.values.get("results", {})
            
            # 整理结果存入 CSV
            output_df = self.df.copy()
            for str_idx, res in final_results_dict.items():
                idx = int(str_idx)
                output_df.at[idx, "predicted_label"] = res["label"]
                output_df.at[idx, "inference_path"] = res["path"]
                output_df.at[idx, "acc_cost_usd"] = res["cost"]
            
            # output_file = self.outfiles_dir / f"{dataset_name}_agent_results.csv"
            output_df.to_csv("../examples/TopicExperiment/outfiles/final_v3_results.csv", index=False)
            print(f"--- Mission Accomplished! Results saved to {self.outfiles_dir / 'final_v3_results.csv'} ---")
            
        except Exception as e:
            print(f"Error: {e}")