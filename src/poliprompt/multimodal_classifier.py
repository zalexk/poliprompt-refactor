import os
import logging
import numpy as np
from typing import List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage

from .base_classifier import BaseClassifier
from . import utils                                 
from .utils import encode_image, get_base64_image   
from .retrieves import select_kshots


class MultiModalClassifier(BaseClassifier):

    def _convert_df_to_docs(self, df):
        """将数据框转为带图片路径的字典列表"""
        return utils.read_multimodal_docs_from_dataframe(
            df, self.text_col, self.image_col, self.image_dir
        )

    # def _get_embeddings(self, docs: List[dict]) -> np.ndarray:
    #     """调用多线程 Qwen-VL 嵌入引擎"""
    #     # 注意：这里调用你之前优化的那个 get_qwen_embeddings
    #     return get_qwen_embeddings(
    #         docs, 
    #         batch_size=5, 
    #         model=self.embedding_llm_name, 
    #         max_workers=self.num_workers
    #     )
    
    # def _get_embeddings(self, docs: List[dict]) -> np.ndarray:
    #     # 🚀 删掉 os.getcwd() 和 os.chdir()
    #     # 现在的 docs 里面已经是 BaseClassifier 帮你拼好的绝对路径了
    #     conf = self.embedding_llm_configs.get(self.embedding_llm_name) # 这里的 name 是 "qwen"
    #     real_model_name = conf.get("model")
    #     return get_qwen_embeddings(
    #         docs, 
    #         model=real_model_name, 
    #         max_workers=self.num_workers
    #     )

    def _get_embeddings(self, docs: List[dict]) -> np.ndarray:
        from .llm_contribs import get_universal_embeddings
        conf = self.embedding_llm_configs.get(self.embedding_llm_name, {})
        real_model_name = conf.get("model")
        return get_universal_embeddings(docs, real_model_name, self.num_workers)


    def _prepare_agent_messages(self, idx: int, item: dict, is_expert: bool = False):
        """
        多模态版本：拼接图文 Few-shot + 当前图文 Task
        """
        # 1. 检索 K-Shot (强制使用 Cosine 对齐 3.15 结论)
        examples, dists, labels, r_indices = select_kshots(
            self.df, self.text_col, self.image_col, self.answer_col, 
            self.k_shots, idx, self.indices, self.faiss_index,pool_embeddings=self.pool_embeddings_cache,
            lambda_param=self.lambda_param, options=self.options,
            metric="cosine",
            rules_dict=self.rules_dict
        )
        
        # 2. 构建 System Content (Seed Prompt + 动态规则)
        system_content = self.prompt_text 
        if hasattr(self, 'enhanced_rules') and self.enhanced_rules:
            system_content += f"\n\n### CRITICAL RULES FROM HUMAN EXPERTS:\n{self.enhanced_rules}"
        
        # 3. 初始化 User Payload (多模态列表格式)
        user_payload = []
        user_payload.append({"type": "text", "text": "### REFERENCE EXAMPLES (Analyze both Image and Text):\n"})

        # --- 遍历 Few-shot 示例 ---
        for i, ex in enumerate(examples):
            user_payload.append({
                "type": "text",
                "text": f"Example {i+1}:\nText: {ex['content']}\nAnswer: {ex['answer']}\nReasoning: {ex['explanation']}"
            })
            
            # 处理示例图
            img_path = ex.get('image_path')
            if img_path and os.path.exists(img_path):
                b64_img = get_base64_image(img_path)
                # b64_img = encode_image(img_path)
                user_payload.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_img}", "detail": "low"}
                })
            user_payload.append({"type": "text", "text": "---\n"})

        # 4. 处理当前待测任务
        user_payload.append({"type": "text", "text": f"\n### CURRENT TASK:\nText: {item[self.text_col]}"})
        
        if item.get('image_path') and os.path.exists(item['image_path']):
            b64_query_img = get_base64_image(item['image_path'])
            # 💡 亮点：如果是专家模式(L3)，给高清晰度图
            detail_level = "high" if is_expert else "low"
            user_payload.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_query_img}", "detail": detail_level}
            })

        messages = [SystemMessage(content=system_content), HumanMessage(content=user_payload)]
        return messages, dists, labels, r_indices

    
    def _display_item_for_hitl(self, item: dict):
        """Jupyter 环境下的图片展示"""
        from PIL import Image
        from IPython.display import display
        print(f"Text Content: {item[self.text_col]}")
        img_path = item.get('image_path')
        if img_path and os.path.exists(img_path):
            img = Image.open(img_path)
            display(img)
        else:
            print(f"⚠️ Image file missing: {img_path}")