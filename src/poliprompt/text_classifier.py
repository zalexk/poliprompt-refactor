import os
import logging
import numpy as np
from typing import List, Dict, Tuple, Any, Optional # 🚀 补齐了 Any
from langchain_core.messages import SystemMessage, HumanMessage

from .base_classifier import BaseClassifier
from .retrieves import select_kshots
from . import utils

logger = logging.getLogger(__name__)

class TextClassifier(BaseClassifier):
    """
    PoliPrompt v3.2 纯文本分类器
    完全对齐 BaseClassifier 契约，支持 LangChain 0.3.0
    """

    def _convert_df_to_docs(self, df):
        """将 DataFrame 转换为纯文本列表"""
        # return utils.read_docs_from_dataframe(df, self.text_col)
        docs = []
        for _, row in df.iterrows():
            docs.append({
               "text": str(row[self.text_col]) 
                # 这里不需要 image_path 键，select_kshots 会自动处理 None
            })
        return docs

    # def _get_embeddings(self, docs: List[str]) -> np.ndarray:
    #     """获取文本嵌入"""
    #     return get_openai_embeddings(
    #         docs, 
    #         batch_size=100, 
    #         # model=self.embedding_llm_name
    #     )
    def _get_embeddings(self, docs_dicts: List[dict]) -> np.ndarray:
        # 直接调用父类统一封装好的逻辑即可，或者在此处传参
        from .llm_contribs import get_universal_embeddings
        return get_universal_embeddings(
                    docs_dicts,
                    self.embedding_llm_name,
                    max_workers=self.embedding_workers,
                    batch_size=self.embedding_batch_size,
                )

    def _prepare_agent_messages(self, idx: int, item: dict, is_expert: bool = False):
        """
        构建纯文本推理消息
        """
        # 1. 检索 Few-shot (image_col 传 None)
        examples, dists, labels, r_indices = select_kshots(
            self.df, self.text_col, None, self.answer_col, 
            self.k_shots, idx, self.indices, self.faiss_index, pool_embeddings=self.pool_embeddings_cache,
            lambda_param=self.lambda_param, options=self.options,
            metric="cosine",
            rules_dict=self.rules_dict
        )
        
        # 2. 构造 System Message
        sys_content = self.prompt_text 
        if self.enhanced_rules:
            sys_content += f"\n\n### GLOBAL REASONING RULES:\n{self.enhanced_rules}"
        
        # 3. 构造 User Message (拼接 Few-shot 逻辑)
        few_shot_text = ""
        for i, ex in enumerate(examples):
            few_shot_text += (
                f"Example {i+1}:\n"
                f"Text: {ex['content']}\n"
                f"Label: {ex['answer']}\n"
                f"Reasoning: {ex['explanation']}\n"
                f"---\n"
            )

        user_content = (
            f"### REFERENCE EXAMPLES:\n{few_shot_text}\n"
            f"### CURRENT TASK:\n"
            f"Text: {item[self.text_col]}\n\n"
            f"Please output your decision as a JSON object with 'label' and 'reason' fields."
        )

        return [SystemMessage(content=sys_content), HumanMessage(content=user_content)], dists, labels, r_indices

    def _display_item_for_hitl(self, item: dict):
        """终端/Jupyter 文本展示"""
        print("-" * 50)
        print(f"📄 TEXT CONTENT:\n{item[self.text_col]}")
        print("-" * 50)