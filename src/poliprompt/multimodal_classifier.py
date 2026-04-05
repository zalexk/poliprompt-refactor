import os
import logging
import numpy as np
from typing import List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage

from .base_classifier import BaseClassifier
from . import utils
from .utils import encode_image, get_base64_image
from .retrieves import select_kshots

logger = logging.getLogger(__name__)


class MultiModalClassifier(BaseClassifier):

    def _convert_df_to_docs(self, df):
        """将数据框转为带图片路径的字典列表"""
        return utils.read_multimodal_docs_from_dataframe(
            df, self.text_col, self.image_col, self.image_dir
        )

    def _get_embeddings(self, docs: List[dict]) -> np.ndarray:
        from .llm_contribs import get_universal_embeddings
        conf = self.embedding_llm_configs.get(self.embedding_llm_name, {})
        real_model_name = conf.get("model", self.embedding_llm_name)
        return get_universal_embeddings(
            docs,
            real_model_name,
            max_workers=self.embedding_workers,
            batch_size=self.embedding_batch_size,
        )

    def _abs_image_path(self, raw_path) -> str | None:
        """
        将任意来源的图片路径统一转换为绝对路径。
        - 如果已经是绝对路径且存在，直接返回。
        - 否则尝试 image_dir / raw_path 拼接。
        - 路径不存在则返回 None。
        """
        if not raw_path or str(raw_path).strip() in ("", "None", "nan"):
            return None
        p = str(raw_path)
        # 已经是绝对路径
        if os.path.isabs(p) and os.path.exists(p):
            return p
        # 尝试 image_dir 拼接
        if self.image_dir:
            candidate = str(self.image_dir / p) if hasattr(self.image_dir, '__truediv__') else os.path.join(str(self.image_dir), p)
            if os.path.exists(candidate):
                return candidate
        # 相对路径原样检查（极少情况）
        if os.path.exists(p):
            return p
        logger.warning(f"Image not found: {p}")
        return None

    def _prepare_agent_messages(self, idx: int, item: dict, is_expert: bool = False):
        """
        多模态版本：拼接图文 Few-shot + 当前图文 Task
        item 由 _get_item_standardized 生成，image_path 已是绝对路径。
        """
        # 1. 检索 K-Shot
        examples, dists, labels, r_indices = select_kshots(
            self.df, self.text_col, self.image_col, self.answer_col,
            self.k_shots, idx, self.indices, self.faiss_index,
            pool_embeddings=self.pool_embeddings_cache,
            lambda_param=self.lambda_param, options=self.options,
            metric="cosine",
            rules_dict=self.rules_dict
        )

        # 2. System Content
        system_content = self.prompt_text
        if hasattr(self, 'enhanced_rules') and self.enhanced_rules:
            system_content += f"\n\n### CRITICAL RULES FROM HUMAN EXPERTS:\n{self.enhanced_rules}"

        # 3. User Payload
        user_payload = []
        user_payload.append({"type": "text",
                              "text": "### REFERENCE EXAMPLES (Analyze both Image and Text):\n"})

        # --- Few-shot 示例 ---
        for i, ex in enumerate(examples):
            user_payload.append({
                "type": "text",
                "text": (f"Example {i+1}:\n"
                         f"Text: {ex['content']}\n"
                         f"Answer: {ex['answer']}\n"
                         f"Reasoning: {ex['explanation']}")
            })

            # 修复：ex['image_path'] 来自 select_kshots，是 DataFrame 里的相对路径
            # 必须通过 _abs_image_path 转换为绝对路径
            ex_img = self._abs_image_path(ex.get('image_path'))
            if ex_img:
                b64_img = get_base64_image(ex_img)
                user_payload.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_img}",
                                  "detail": "low"}
                })
            user_payload.append({"type": "text", "text": "---\n"})

        # 4. 当前 Query
        user_payload.append({
            "type": "text",
            "text": f"\n### CURRENT TASK:\nText: {item[self.text_col]}"
        })

        # item['image_path'] 由 _get_item_standardized 生成，已是绝对路径
        query_img = item.get('image_path')
        if query_img and os.path.exists(query_img):
            b64_query = get_base64_image(query_img)
            detail = "high" if is_expert else "low"
            user_payload.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_query}",
                              "detail": detail}
            })
        elif query_img:
            logger.warning(f"[Row {idx}] Query image not found: {query_img}")

        messages = [SystemMessage(content=system_content),
                    HumanMessage(content=user_payload)]
        return messages, dists, labels, r_indices

    def _display_item_for_hitl(self, item: dict):
        """Streamlit / Jupyter 环境下的图片展示"""
        print(f"Text Content: {item[self.text_col]}")
        img_path = item.get('image_path')
        if img_path and os.path.exists(img_path):
            try:
                from PIL import Image
                from IPython.display import display
                display(Image.open(img_path))
            except Exception:
                print(f"Image: {img_path}")
        else:
            print(f"⚠️ Image not found: {img_path}")
