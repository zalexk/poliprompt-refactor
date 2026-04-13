import logging
import numpy as np
from typing import List
from langchain_core.messages import SystemMessage, HumanMessage

from .base_classifier import BaseClassifier
from .retrieves import select_kshots
from . import utils

logger = logging.getLogger(__name__)


class TextClassifier(BaseClassifier):
    """Classifier for text-only datasets."""

    def _convert_df_to_docs(self, df):
        """Convert a DataFrame to a list of text dicts for embedding."""
        docs = []
        for _, row in df.iterrows():
            docs.append({"text": str(row[self.text_col])})
        return docs

    def _get_embeddings(self, docs_dicts: List[dict]) -> np.ndarray:
        from .llm_contribs import get_universal_embeddings
        return get_universal_embeddings(
            docs_dicts,
            self.embedding_llm_name,
            max_workers=self.embedding_workers,
            batch_size=self.embedding_batch_size,
        )

    def _prepare_agent_messages(self, idx: int, item: dict, is_expert: bool = False):
        """Build the prompt messages for a text-only inference call."""
        examples, dists, labels, r_indices = select_kshots(
            self.df, self.text_col, None, self.answer_col,
            self.k_shots, idx, self.indices, self.faiss_index,
            pool_embeddings=self.pool_embeddings_cache,
            lambda_param=self.lambda_param, options=self.options,
            metric="cosine",
            rules_dict=self.rules_dict
        )

        sys_content = self.prompt_text
        if self.enhanced_rules:
            sys_content += f"\n\n### GLOBAL REASONING RULES:\n{self.enhanced_rules}"

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
        """Print the text content for terminal-mode HITL review."""
        print("-" * 50)
        print(f"TEXT CONTENT:\n{item[self.text_col]}")
        print("-" * 50)
