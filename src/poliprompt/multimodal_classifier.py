import logging
import numpy as np
from pathlib import Path
from typing import List
from langchain_core.messages import SystemMessage, HumanMessage

from .base_classifier import BaseClassifier
from . import utils
from .utils import get_base64_image

logger = logging.getLogger(__name__)


class MultiModalClassifier(BaseClassifier):
    """Classifier for datasets that combine text and images."""

    def _convert_df_to_docs(self, df):
        """Convert a DataFrame to a list of text + image path dicts for embedding."""
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
        Resolve any image path to an absolute path.

        - If already absolute and exists, return as-is.
        - Otherwise, try joining with image_dir.
        - Returns None if the file cannot be found.
        """
        if not raw_path or str(raw_path).strip() in ("", "None", "nan"):
            return None
        p = Path(str(raw_path))
        if p.is_absolute() and p.exists():
            return str(p)
        if self.image_dir:
            candidate = self.image_dir / p
            if candidate.exists():
                return str(candidate)
        if p.exists():
            return str(p)
        logger.warning(f"Image not found: {raw_path}")
        return None

    def _prepare_agent_messages(self, idx: int, item: dict, is_expert: bool = False):
        """Build the multimodal prompt messages (text + base64 images) for an inference call."""
        examples, dists, labels, r_indices = self._retrieve_kshots(idx)

        system_content = self.prompt_text
        if hasattr(self, 'enhanced_rules') and self.enhanced_rules:
            system_content += f"\n\n### CRITICAL RULES FROM HUMAN EXPERTS:\n{self.enhanced_rules}"

        user_payload = []
        user_payload.append({"type": "text",
                              "text": "### REFERENCE EXAMPLES (Analyze both Image and Text):\n"})

        for i, ex in enumerate(examples):
            user_payload.append({
                "type": "text",
                "text": (f"Example {i+1}:\n"
                         f"Text: {ex['content']}\n"
                         f"Answer: {ex['answer']}\n"
                         f"Reasoning: {ex['explanation']}")
            })

            # ex['image_path'] is a relative path from the DataFrame; resolve to absolute
            ex_img = self._abs_image_path(ex.get('image_path'))
            if ex_img:
                b64_img = get_base64_image(ex_img)
                user_payload.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_img}",
                                  "detail": "low"}
                })
            user_payload.append({"type": "text", "text": "---\n"})

        user_payload.append({
            "type": "text",
            "text": f"\n### CURRENT TASK:\nText: {item[self.text_col]}"
        })

        # item['image_path'] is already an absolute path set by _get_item_standardized
        query_img = item.get('image_path')
        if query_img and Path(query_img).exists():
            b64_query = get_base64_image(query_img)
            detail = "high" if is_expert else "low"
            # detail = "high"
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
        """Display the text and image for HITL review (inline in Jupyter, path in terminal)."""
        print(f"Text Content: {item[self.text_col]}")
        img_path = item.get('image_path')
        if img_path and Path(img_path).exists():
            try:
                from PIL import Image
                from IPython.display import display
                display(Image.open(img_path))
            except Exception:
                print(f"Image: {img_path}")
        else:
            print(f"Image not found: {img_path}")
