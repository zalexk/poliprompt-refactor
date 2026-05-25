import os
import json
import logging
import sqlite3
import time
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm
from typing import Annotated, Any, Dict, List, Optional, TypedDict
import operator
from dotenv import load_dotenv
from . import utils
import numpy as np
from pydantic import BaseModel, Field
import faiss
from langgraph.graph import StateGraph, START, END
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from .llm_contribs import create_llm
from .selectors import create_selector
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress verbose HTTP request logs from the LLM clients
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class ClassificationResult(BaseModel):
    label: str = Field(description="Selected label from options")
    reason: str = Field(description="Detailed reasoning based on the input data.")


class AgentState(TypedDict):
    pending_indices: List[int]
    current_index: Optional[int]

    l1_res: Optional[dict]
    l2_res: Optional[dict]
    l3_res_list: List[dict]

    rag_distances: List[float]
    rag_labels: List[str]
    rag_indices: List[int]

    results: Annotated[Dict[str, dict], operator.ior]
    final_path: str
    temp_prediction: Optional[str]


class BaseClassifier(ABC):
    def __init__(self, config_path, prompt_path, env_path=None,
                 system_configs_dir: Optional[str | Path] = None, **kwargs):
        self.config_path = Path(config_path).expanduser().absolute()
        self.prompt_path = Path(prompt_path).expanduser().absolute()

        self.ui_model_configs = kwargs.get('ui_model_configs')

        if system_configs_dir:
            self.system_configs_dir = Path(system_configs_dir).absolute()
        else:
            self.system_configs_dir = Path(__file__).parent / "configs"

        for p in [self.config_path, self.prompt_path]:
            if not p.exists():
                raise FileNotFoundError(f"CRITICAL: Required file not found at {p}")

        if env_path is not None:
            load_dotenv(Path(env_path).expanduser().absolute(), override=True)
        else:
            load_dotenv(override=True)
        self.config = utils.load_yaml_config(self.config_path)
        self.prompt_text = self.prompt_path.read_text(encoding='utf-8')

        self._parse_config_to_self()
        self._setup_models()
        self._setup_rag_resources(mandatory=False)

        self.outfiles_dir.mkdir(parents=True, exist_ok=True)

        self._setup_observability()

        self.hitl_lock = threading.Lock()
        self.log_lock  = threading.Lock()

        # Streamlit HITL queues (None = terminal mode)
        self.hitl_request_queue  = None
        self.hitl_response_queue = None

        self._post_init()

    def _parse_config_to_self(self):
        user = self.config.get('user_settings', {})
        self.random_state = user.get('random_state', 42)
        self.lambda_param = user.get('lambda_param', 0.5)
        self.k_shots = user.get('k_shots', 5)

        self.options = [str(opt) for opt in user.get('options', [])]
        if not self.options:
            raise ValueError("'options' must be a non-empty list in the config.")
        if self.k_shots < 0:
            raise ValueError(f"'k_shots' must be a non-negative integer, got {self.k_shots}.")

        self.testing = user.get('testing', False)
        self.testing_size = user.get('testing_size', 128)

        proj = self.config.get('project', {})
        self.project_name = proj.get('name', 'PoliPrompt')
        self.modality = proj.get('modality', 'text')
        self.version = proj.get('version', 'v1')
        self.work_station = Path(proj.get('work_station', '.')).absolute()
        self.data_path = self.work_station / proj.get('data_path', 'data.csv')
        self.image_dir = self.work_station / proj['image_dir'] if proj.get('image_dir') else None
        self.outfiles_dir = self.work_station / proj.get('outfiles_dir', 'outfiles')

        cols = self.config.get('column_mapping', {})
        self.text_col = cols.get('text_col', 'text')
        img_col_raw = cols.get('image_col')
        self.image_col = None if str(img_col_raw).strip().lower() in ['none', ''] else img_col_raw
        self.answer_col = cols.get('answer_col', 'label')

        models = self.config.get('models', {})
        self.embedding_llm_name = models.get('embedding_llm', 'qwen-vl-max')
        self.primary_llm_name = models.get('primary_llm', 'gpt-4o-mini')
        self.secondary_llm_name = models.get('secondary_llm', 'qwen-vl-max')
        self.expert_llm_name = models.get('expert_llm', 'gpt-4o')

        # Load system-level JSON configs
        self.llm_configs = utils.load_json_config(self.system_configs_dir / 'llm_configs.json')
        self.embedding_llm_configs = utils.load_json_config(self.system_configs_dir / 'embedding_llm_configs.json')
        self.reduce_configs = utils.load_json_config(self.system_configs_dir / 'reduce_configs.json')
        self.select_configs = utils.load_json_config(self.system_configs_dir / 'select_configs.json')

        retrieval = self.config.get('retrieval', {})
        self.n_exemplars_pool = retrieval.get('n_exemplars_pool', 256)

        parallel = self.config.get('parallel', {})
        self.num_workers          = parallel.get('inference_workers',
                                      parallel.get('num_workers', 2))
        self.embedding_workers    = parallel.get('embedding_workers', 8)
        self.embedding_batch_size = parallel.get('embedding_batch_size', 5)

        obs = self.config.get('observability', {})
        self.observability_enabled  = obs.get('enabled', False)
        self.observability_provider = obs.get('provider', 'langfuse')

        self.indices = []
        self.rules_dict = {}
        self.faiss_index = None
        self.pool_embeddings_cache = None
        self.enhanced_rules = ""
        self.df = None

    @staticmethod
    def _model_name_from(data: Any) -> str:
        """Extract the model name string from a model config (str or dict)."""
        if isinstance(data, dict):
            return data.get('model') or data.get('model_name') or ''
        return str(data) if data else ''

    def _setup_models(self):
        """Instantiate the three LLM layers via the factory."""
        if self.ui_model_configs:
            # Config supplied at runtime (e.g. from a UI)
            source = self.ui_model_configs
            l1_data = source.get('l1')
            l2_data = source.get('l2')
            l3_data = source.get('l3')
        else:
            # Config read from config.yaml
            source = self.config.get('models', {})
            l1_data = source.get('primary_llm', 'gpt-4o-mini')
            l2_data = source.get('secondary_llm', 'qwen-vl-max')
            l3_data = source.get('expert_llm', 'gpt-4o')

        # Derive the config-lookup key from the actual model name so that
        # runtime UI overrides don't accidentally load the wrong defaults.
        l1_cfg = self._model_name_from(l1_data) or self.primary_llm_name
        l2_cfg = self._model_name_from(l2_data) or self.secondary_llm_name
        l3_cfg = self._model_name_from(l3_data) or self.expert_llm_name

        self.llm_a   = create_llm(l1_data, self.llm_configs.get(l1_cfg))
        self.llm_b   = create_llm(l2_data, self.llm_configs.get(l2_cfg))
        self.llm_adv = create_llm(l3_data, self.llm_configs.get(l3_cfg))

    def _setup_observability(self):
        """Initialize the Langfuse tracing handler if credentials are present."""
        pk = os.getenv("LANGFUSE_PUBLIC_KEY")
        sk = os.getenv("LANGFUSE_SECRET_KEY")

        if self.observability_enabled and self.observability_provider == "langfuse" and pk and sk:
            try:
                from langfuse.langchain import CallbackHandler
                # self.lf_handler = CallbackHandler(
                #     public_key=pk,
                #     secret_key=sk,
                #     host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
                # )
                self.lf_handler = CallbackHandler()
                logger.info("Langfuse initialized successfully.")
            except Exception as e:
                logger.warning(f"Langfuse init failed: {e}. Running without observability.")
                self.lf_handler = None
        else:
            self.lf_handler = None
            logger.info("Running without Langfuse observability.")

    def create_few_shot_pool(self):
        """Phase 1: Embed the dataset and select an elite exemplar pool."""
        embeddings_file = self.outfiles_dir / "embeddings.index"
        exemplars_file  = self.outfiles_dir / "exemplar_indices.json"
        self._ensure_data_loaded()

        if embeddings_file.exists() and exemplars_file.exists():
            print(f"--- Skipping Phase 1: existing pool files found. ---")
            self._setup_rag_resources(mandatory=True)
            return

        print("--- Phase 1: Building Few-Shot Exemplar Pool ---")
        start_time = time.time()
        self.df = utils.load_and_validate_data(self.data_path, self.text_col, self.answer_col, self.image_col)
        self.df[self.answer_col] = self.df[self.answer_col].astype(str)
        docs = self._convert_df_to_docs(self.df)

        from .llm_contribs import get_universal_embeddings
        emb_config = self.embedding_llm_configs.get(self.embedding_llm_name, {})
        real_model_name = emb_config.get("model", self.embedding_llm_name)

        embeddings = get_universal_embeddings(
            docs,
            model_name=real_model_name,
            max_workers=self.embedding_workers,
            batch_size=self.embedding_batch_size,
        )

        try:
            # Normalize embeddings so that inner product == cosine similarity
            faiss.normalize_L2(embeddings)
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings)
            faiss.write_index(index, str(embeddings_file))
            self.faiss_index = index
            logger.info(f"FAISS index saved to {embeddings_file}")
        except Exception as e:
            raise Exception(f"FAISS storage failed: {e}")

        actual_n = min(self.n_exemplars_pool, len(embeddings))
        selector = create_selector(method="kmeans")
        exemplar_indices = selector.select_exemplars(
            embeddings, n_exemplars=actual_n, random_state=self.random_state
        )
        exemplar_indices = [int(idx) for idx in exemplar_indices]

        try:
            exemplars_file.write_text(json.dumps(exemplar_indices, indent=4), encoding='utf-8')
            self.indices = exemplar_indices
            logger.info(f"Exemplar indices saved to {exemplars_file}")
        except Exception as e:
            raise Exception(f"Failed to save exemplar indices: {e}")

        elapsed = time.time() - start_time
        self._setup_rag_resources(mandatory=True)
        print(f"✅ Few-shot pool created in {elapsed:.2f} seconds.")

    def _ensure_data_loaded(self):
        """Load the dataset DataFrame if it has not been loaded yet."""
        if self.df is None:
            print(f"Loading dataset: {self.data_path.name}")
            self.df = utils.load_and_validate_data(
                self.data_path, self.text_col, self.answer_col, self.image_col
            )

    def optimize_task_description(self):
        """Phase 2: Extract per-exemplar reasoning (Map) and synthesize class-level rules (Reduce)."""
        self._setup_rag_resources(mandatory=True)
        self._ensure_data_loaded()

        rules_path    = self.outfiles_dir / "rules.json"
        enhanced_path = self.outfiles_dir / "enhanced_rules.txt"

        if rules_path.exists() and enhanced_path.exists():
            print(f"--- Phase 2: Existing rules found. Loading... ---")
            self.rules_dict    = json.loads(rules_path.read_text(encoding='utf-8'))
            self.enhanced_rules = enhanced_path.read_text(encoding='utf-8')
            return

        # ── MAP: extract reasoning for each exemplar in parallel ──
        if not rules_path.exists():
            extracted_rules_dict = {}
            print(f"--- MAP: Parallel logic extraction for {len(self.indices)} exemplars ---")

            def _extract_single_logic(idx):
                try:
                    item = self._get_item_standardized(idx)
                    ans  = self.df.loc[idx, self.answer_col]

                    msgs, _, _, _ = self._prepare_agent_messages(idx, item)
                    extra_text = (
                        f"\n\n[Ground Truth Answer]: {ans}\n"
                        f"Please explain the logic why this content is labeled as '{ans}'."
                    )

                    raw_msgs = self._format_messages_for_llm(msgs)

                    last_content = raw_msgs[-1]["content"]
                    if isinstance(last_content, list):
                        raw_msgs[-1]["content"].append({"type": "text", "text": extra_text})
                    else:
                        raw_msgs[-1]["content"] += extra_text

                    config = {"callbacks": [self.lf_handler]} if self.lf_handler else {}
                    response = self.llm_a.invoke(raw_msgs, config=config)

                    parsed = utils.parse_llm_response_generic(response.content, self.options)
                    return str(idx), parsed["reason"]
                except Exception as e:
                    logger.error(f"Error at index {idx}: {e}")
                    return str(idx), f"Extraction failed: {str(e)}"

            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_idx = {executor.submit(_extract_single_logic, idx): idx for idx in self.indices}
                with tqdm(total=len(self.indices), desc="RAFS Logic Mapping") as pbar:
                    for future in as_completed(future_to_idx):
                        idx_str, reason = future.result()
                        extracted_rules_dict[idx_str] = reason
                        pbar.update(1)

            rules_path.write_text(
                json.dumps(extracted_rules_dict, indent=4, ensure_ascii=False), encoding='utf-8'
            )
            self.rules_dict = extracted_rules_dict
        else:
            with open(rules_path, 'r', encoding='utf-8') as f:
                self.rules_dict = json.load(f)

        # ── REDUCE: synthesize class-level rules ──
        print("--- REDUCE: Synthesizing global knowledge by category ---")

        category_logic_map = {opt: [] for opt in self.options}
        for idx_str, reason in self.rules_dict.items():
            real_label = str(self.df.loc[int(idx_str), self.answer_col])
            if real_label in category_logic_map:
                category_logic_map[real_label].append(reason)

        final_rules_segments = []

        for label in self.options:
            reasons = category_logic_map.get(label, [])
            if not reasons:
                logger.warning(f"Category '{label}' has no elite exemplars in the pool.")
                continue

            num_to_take = min(len(reasons), 20)
            print(f"   Synthesizing rules for '{label}' using {num_to_take} samples...")

            logic_blob = "\n- ".join(reasons[:num_to_take])

            reduce_instruction = f"""
            TASK CONTEXT:
            {self.prompt_text}

            OBSERVED PATTERNS FOR CATEGORY '{label}':
            {logic_blob}

            [TASK]: Based on the task context and the {num_to_take} samples above, synthesize a concise summary of actionable rules for identifying category '{label}'. Ensure the rules are clear, specific, and clearly distinguishable from other options.

            SUMMARY RULES FOR '{label}':
            """

            config = {"callbacks": [self.lf_handler]} if self.lf_handler else {}
            try:
                response = self.llm_adv.invoke(reduce_instruction, config=config)
                final_rules_segments.append(response.content)
            except Exception as e:
                logger.error(f"Failed to synthesize category '{label}': {e}")

        self.enhanced_rules = "\n\n".join(final_rules_segments)
        enhanced_path.write_text(self.enhanced_rules, encoding='utf-8')

        logger.info("Map-Reduce optimization completed successfully.")
        return self.enhanced_rules

    def cache_embeddings(self, faiss_index, indices: List[int]):
        """Pre-fetch and cache embeddings for the exemplar pool from the FAISS index."""
        embeddings = np.array([faiss_index.reconstruct(i) for i in indices], dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        return embeddings

    def _format_messages_for_llm(self, msgs: List):
        """Convert LangChain message objects to raw dicts for direct API calls."""
        raw_msgs = []
        for m in msgs:
            role = "system" if m.type == "system" else "user"
            raw_msgs.append({"role": role, "content": m.content})
        return raw_msgs

    def _get_item_standardized(self, idx: int):
        """
        Extract a standardized sample dict from the DataFrame.

        Always includes the text column and an 'image_path' key
        (None for text-only tasks).
        """
        row = self.df.iloc[idx]
        item = {
            self.text_col: row[self.text_col],
            "text": row[self.text_col]
        }

        if self.modality == "multimodal" and self.image_col:
            img_val = row[self.image_col]
            if self.image_dir:
                item["image_path"] = str(self.image_dir / img_val)
            else:
                item["image_path"] = str(img_val)
        else:
            item["image_path"] = None

        return item

    # ── LangGraph nodes ──

    def _l1_node(self, state: AgentState):
        idx  = state["pending_indices"][0]
        item = self._get_item_standardized(idx)

        messages, dists, labels, r_indices = self._prepare_agent_messages(item=item, idx=idx)
        raw_messages = self._format_messages_for_llm(messages)
        config = {"callbacks": [self.lf_handler]} if self.lf_handler else {}
        try:
            response = self.llm_a.invoke(raw_messages, config=config)
            parsed   = utils.parse_llm_response_generic(response.content, options=self.options)
            label    = parsed["label"]
            reason   = parsed["reason"]
        except Exception as e:
            logger.error(f"Row {idx} L1 inference failed: {e}")
            label  = None
            reason = f"L1_SAFETY_BLOCKED_OR_ERROR: {str(e)}"

        if label not in self.options:
            logger.warning(f"Row {idx}: label '{label}' not in options {self.options}.")
            label = None

        return {
            "current_index": idx,
            "l1_res": {"label": label, "reason": reason},
            "temp_prediction": label,
            "rag_distances": dists,
            "rag_labels": labels,
            "rag_indices": r_indices
        }

    def _l2_node(self, state: AgentState):
        idx  = state["current_index"]
        item = self._get_item_standardized(idx)

        messages, _, _, _ = self._prepare_agent_messages(item=item, idx=idx)
        raw_messages = self._format_messages_for_llm(messages)
        config = {"callbacks": [self.lf_handler]} if self.lf_handler else {}
        try:
            response = self.llm_b.invoke(raw_messages, config=config)
            parsed   = utils.parse_llm_response_generic(response.content, options=self.options)
            label    = parsed["label"]
            reason   = parsed["reason"]
        except Exception as e:
            logger.error(f"Row {idx} L2 inference failed: {e}")
            label  = None
            reason = f"L2_SAFETY_BLOCKED_OR_ERROR: {str(e)}"

        if label not in self.options:
            logger.warning(f"Row {idx}: label '{label}' not in options {self.options}.")
            label = None

        return {
            "l2_res": {"label": label, "reason": reason},
            "temp_prediction": label,
        }

    def _expert_node(self, state: AgentState):
        idx  = state["current_index"]
        item = self._get_item_standardized(idx)

        messages, _, _, _ = self._prepare_agent_messages(item=item, idx=idx, is_expert=True)
        raw_messages = self._format_messages_for_llm(messages)
        config = {"callbacks": [self.lf_handler]} if self.lf_handler else {}

        l3_res_list = []
        try:
            for _ in range(2):
                response = self.llm_adv.invoke(raw_messages, config=config)
                parsed   = utils.parse_llm_response_generic(response.content, options=self.options)
                l3_res_list.append(parsed)

            label_0 = l3_res_list[0].get("label")
            label_1 = l3_res_list[1].get("label")
            final_l3_label = label_0 if (label_0 == label_1 and label_0 is not None) else None

        except Exception as e:
            logger.error(f"Row {idx} L3 inference failed: {e}")
            l3_res_list    = [{"label": None, "reason": f"L3_SAFETY_BLOCKED_OR_ERROR: {str(e)}"}] * 2
            final_l3_label = None

        return {
            "l3_res_list": l3_res_list,
            "temp_prediction": final_l3_label
        }

    def _human_node(self, state: AgentState):
        idx  = state["current_index"]
        item = self._get_item_standardized(idx)

        with self.hitl_lock:
            if self.hitl_request_queue is not None:
                # Streamlit mode: send to the UI queue and block until the response arrives
                print(f"\n--- [HITL] Row {idx}: waiting for Streamlit UI ---")
                self.hitl_request_queue.put({"idx": idx, "item": item, "options": self.options})
                user_input = self.hitl_response_queue.get(timeout=7200)
                print(f"[Row {idx}] Received UI label: {user_input}")
            else:
                # Terminal fallback
                print(f"\n--- [HUMAN INTERVENTION REQUIRED] Row {idx} ---")
                self._display_item_for_hitl(item)
                options_hint = "/".join(self.options)
                user_input   = input(f"Definitive Label ({options_hint}): ").strip()

            # Update the in-memory label for this row
            self.df.at[idx, self.answer_col] = str(user_input)

            # Inject the human-labeled sample into the exemplar pool
            if idx not in self.indices:
                self.indices.append(idx)

                new_vector = self.faiss_index.reconstruct(int(idx)).reshape(1, -1).astype('float32')
                if self.pool_embeddings_cache is None or self.pool_embeddings_cache.size == 0:
                    self.pool_embeddings_cache = new_vector
                else:
                    self.pool_embeddings_cache = np.vstack([self.pool_embeddings_cache, new_vector])

                with open(self.outfiles_dir / "exemplar_indices.json", "w", encoding='utf-8') as f:
                    json.dump(self.indices, f, indent=4)
                print(f"[Row {idx}] Added to exemplar pool.")

        return {
            "l3_res_list": state["l3_res_list"] + [{"label": user_input, "reason": "Manually labeled."}],
            "temp_prediction": user_input,
            "final_path": "HITL_Manual"
        }

    # ── Routing functions ──

    def _l2_router(self, state: AgentState):
        l1_label = state.get("l1_res", {}).get("label")
        l2_label = state.get("l2_res", {}).get("label")

        if l1_label is None or l2_label is None:
            print(f"[Row {state['current_index']}] Model failure detected. Escalating to Expert.")
            return "expert_node"

        if l1_label == l2_label:
            return "auto_update_node"

        return "expert_node"

    def _expert_router(self, state: AgentState):
        final_l3_label = state.get("temp_prediction")
        if final_l3_label is not None:
            return "auto_update_node"
        else:
            print(f"[Row {state['current_index']}] Expert disagreement. Escalating to HITL.")
            return "HITL_node"

    def _auto_update_node(self, state: AgentState):
        idx      = state["current_index"]
        final_res = {"label": "Safety_Blocked", "reason": "All agents failed or blocked."}

        if state.get("final_path") == "HITL_Manual":
            path_taken = "HITL_Manual"
            final_res  = state["l3_res_list"][-1]
        elif state.get("l3_res_list"):
            # L1/L2 disagreed; L3 resolved
            path_taken = "L3_Expert_Consensus"
            final_res  = state["l3_res_list"][-1]
        elif state.get("l2_res"):
            # L1 and L2 agreed
            path_taken = "L2_Heterogeneous_Match"
            final_res  = state["l2_res"]
        else:
            path_taken = "L1_Initial_Exit"
            final_res  = state["l1_res"]

        log_entry = {
            "row_id": idx,
            "final_label": final_res["label"] if final_res else None,
            "path": path_taken,
            "rag": {
                "indices":   state["rag_indices"],
                "labels":    state["rag_labels"],
                "distances": state["rag_distances"]
            },
            "reason": final_res.get("reason", "")
        }

        with self.log_lock:
            with open(self.outfiles_dir / "observability_logs.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        # Advance the pending queue
        remaining_pending = state["pending_indices"][1:]

        return {
            "results":          {str(idx): log_entry},
            "pending_indices":  remaining_pending,
            "current_index":    None,
            "l1_res":           None,
            "l2_res":           None,
            "l3_res_list":      [],
            "rag_distances":    [],
            "rag_labels":       [],
            "rag_indices":      [],
            "temp_prediction":  None,
            "final_path":       ""
        }

    def _build_graph(self, memory):
        builder = StateGraph(AgentState)

        builder.add_node("predict_l1",  self._l1_node)
        builder.add_node("predict_l2",  self._l2_node)
        builder.add_node("predict_l3",  self._expert_node)
        builder.add_node("auto_update", self._auto_update_node)
        builder.add_node("human_label", self._human_node)

        builder.add_edge(START, "predict_l1")
        builder.add_edge("predict_l1", "predict_l2")

        builder.add_conditional_edges("predict_l2", self._l2_router, {
            "auto_update_node": "auto_update",
            "expert_node":      "predict_l3"
        })

        builder.add_conditional_edges("predict_l3", self._expert_router, {
            "auto_update_node": "auto_update",
            "HITL_node":        "human_label"
        })

        builder.add_edge("human_label", "auto_update")

        def _check_next_step(state: AgentState):
            return "next" if state["pending_indices"] else "end"

        builder.add_conditional_edges("auto_update", _check_next_step, {
            "next": "predict_l1",
            "end":  END
        })

        return builder.compile(checkpointer=memory)

    def _setup_rag_resources(self, mandatory: bool = False):
        """
        Load FAISS index, exemplar indices, and rules from disk.

        mandatory=True: raises an error if files are missing (use before inference).
        mandatory=False: silently skips missing files (use during initialization).
        """
        idx_path   = self.outfiles_dir / "exemplar_indices.json"
        index_path = self.outfiles_dir / "embeddings.index"
        rules_path = self.outfiles_dir / "rules.json"

        if idx_path.exists():
            self.indices = json.loads(idx_path.read_text())
        elif mandatory:
            raise FileNotFoundError(
                f"Exemplar index missing: {idx_path}. Run create_few_shot_pool() first."
            )

        if index_path.exists():
            self.faiss_index = faiss.read_index(str(index_path))
            self.pool_embeddings_cache = self.cache_embeddings(self.faiss_index, self.indices)
        elif mandatory:
            raise FileNotFoundError(
                f"FAISS index missing: {index_path}. Run create_few_shot_pool() first."
            )

        if rules_path.exists():
            with open(rules_path, 'r', encoding='utf-8') as f:
                self.rules_dict = json.load(f)

    def annotate(self):
        """Phase 3: Classify all rows through the three-layer LangGraph pipeline."""
        from langgraph.checkpoint.sqlite import SqliteSaver

        self._setup_rag_resources(mandatory=True)
        self._ensure_data_loaded()
        if not self.enhanced_rules and (self.outfiles_dir / "enhanced_rules.txt").exists():
            self.enhanced_rules = (self.outfiles_dir / "enhanced_rules.txt").read_text(encoding='utf-8')
            
        self.df[self.answer_col] = self.df[self.answer_col].astype(str)

        db_path = self.outfiles_dir / "poliprompt_checkpoints.db"
        conn    = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        memory = SqliteSaver(conn)
        self.agent_app = self._build_graph(memory)

        all_indices    = self.df.index.tolist()
        target_indices = all_indices[:self.testing_size] if self.testing else all_indices

        # Skip rows that already have a completed result in the checkpoint database
        pending_indices = []
        for idx in target_indices:
            t_id = f"{self.project_name}_{self.version}_row_{idx}"
            snap = self.agent_app.get_state({"configurable": {"thread_id": t_id}})
            if not snap.values or not snap.values.get("results") or str(idx) not in snap.values["results"]:
                pending_indices.append(idx)

        if not pending_indices:
            print("✅ All tasks already finished!")
            return

        print(f"Processing {len(pending_indices)} remaining tasks...")

        self.backup_file = self.outfiles_dir / f"{self.project_name}_backup.csv"

        def sync_to_backup_file(indices):
            """Flush all completed results from the checkpoint DB to the backup CSV."""
            temp_df = self.df.loc[indices].copy()
            for i in indices:
                t_id = f"{self.project_name}_{self.version}_row_{i}"
                snap = self.agent_app.get_state({"configurable": {"thread_id": t_id}})
                if snap.values and "results" in snap.values:
                    res = snap.values["results"].get(str(i))
                    if res:
                        temp_df.at[i, "predicted_label"] = res["final_label"]
                        temp_df.at[i, "inference_path"]  = res["path"]
            temp_df.to_csv(self.backup_file, encoding='utf-8-sig')

        def _process_one(row_idx):
            t_id   = f"{self.project_name}_{self.version}_row_{row_idx}"
            config = {"configurable": {"thread_id": t_id}, "recursion_limit": 100}

            initial_state = {
                "pending_indices": [row_idx],
                "current_index":   None,
                "results":         {},
                "l1_res": None, "l2_res": None, "l3_res_list": [],
                "rag_distances": [], "rag_labels": [], "rag_indices": []
            }
            try:
                return self.agent_app.invoke(initial_state, config)
            except Exception as e:
                print(f"\nError at Row {row_idx}: {e}")
                return None

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(_process_one, idx) for idx in pending_indices]

            completed_count = 0
            for f in tqdm(as_completed(futures), total=len(pending_indices), desc="Annotating"):
                f.result()
                completed_count += 1
                # Flush to disk every 10 completions
                if completed_count % 10 == 0:
                    sync_to_backup_file(target_indices)

        sync_to_backup_file(target_indices)
        print(f"--- Finished. Backup saved at {self.backup_file} ---")

    def evaluate(self) -> dict:
        """
        Compute classification metrics by comparing predictions against ground-truth labels.

        Reads from the backup CSV produced by annotate(). Rows without a ground-truth
        label are excluded. Works with int, float, or string label types.
        """
        backup_path = self.outfiles_dir / f"{self.project_name}_backup.csv"
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup CSV not found: {backup_path}. Run annotate() first.")

        df    = pd.read_csv(backup_path, encoding='utf-8-sig')
        total = len(df)

        def norm(series):
            """Normalize a label series to stripped strings without trailing '.0'."""
            return (series.fillna("")
                          .astype(str)
                          .str.strip()
                          .str.replace(r"\.0$", "", regex=True))

        valid = {str(o).strip().replace(".0", "") for o in self.options}

        has_gt   = norm(df[self.answer_col]).isin(valid)
        has_pred = norm(df["predicted_label"]).isin(valid)

        n_no_gt      = int((~has_gt).sum())
        n_infer_fail = int((has_gt & ~has_pred).sum())
        n_evaluated  = int((has_gt & has_pred).sum())

        print(f"\n{'='*60}\nEvaluation Summary\n{'='*60}")
        print(f"  Total rows             : {total}")
        print(f"  No ground truth        : {n_no_gt}")
        print(f"  Inference failed       : {n_infer_fail}")
        print(f"  Evaluated              : {n_evaluated}")
        print(f"{'='*60}\n")

        if n_evaluated == 0:
            print("No valid samples to evaluate.")
            return {"total": total, "n_no_gt": n_no_gt,
                    "n_infer_fail": n_infer_fail, "n_evaluated": n_evaluated}

        eval_df = df[has_gt & has_pred].copy()
        y_true  = norm(eval_df[self.answer_col]).tolist()
        y_pred  = norm(eval_df["predicted_label"]).tolist()
        labels  = sorted(valid)

        cm    = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=[f"True_{l}"  for l in labels],
                                  columns=[f"Pred_{l}" for l in labels])
        print("Confusion Matrix:\n", cm_df.to_string(), "\n")

        report_str  = classification_report(y_true, y_pred, labels=labels, digits=4)
        report_dict = classification_report(y_true, y_pred, labels=labels,
                                            output_dict=True, zero_division=0)
        print("Classification Report:\n", report_str)

        accuracy  = accuracy_score(y_true, y_pred)
        per_class = {
            lbl: {k: (round(v, 4) if isinstance(v, float) else int(v))
                  for k, v in report_dict.get(lbl, {}).items()}
            for lbl in labels
        }
        averages = {
            avg: {k: (round(v, 4) if isinstance(v, float) else int(v))
                  for k, v in report_dict[avg].items()}
            for avg in ("macro avg", "weighted avg")
            if avg in report_dict
        }

        return {
            "total":            total,
            "n_no_gt":          n_no_gt,
            "n_infer_fail":     n_infer_fail,
            "n_evaluated":      n_evaluated,
            "accuracy":         round(accuracy, 4),
            "confusion_matrix": cm_df,
            "per_class":        per_class,
            "averages":         averages,
            "labels":           labels,
        }

    def _post_init(self):
        pass

    @abstractmethod
    def _prepare_agent_messages(self, **kwargs):
        pass

    @abstractmethod
    def _display_item_for_hitl(self, item: dict):
        pass

    @abstractmethod
    def _get_embeddings(self, data: List[dict]) -> np.ndarray:
        pass

    @abstractmethod
    def _convert_df_to_docs(self, df):
        pass
