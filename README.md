
<p align="center">
  <img src="https://github.com/geshijoker/PoliPrompt/blob/main/poliprompt_logo.png" width="800" />
</p>

---

[![PyPI - Python](https://img.shields.io/badge/python-v3.10+-blue.svg)](https://pypi.org/project/PoliPrompt/)
[![Build](https://img.shields.io/github/actions/workflow/status/geshijoker/PoliPrompt/ci.yaml?branch=main)](https://github.com/geshijoker/PoliPrompt/actions)
[![docs](https://img.shields.io/badge/docs-Passing-green.svg)](https://poliprompt-tutorial.readthedocs.io/en/latest/)
[![PyPI - PyPi](https://img.shields.io/pypi/v/PoliPrompt)](https://pypi.org/project/poliprompt/)
[![PyPI - License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/geshijoker/PoliPrompt/blob/main/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2409.01466-<COLOR>.svg)](https://arxiv.org/pdf/2409.01466)


# 🚀 PoliPrompt: Multimodal Agentic Framework

PoliPrompt is a state-of-the-art computational social science tool for automated policy analysis. This project has evolved from a linear script into a **Stateful Agentic System**, integrating multimodal perception, high-performance retrieval, and Human-in-the-Loop (HITL) active learning.

## 🛠 Recent Progress (Milestones)

### 1. Multimodal Reasoning (Week 1)
- **Feature**: Joint Image-Text Processing.
- **Implementation**: Integrated **Qwen3-VL-Embedding** (1024-dim fused vectors) and **GPT-4o** vision reasoning.
- **Outcome**: Successfully identified nuanced visual cues (e.g., LGBTQ+ flags, political iconography) to improve classification reasoning.

### 2. High-Performance Vector Storage (Week 2)
- **Feature**: Transitioned to **Faiss (Facebook AI Similarity Search)**.
- **Implementation**: Replaced `.npy` with `IndexFlatIP` and L2 normalization.
- **Benefit**: Optimized retrieval speed and memory efficiency, enabling the system to scale to datasets with 10k+ instances.

### 3. Agentic HITL & Active Learning (Week 3 - Current)
- **Feature**: **LangGraph-driven** State Machine for Adaptive Annotation.
- **Architecture**:
  - **Persistence**: Implemented `MemorySaver` to provide "Save-and-Resume" capability.
  - **Interrupt Mechanism**: Automatic system pause (`interrupt_before`) at the `human_labeling` node when model confidence is low.
- **User Workflow**:
  - If `Confidence < 0.96`, system prints: `[PAUSED] Row X needs manual intervention.`
  - **To Resume**: User reruns the cell and provides input via the IDE prompt. The agent updates its state and proceeds seamlessly.

---

## 🔬 Research Findings & Analysis
During testing on the *Harmful Memes* dataset, a critical **Calibration Issue** was observed:
- **Over-confidence**: GPT-4o consistently assigns high confidence (e.g., 0.95) despite incorrect classifications.
- **Insight**: This underscores the necessity of HITL for high-stakes social science research where model self-evaluation is unreliable.

---

## 📅 Future Roadmap
- **Active Learning Loop**: Automate the injection of human-labeled samples back into the `exemplar_indices.json` pool.
- **Dynamic Thresholding**: Allow users to set custom confidence thresholds for different levels of supervision.
- **Disagreement Triggering**: Transition from confidence-based to multi-model disagreement-based intervention.


## Citation
To cite the **[PoliPrompt](https://arxiv.org/abs/2409.01466)** paper, please use the following BibTeX reference:

```bibtex
@article{liu2024poliprompt,
  title={PoliPrompt: A High-Performance Cost-Effective LLM-Based Text Classification Framework for Political Science},
  author={Liu, Menglin and Shi, Ge},
  journal={arXiv preprint arXiv:2409.01466},
  year={2024}
}
```


