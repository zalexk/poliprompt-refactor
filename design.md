###
Project Goal (Objective)
Upgrade the existing PoliPrompt text classification framework to a multimodal analysis system. Enable simultaneous processing of text and corresponding images to improve annotation accuracy in complex policy scenarios (e.g., hate speech detection, graphic propaganda) and other fields.

User Interface Changes
To support multimodal, users need to provide two additional parameters when initializing TextClassifier:
•	image_col: Specifies the column storing relative image paths (e.g., 'img/001.png').
•	data_root: Specifies the local directory for images to construct full paths.

Core Integration Logic
The strategy is to minimize changes to the original process, activating "text + image" mode only when images are provided.
1. Data Loading Optimization:
Supports .csv and .jsonl formats. Code automatically detects the suffix and loads metadata accordingly.
2. Message Construction Upgrade (Key Point):
•	Shift from "pure string" prompt mode.
•	Introduce "text-image parallel list" structure: Package prompt text and image Base64 encoding into a list for the model.
•	Logic Judgment (If-Else): Activate "text-image merge mode" if image parameters are provided; otherwise, retain the original pure text flow to ensure old projects remain intact.
3. Delayed Encoding Strategy (Just-in-Time Encoding):
Avoid converting all images at initialization; encode Base64 only when annotate processes a specific data entry to save memory.

Implementation Roadmap
Step 1: Basic Infrastructure Upgrade
•	Add image-to-Base64 utility function in utils.py.
•	Enable OpenAI model to support proxy addresses (Base URL) in llm_contribs.py.
Step 2: Inference Layer Integration
•	Modify TextClassifier.annotate method. Add logic judgment: If an image column exists, append image data after the prompt for dual-dimensional text-image inference.
Step 3: Global Extension (Future Plan)
•	Extend the same logic to mismatch_solver (error correction) and judge (adjudication) stages.
•	Research CLIP model to introduce image features in the Embedding (vectorization) stage for visual similarity-based sample retrieval.

Current Progress (Current Status)
1. Prototype Validation: Completed text-image analysis test on 8 samples in demo_multimodal.ipynb; model accurately identifies visual features in images.
2.  Code Integration: Ongoing non-destructive modifications to src/ source code.


###