# CAP and EuroParlVote: Dataset Exploration and Experiment Planning

## 1. Background

Previous PoliPrompt experiments mainly used text and multimodal datasets to evaluate classification performance and task-difficulty diagnosis. Our next political science paper will focus not only on classification accuracy, but also on whether PoliPrompt can help answer substantive political science questions. For example:

1. Can we reproduce findings from previous political science research?
2. Can PoliPrompt improve existing classification tasks or reveal their limitations?
3. Do classification errors reflect model limitations or ambiguous boundaries between political categories?
4. Can ECR help identify difficult cases that deserve further investigation?

This week's main task is to become familiar with the **CAP** and **EuroParlVote** datasets, consider how follow-up experiments could be designed, and discuss your ideas at the next meeting.

When thinking about the experiments, consider which fields should be used as PoliPrompt inputs, how the data should be reorganized, which conditions should be compared, how the prompts should be written, and how the results could be interpreted.

---

## 2. CAP Dataset

### Dataset Information

- Original dataset: [Comparative Agendas Project - U.S. Datasets](https://www.comparativeagendas.net/project/us/datasets)
- Select **Congressional Bills** on the download page
- Text field: `description`
- Label field: `majortopic`
- Number of policy-topic labels: 20
- Because the original dataset is large, our previous experiments used a balanced sample of 3,000 instances, with 150 instances per label
- The sampled data and previous 0-shot and 5-shot results are included in this directory

### Research Goal

Our previous experiments suggest that CAP classification errors are not evenly distributed across all labels. Instead, errors tend to concentrate between particular policy categories.

We would like to determine whether these errors occur because the model is distracted by competing options in the original 20-class task, or because some political categories have genuinely ambiguous boundaries.

One possible follow-up is to extract cases involving a particular pair of confused labels and reconstruct them as a binary classification task:

- If accuracy improves substantially and ECR decreases in the binary task, the original errors may mainly reflect **label competition** in the 20-class setting.
- If ECR remains high and the model continues to confuse the two labels, the boundary between those policy categories may itself be ambiguous.

### Questions for This Week

Please examine the CAP data and the existing 0-shot and 5-shot results. Be prepared to discuss:

1. Which labels are most frequently confused?
2. Which label pairs would you select for a binary classification experiment, and why?
3. Which instances should be selected from the existing data and error results?
4. How should the binary dataset be constructed so that it can be compared with the original 20-class results?
5. What information should be included in the binary-classification prompt?
6. Should both 0-shot and 5-shot settings be tested? Which models and parameters should remain fixed?
7. What results would support the label-competition explanation, and what results would support the category-boundary explanation?

---

## 3. EuroParlVote Dataset

### Dataset and Paper

- Paper: [Demographics and Democracy: Benchmarking LLMs' Gender Bias and Political Leaning in European Parliament](https://aclanthology.org/2025.icnlsp-1.41.pdf)
- Original dataset: [EuroParlVote on Hugging Face](https://huggingface.co/datasets/unimelb-nlp/EuroParlVote)

Please download the dataset and read both the paper and the dataset documentation.

EuroParlVote connects European Parliament speeches, political groups, and actual voting behavior. The original paper reports three main findings:

1. LLM gender classification shows performance disparities in the European Parliament context; for example, female MEPs are more likely to be misclassified as male.
2. Vote-prediction performance differs across political groups. Models perform better for some centrist or liberal groups and worse for some ideological extremes.
3. Adding political-group information improves prediction performance for some groups at the ideological extremes.

We first aim to understand and reproduce the paper's vote-prediction findings. We will then examine whether PoliPrompt improves classification performance or reveals cases in which political speeches and actual votes do not align.

### Main Experimental Comparison

The follow-up experiment will compare two conditions:

- **Speech-only:** topic + speech -> FOR/AGAINST
- **Speech+group:** topic + speech + political group -> FOR/AGAINST

The two conditions should use the same instances. Political-group information should be the only difference between them.

### Questions for This Week

After reading the paper and exploring the original data, consider:

1. What tasks, input information, and evaluation metrics were used in the original paper?
2. How should the original data be reorganized into Speech-only and Speech+group inputs for PoliPrompt?
3. How should the vote-prediction prompts be designed?
4. Should 0-shot and 5-shot settings both be tested? Which models and other parameters should be used?
5. How could the results be analyzed to determine whether the original paper's findings are reproduced?
