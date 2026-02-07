import os
import sys
import time
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import List,Optional #类型提示

import json
import yaml  
#一种配置文件的格式，比json更易读，负责把.yaml格式文件读成python字典
import numpy as np
from tqdm import tqdm
import pandas as pd
import faiss

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate, ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
# 把文本一股脑塞进一个prompt里
from langchain.docstore.document import Document
# 包括文本和metadata
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain

from .utils import *
from .llm_contribs import create_llm, call_llm_wrapper, get_llm_embeddings, _prepare_multimodal_message
from .reducers import create_reducer
from .selectors import create_selector
from .retrieves import select_kshots


# Set the logging level to WARNING to ignore INFO and DEBUG logs of LLM requests
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)
# 除非有warning，否则不需要打印

# Setup logger
logger = logging.getLogger(__name__)
# 给当前文件创建一个叫logger的记录员

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

        # try:
        #     data = pd.read_csv(self.data_file)
        # except Exception as e:
        #     raise ValueError(f"Failed to read the csv file '{self.data_file}': {e}")
        # suffix = Path(self.data_file).suffix.lower()
        # try:
        #     if suffix == '.csv':
        #         data = pd.read_csv(self.data_file)
        #     elif suffix == '.json':
        #         data = pd.read_json(self.data_file)
        #     elif suffix == '.jsonl':
        #         data = pd.read_json(self.data_file, lines=True)
        #     else:
        #         raise ValueError(f"Unsupported file format: {suffix}")
            
        #     logger.info(f"Successfully loaded data from {self.data_file}({suffix})")
        # except Exception as e:
        #     raise ValueError(f"Error reading file '{self.data_file}':{e}")
        #     # 这里'{self.data_file}'加上单引号是为了防止路径当中有空格会看起来很乱
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

        # Construct large language chat model
        model_config = load_config(llm_configs, llm_name)
        model = create_llm(llm_name=llm_name, model_config=model_config)

        # Load prompt and indices
        prompt = get_prompt(prompt_file=prompt_file)
        exemplars_file = self.outfiles_dir / "exemplar_indices.json"
        indices = json.loads(exemplars_file.read_text())

        # Read docs dataset
        df = load_and_validate_data(self.data_file,self.feature_col,self.answer_col,self.image_col) # 读取文档
        contexts = read_multimodal_docs_from_dataframe(df, self.feature_col,self.image_col,self.image_dir) # 输出为字典列表格式
        answers = df[self.answer_col].tolist()
        contexts = [contexts[i] for i in indices]
        answers = [answers[i] for i in indices]

        print(contexts)

        # Generate reasons for human's annotation of each text and save it in the rules_path
        if not os.path.isfile(rules_path):  #如果rules.json里已经有内容就直接用，不再生成了

            # Ask LLM to justify each answer to a question
            qa_sys_prompt = """You are a helpful AI assistant specialized in multimodal analysis. \
                You will be provided with a task description and an example consisting of a (Text, Image, and Answer) triplet. \
                The Text and Image represent the context of the task, and the Answer is a human's choice based on that context. \
                Your job is to analyze both the visual information in the Image and the semantic meaning in the Text to understand the underlying logic of the human's choice. \
                Briefly justify why the choice is appropriate by connecting the visual cues with the textual context. \
                Summarize a rule that explains why this option is better compared to other options. \
                The rule must start with \"The correct option is [HUMAN ANSWER].\"\n\nTask:\"\"\"\n{task}\"\"\"\n
                """
            qa_sys_prompt = qa_sys_prompt.format(task=prompt)

            results = []
            call_llm_with_limits = call_llm_wrapper(self.requests_per_period, self.secondf_per_period)
            for i, (context, answer) in tqdm(enumerate(zip(contexts, answers)), total=len(indices), leave=False):
                msg_text = f"Text: {context['text']}\nHuman Answer: {answer}"
                messages = _prepare_multimodal_message(qa_sys_prompt, msg_text, context['image_path'])
                response = model.invoke(messages)
                results.append(response.content)
            # Save the extracted rules
            rules_path.write_text(json.dumps(results, indent=4))
        else:
            results = json.loads(rules_path.read_text())

        # Concatenate all rules as a document
        docs = []
        for i, result in tqdm(enumerate(results), total=len(results), disable=disable_progress_bar):
            doc = Document(page_content=result, metadata={"idx": str(i)})
            docs.append(doc)

        # Ask LLM to summarize the rules with MapReduce
        sum_sys_prompt = """You are give a task description. \nTask:\"\"\"\n{task}\"\"\"\nThe task provides a few options to answer a question. \
        The following texts are the answer traces with rules extracted to summarize why an option is correct to the question.\
        Please briefly summarize the rules for valid existing options. \
        The content can be, for example, the meaning of the options, or the reason why we prefer one option than the other in different situations. \
        This summarization will be used to guide human to answer such questions. \
        Please make the summarizaiton concise and actionable."""
        sum_sys_prompt = sum_sys_prompt.format(task=prompt)
        map_template = """\n\"\"\"{docs}\"\"\"\nCONCISE RULE:"""

        map_prompt = PromptTemplate.from_template(sum_sys_prompt + map_template)
        map_chain = LLMChain(llm=model, prompt=map_prompt) 

        reduce_template = """The following is set of summary of rules to justify a few options:
        {docs}
        Take these and distill it into a final, consolidated summary of the main rules for valid options. 
        CONCISE SUMMARY RULES:"""
        reduce_prompt = PromptTemplate.from_template(sum_sys_prompt + reduce_template)
        reduce_chain = LLMChain(llm=model, prompt=reduce_prompt)

        # Take a list of documents, combine them into a single string, and pass it to an LLMChain
        combine_documents_chain = StuffDocumentsChain(llm_chain=reduce_chain, document_variable_name="docs")

        # Combine and iteratively reduce the mapped documents
        reduce_documents_chain = ReduceDocumentsChain(
            # This is final chain that is called.
            combine_documents_chain=combine_documents_chain,
            # If documents exceed context for `StuffDocumentsChain`
            collapse_documents_chain=combine_documents_chain,
            # The maximum number of tokens to group documents into.
            token_max=4000,
        )

        # Combine documents by mapping a chain over them, then combine results
        map_reduce_chain = MapReduceDocumentsChain(
            # Map chain
            llm_chain=map_chain,
            # Reduce chain
            reduce_documents_chain=reduce_documents_chain,
            # The variable name in the llm_chain to put the documents in
            document_variable_name="docs",
            # Return the results of the map steps in the output
            return_intermediate_steps=False,
        )

        output_summary = map_reduce_chain.invoke(docs)
        logger.warning(output_summary["output_text"])

        return output_summary["output_text"]

    def annotate(
            self, 
            llm_name: str,
            prompt_file_name: str,
            kshots: int=0,
            lambda_param: float=1.0,
            disable_progress_bar: bool=False,
            testing: bool=False,
            testing_size: int=None,
        ):
        # Input files
        dataset_name = self.data_file.stem
        prompt_file = self.prompts_dir / prompt_file_name
        prompt_name = prompt_file.stem
        llm_configs = self.configs_dir / "llm_configs.json"
        index_file = str(self.outfiles_dir / "embeddings.index")
        # embeddings = np.load(embeddings_file)
        index = faiss.read_index(index_file)
        exemplars_file = self.outfiles_dir / "exemplar_indices.json"
        indices = json.loads(exemplars_file.read_text())
        if len(indices) < kshots:
            logger.info(f"The number of exemplars in the pool ({len(indices)}) is smaller than the user specified {kshots}-shots, "
                    f"we will only use the first {len(indices)}-shots.")
            kshots = len(indices)

        # Create the file to save responses
        if testing:
            output_file_name = f"{dataset_name}_{llm_name}_{prompt_name}_{kshots}shots_test.csv"
        else:
            output_file_name = f"{dataset_name}_{llm_name}_{prompt_name}_{kshots}shots.csv"
        output_file = self.outfiles_dir / output_file_name

        if os.path.isfile(output_file):
            results = read_docs_from_dataframe(pd.read_csv(output_file, index_col=False), self.answer_col)
            logger.info(f"Saving output to an existing file \n{output_file}")
        else:
            results = []
            logger.info(f"Saving output to a new file \n{output_file}")
        processed_count = max(len(results), 0)

        # Construct large language chat model
        model_config = load_config(llm_configs, llm_name)
        model = create_llm(llm_name=llm_name, model_config=model_config)

        # # Prepare the unlabelled data to annotate
        # df = pd.read_csv(self.data_file, index_col=False)
        # df = df[[self.feature_col, self.answer_col]]
        # print("Examples of data\n", df.head(n=min(5, len(df))))
        # if testing:
        #     if testing_size is None:
        #         raise ValueError("testing_size cannot be None in the testing mode. It must be a valid integer.")
        #     if len(df) < testing_size:
        #         logger.info(f"Provided testing_size ({testing_size}) is greater than the number of questions ({len(questions)}), "
        #             f"we will only use the first {len(df)} questions.")
        #         testing_size = len(df)

        #     df = df[:testing_size]
        #     indices = [idx for idx in indices if idx < testing_size]
        
        # Prepare the unlabelled data to annotate
        df = load_and_validate_data(self.data_file, self.feature_col,self.answer_col,self.image_col)
        multimodal_data = read_multimodal_docs_from_dataframe(df,self.feature_col,self.image_col,self.image_dir)
        print("Examples of data\n", df.head(n=min(5, len(df))))
        if testing:
            if testing_size is None:
                raise ValueError("testing_size cannot be None in the testing mode. It must be a valid integer.")
            if len(df) < testing_size:
                logger.info(f"Provided testing_size ({testing_size}) is greater than the number of questions ({len(questions)}), "
                    f"we will only use the first {len(df)} questions.")
                testing_size = len(df)
            df = df[:testing_size]
            multimodal_data=multimodal_data[:testing_size]
            indices = [idx for idx in indices if idx < testing_size]
            # 如果testing_size是50，那么df也被切成50，如果few-shot的例子索引超过50就会被删去





        # Create the prefix and suffix from the prompt 
        prefix = get_prompt(prompt_file=prompt_file)
        suffix_examples = ""
        if kshots > 0:
            prefix = prefix + f" You're given {kshots} examples for references. " 
            suffix_examples = f"That's all {kshots} examples. "
        options_str = ", ".join(self.options)
        suffix = suffix_examples + f"""Perform the task based on the provided **IMAGE** and the **TEXT** below. 
                    Choose the correct answer from the options ({options_str}) in a single-choice format with options in '<' and '>' and then provide analysis process or reason.
                    Text: \"\"\"
                    {{input}}
                    \"\"\"
                    Answer: """
        # suffix = suffix + "Perform the task based on the next given text, choose the correct answer from the options (" + ", ".join(self.options) + ') in a single-choice format with options in \'<\' and \'>\'.\nText: """\n{input}\n"""\nAnswer: '
        call_llm_with_limits = call_llm_wrapper(self.requests_per_period, self.secondf_per_period)

        start_time = time.time()
        # Call LLM to annotate unlabelled instances
        try:
            progress_bar = tqdm(total=len(df), initial=processed_count, disable=disable_progress_bar, leave=False)
            
            for i, item in enumerate(multimodal_data):
                if i < processed_count:
                    continue
                
                if i not in indices:
                    examples = select_kshots(
                        df, self.feature_col, self.answer_col, kshots, i, indices, index, lambda_param, self.options
                    )
                    example_prompt = PromptTemplate(
                        input_variables=["content", "answer"], template='Text: """\n{content}\n"""\nAnswer: {answer}'
                    )

# 2. 生成 Few-shot 文本提示词
                    example_prompt = PromptTemplate(
                        input_variables=["content", "answer"], 
                        template='Text: """\n{content}\n"""\nAnswer: {answer}'
                    )
                    few_shot_template = FewShotPromptTemplate(
                        examples=examples,
                        example_prompt=example_prompt,
                        prefix=prefix,
                        suffix=suffix,
                        input_variables=["input"],
                    )
                    
                    # 3. 渲染出完整的文本指令 (含例子 + 当前文本)
                    # few_shot_template.format 会把 item['text'] 填入 suffix 里的 {input}
                    full_text_instruction = few_shot_template.format(input=item['text'])

                    # 4. 重点：组装多模态消息体 (Text + Image)
                    # 调用你写在 llm_contribs 里的那个翻译官函数
                    messages = _prepare_multimodal_message(
                        system_prompt="You are a professional policy image and text classifier.",
                        text_content=full_text_instruction,
                        image_path=item['image_path']
                    )

                    # 5. 调用模型（注意这里直接用 model.invoke，因为 LangChain 的 Chain 
                    # 默认处理字符串，处理多模态 Message List 建议直接 invoke）
                    response = call_llm_with_limits(model.invoke, messages)
                    result = response.content # 获取模型返回的文本内容
                    
                else:
                    # 如果这一行本身就是例子，直接从 df 里拿正确答案
                    result = df.loc[i, self.answer_col]
                
                results.append(result)
                progress_bar.update(1)
                
            progress_bar.close()
            
        except Exception as e:  # Naked execpt, blame the your LLM API
            logger.error(f"Error when calling LLM API to annotate texts: {e}")
            
        finally:
            # Track and print the computation time
            end_time = time.time()
            elapsed = end_time - start_time
            hours, minutes, secondf = track_computation_time(elapsed)
            logger.info(f"Computation time: {int(hours):02}:{int(minutes):02}:{int(secondf):02}")
            if results:
                results_df = pd.DataFrame()
                results_df[self.answer_col] = results
                results_df.to_csv(output_file, index=False)
                logger.info(f"Results saved to {output_file}")
            else:
                raise Exception("Empty results, please check the log.")

    def cot_mismatch_solver(
            self,
            llm_name: str,
            mismatch_file_name: str,
            cot_prompt_file_name: str,
            kshots: int=0,
            lambda_param: float=1.0,
            disable_progress_bar: bool=False,
            testing: bool=False,
            testing_size: int=None,
        ):
        # Input files
        dataset_name = self.data_file.stem
        cot_prompt_file = self.prompts_dir / cot_prompt_file_name
        prompt_name = cot_prompt_file.stem
        llm_configs = self.configs_dir / "llm_configs.json"
        embeddings_file = self.outfiles_dir / "embeddings.index"
        embeddings = np.load(embeddings_file)
        exemplars_file = self.outfiles_dir / "exemplar_indices.json"
        indices = json.loads(exemplars_file.read_text())
        if len(indices) < kshots:
            logger.info(f"The number of exemplars in the pool ({len(indices)}) is smaller than the user specified {kshots}-shots, "
                    f"we will only use the first {len(indices)}-shots.")
            kshots = len(indices)
        mismatch_file = self.infiles_dir / mismatch_file_name
        mismatch_name = mismatch_file.stem
        mismatch_indices = json.loads(mismatch_file.read_text())

        # Create the file to save responses
        if testing:
            output_file_name = f"cot_{mismatch_name}_{dataset_name}_{llm_name}_{prompt_name}_{kshots}shots_test.csv"
        else:
            output_file_name = f"cot_{mismatch_name}_{dataset_name}_{llm_name}_{prompt_name}_{kshots}shots.csv"
        output_file = self.outfiles_dir / output_file_name

        # Construct large language chat model
        model_config = load_config(llm_configs, llm_name)
        model = create_llm(llm_name=llm_name, model_config=model_config)

        # Prepare the unlabelled data to annotate
        df = pd.read_csv(self.data_file, index_col=False)
        df = df[[self.feature_col, self.answer_col]]
        if testing:
            if testing_size is None:
                raise ValueError("testing_size cannot be None in the testing mode. It must be a valid integer.")
            if len(df) < testing_size:
                logger.info(f"Provided testing_size ({testing_size}) is greater than the number of questions ({len(questions)}), "
                    f"we will only use the first {len(df)} questions.")
                testing_size = len(df)

            df = df[:testing_size]
            indices = [idx for idx in indices if idx < testing_size]
            mismatch_indices = [idx for idx in mismatch_indices if idx < testing_size]

        # Create the prefix and suffix from the prompt 
        prefix = get_prompt(prompt_file=cot_prompt_file)
        suffix = "Please "
        if kshots > 0:
            prefix = prefix + f" You're given {kshots} examples for references. " 
            suffix = f"That's all {kshots} examples with succinct answers. However, For the next text, you need to first analyze it step by step and provide reasoning. Then "
        suffix = suffix + "perform the task by analyzing the following text step by step and provide your reasoning. Do not exceed 50 words. Finally, based on the reasoning, choose the correct answer from the options (" + ", ".join(self.options) + ') in a single-choice format with options in \'<\' and \'>\'.\nText: """\n{input}\n"""\nAnswer: Let\'s think step by step. '
        call_llm_with_limits = call_llm_wrapper(self.requests_per_period, self.secondf_per_period)

        start_time = time.time()
        # Call LLM to annotate unlabelled instances
        results = []
        processed_ids = []
        try:            
            for i in tqdm(mismatch_indices, total=len(mismatch_indices), disable=disable_progress_bar, leave=False, desc="Processing indices"):
                if i not in indices:
                    examples = select_kshots(
                        df, self.feature_col, self.answer_col, kshots, i, indices, embeddings, lambda_param, self.options
                    )
                    example_prompt = PromptTemplate(
                        input_variables=["content", "answer"], template='Text: """\n{content}\n"""\nAnswer: {answer}'
                    )

                    cot_prompt = FewShotPromptTemplate(
                        examples=examples,
                        example_prompt=example_prompt,
                        prefix=prefix,
                        suffix=suffix,
                        input_variables=["input"],
                    )
                    row = df.loc[i, self.feature_col]
                    question = {"input": row}

                    chain = cot_prompt | model | StrOutputParser()
                    result = call_llm_with_limits(chain, question)
                else:
                    result = df.loc[i, self.answer_col]
                results.append(result)
                processed_ids.append(i)

        except Exception as e:  # Naked execpt, blame the your LLM API
            logger.error(f"Error when calling LLM API to annotate texts: {e}")

        finally:
            # Track and print the computation time
            end_time = time.time()
            elapsed = end_time - start_time
            hours, minutes, secondf = track_computation_time(elapsed)
            logger.info(f"Computation time: {int(hours):02}:{int(minutes):02}:{int(secondf):02}")
            if results:
                results_df = pd.DataFrame()
                results_df["mismatch ids"] = processed_ids
                results_df[self.answer_col] = results
                results_df.to_csv(output_file, index=False)
                logger.info(f"Results saved to {output_file}")
            else:
                raise Exception("Empty results, please check the log.")

    def judge_mismatch_solver(
            self, 
            judge_llm_name: str,
            llm1_name: str,
            llm2_name: str,
            mismatch_file_name: str,
            cot_prompt_file_name: str,
            judge_prompt_file_name: str,
            kshots: int=0,
            lambda_param: float=1.0,
            disable_progress_bar: bool=False,
            testing: bool=False,
            testing_size: int=None,
        ):
        # Input files
        dataset_name = self.data_file.stem
        cot_prompt_file = self.prompts_dir / cot_prompt_file_name
        judge_prompt_file = self.prompts_dir / judge_prompt_file_name
        prompt_name = judge_prompt_file.stem
        llm_configs = self.configs_dir / "llm_configs.json"
        embeddings_file = self.outfiles_dir / "embeddings.index"
        embeddings = np.load(embeddings_file)
        exemplars_file = self.outfiles_dir / "exemplar_indices.json"
        indices = json.loads(exemplars_file.read_text())
        if len(indices) < kshots:
            logger.info(f"The number of exemplars in the pool ({len(indices)}) is smaller than the user specified {kshots}-shots, "
                    f"we will only use the first {len(indices)}-shots.")
            kshots = len(indices)
        mismatch_file = self.infiles_dir / mismatch_file_name
        mismatch_name = mismatch_file.stem
        mismatch_indices = json.loads(mismatch_file.read_text())

        # Create the file to save responses
        if testing:
            output_file_name = f"judge_{mismatch_name}_{dataset_name}_{judge_llm_name}_{prompt_name}_{kshots}shots_test.csv"
        else:
            output_file_name = f"judge_{mismatch_name}_{dataset_name}_{judge_llm_name}_{prompt_name}_{kshots}shots.csv"
        output_file = self.outfiles_dir / output_file_name

        # Construct large language chat model
        llm1_config = load_config(llm_configs, llm1_name)
        chat_llm1 = create_llm(llm_name=llm1_name, model_config=llm1_config)
        llm2_config = load_config(llm_configs, llm2_name)
        chat_llm2 = create_llm(llm_name=llm2_name, model_config=llm2_config)
        judge_llm_config = load_config(llm_configs, judge_llm_name)
        chat_llm_judge = create_llm(llm_name=judge_llm_name, model_config=judge_llm_config)

        # Prepare the unlabelled data to annotate
        df = pd.read_csv(self.data_file, index_col=False)
        df = df[[self.feature_col, self.answer_col]]
        if testing:
            if testing_size is None:
                raise ValueError("testing_size cannot be None in the testing mode. It must be a valid integer.")
            if len(df) < testing_size:
                logger.info(f"Provided testing_size ({testing_size}) is greater than the number of questions ({len(questions)}), "
                    f"we will only use the first {len(df)} questions.")
                testing_size = len(df)

            df = df[:testing_size]
            indices = [idx for idx in indices if idx < testing_size]
            mismatch_indices = [idx for idx in mismatch_indices if idx < testing_size]

        # Create the prefix and suffix from the prompt 
        cot_prefix = get_prompt(prompt_file=cot_prompt_file)
        cot_suffix = "Please "
        if kshots > 0:
            cot_prefix = cot_prefix + f" You're given {kshots} examples for references. " 
            cot_suffix = f"That's all {kshots} examples with succinct answers. However, For the next text, you need to first analyze it step by step and provide reasoning. Then "
        cot_suffix = cot_suffix + "perform the task by analyzing the following text step by step and provide your reasoning. Do not exceed 50 words. Finally, based on the reasoning, choose the correct answer from the options (" + ", ".join(self.options) + ') in a single-choice format with options in \'<\' and \'>\'.\nText: """\n{input}\n"""\nAnswer: Let\'s think step by step. '
        
        judge_prefix = get_prompt(prompt_file=judge_prompt_file)
        judge_suffix = ""
        if kshots > 0:
            judge_prefix = judge_prefix + f" You're given {kshots} examples for references. " 
            judge_suffix = f"That's all {kshots} examples with succinct answers. "
        judge_suffix = judge_suffix + "Please judge which response to the following text is correct and provide reasoning within 150 words. Finally, answer the question by copy the correct reponse's choice from (" + ", ".join(self.options) + ') in \'<\' and \'>\'.\nText: """\n{input}\n"""\nAnswer: '

        call_llm_with_limits = call_llm_wrapper(self.requests_per_period, self.secondf_per_period)

        start_time = time.time()
        # Call LLM to annotate unlabelled instances
        results = []
        processed_ids = []
        try:
            for i in tqdm(mismatch_indices, total=len(mismatch_indices), disable=disable_progress_bar, leave=False, desc="Processing indices"):
                if i not in indices:
                    examples = select_kshots(
                        df, self.feature_col, self.answer_col, kshots, i, indices, embeddings, lambda_param, self.options
                    )
                    example_prompt = PromptTemplate(
                        input_variables=["content", "answer"], template='Text: """\n{content}\n"""\nAnswer: {answer}'
                    )

                    cot_prompt = FewShotPromptTemplate(
                        examples=examples,
                        example_prompt=example_prompt,
                        prefix=cot_prefix,
                        suffix=cot_suffix,
                        input_variables=["input"],
                    )
                    judge_prompt = FewShotPromptTemplate(
                        examples=examples,
                        example_prompt=example_prompt,
                        prefix=judge_prefix,
                        suffix=judge_suffix,
                        input_variables=["input", "response1", "response2"],
                    )
                    row = df.loc[i, self.feature_col]
                    question = {"input": row}

                    # Ask both LLMs
                    chain1 = cot_prompt | chat_llm1 | StrOutputParser()
                    response1 = call_llm_with_limits(chain1, question)
                    chain2 = cot_prompt | chat_llm2 | StrOutputParser()
                    response2 = call_llm_with_limits(chain2, question)

                    # Format comparison prompts and ask the judging LLM
                    judge_question = {"input": row, "response1": response1, "response2": response2}
                    chain_judge = judge_prompt | chat_llm_judge | StrOutputParser()
                    judge_response = call_llm_with_limits(chain_judge, judge_question)
                else:
                    result = df.loc[i, self.answer_col]
                results.append(judge_response)
                processed_ids.append(i)

        except Exception as e:  # Naked execpt, blame the your LLM API
            logger.error(f"Error when calling LLM API to annotate texts: {e}")
        finally:
            # Track and print the computation time
            end_time = time.time()
            elapsed = end_time - start_time
            hours, minutes, secondf = track_computation_time(elapsed)
            logger.info(f"Computation time: {int(hours):02}:{int(minutes):02}:{int(secondf):02}")
            if results:
                results_df = pd.DataFrame()
                results_df["mismatch ids"] = processed_ids
                results_df[self.answer_col] = results
                results_df.to_csv(output_file, index=False)
                logger.info(f"Results saved to {output_file}")
            else:
                raise Exception("Empty results, please check the log.")
