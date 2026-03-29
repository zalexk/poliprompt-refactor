from setuptools import setup, find_packages

setup(
    name="poliprompt",
    version="0.2.1",
    package_dir={"": "src"},  
    packages=find_packages(where="src"), 
    python_requires=">=3.10",
    install_requires=[
        "langchain",
        "langchain-openai",
        "langgraph",
        "tenacity",
        "python-dotenv",
        "tqdm",
        "openai",
        "dashscope"
    ]
)