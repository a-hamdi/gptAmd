from setuptools import setup, find_packages

setup(
    name="gpt2-implementation",
    version="0.1.0",
    description="A complete implementation of GPT-2 with ROCm support",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.5",
        "transformers>=4.15.0",
        "tokenizers>=0.10.3",
        "tqdm>=4.62.3",
        "wandb>=0.12.0",
        "pytest>=6.2.5",
        "sentencepiece>=0.1.96",
        "einops>=0.3.2",
        "datasets>=1.18.0",
        "regex>=2021.8.3",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "gpt2-generate=sample:main",
        ],
    },
)
