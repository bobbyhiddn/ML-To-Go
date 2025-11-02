from setuptools import setup, find_packages

setup(
    name="ml-classification-training",
    version="0.1.0",
    description="ML Document Classification Training Program",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.1.0",
        "tensorflow>=2.15.0",
        "sentence-transformers>=2.2.2",
        "transformers>=4.35.0",
        "pypdf2>=3.0.1",
        "python-docx>=1.1.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "black>=23.11.0",
            "pylint>=3.0.2",
        ]
    },
)
