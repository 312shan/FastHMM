from setuptools import setup, find_packages

with open("README.en.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()
requirements = ['pathlib;python_version<"3.4"', 'typing;python_version<"3.5"']
setup(
    name='FastHMM',
    version='0.1.1',
    author='Yunshan Chen',
    author_email='chenyunshan312@163.com',
    description='A python package for HMM model with fast train and decoding implementation',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/312shan/FastHMM',
    install_requires=requirements,
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    keywords='hmm viterbi',
)
