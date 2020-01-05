from setuptools import setup

with open("README.en.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()
requirements = ['pathlib;python_version<"3.4"', 'typing;python_version<"3.5"']
setup(
    name='FastHMM',
    version='0.1.1',
    url='https://github.com/312shan/FastHMM',
    install_requires=requirements,
    license='MIT',
    author='Yunshan Chen',
    author_email='chenyunshan312@163.com',
    description='A python package for HMM model with fast train and decoding implementation',
    long_description=long_description,
    keywords='HMM hmm python viterbi',
)
