from setuptools import setup, find_packages

setup(
    name='statecnv',
    version='0.1.0',
    description='Bayesian state-space model for CNV classification from targeted sequencing',
    author='Austin Talbot',
    author_email='talbota@pillarbiosci.com',
    url='https://github.com/austinTalbot7241993/StateCNV',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'particles',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    python_requires='>=3.8',
)

