from setuptools import setup, find_packages

setup(
    name='volforecast',
    version='0.1.0',
    packages=find_packages(),
    install_requires=['yfinance', 'pandas', 'matplotlib', 'numpy'],
    author='Fernando TSI',
    description='Modelo de previsão de retornos baseado em volatilidade e controle de drawdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
