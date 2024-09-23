from setuptools import setup, find_packages

setup(
    name='model-to-predict_weather',
    version='0.0.1',
    author='Edgar Omar Fragoso',
    author_email='edgarfragosogarcia@gmail.com',
    description='This project take open source data from https://www.kaggle.com/jsphyg/weather-dataset-rattle-package to train Decition tree model',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/edgarom/deploying-machine-learning-models',
    packages=find_packages(),  # Automatically find the packages in your project
    python_requires='>=3.6',
    install_requires=[
        # List your project dependencies here
        'pandas',
        'scikit-learn',
        'numpy',
        'feature_engine',
        'strictyaml',
        'pydantic',
        'pathlib',
        'typing'
        # Add more dependencies as needed
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: Edgar License',
        'Operating System :: Linux',
    ],
)