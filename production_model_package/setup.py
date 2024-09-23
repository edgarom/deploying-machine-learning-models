from setuptools import setup, find_packages

# Package meta-data.
NAME = 'predict-regression-model'
DESCRIPTION = "Example regression model package from Train In Data."
URL = "https://github.com/trainindata/testing-and-monitoring-ml-deployments"
EMAIL = "edgarfragosogarcia@gmail.com"
AUTHOR = "Edgar"
REQUIRES_PYTHON = ">=3.10.0"


# The rest you shouldn't have to touch too much :)
# ------------------------------------------------

long_description = DESCRIPTION

# Load the package's VERSION file as a dictionary.
about = {}
ROOT_DIR = Path(__file__).resolve().parent
REQUIREMENTS_DIR = ROOT_DIR / 'requirements'
PACKAGE_DIR = ROOT_DIR / 'production_model_package'
with open(PACKAGE_DIR / "VERSION") as f:
    _version = f.read().strip()
    about["__version__"] = _version


# What packages are required for this module to be executed?
def list_reqs(fname="requirements.txt"):
    with open(REQUIREMENTS_DIR / fname) as fd:
        return fd.read().splitlines()

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
        'production_model_package'
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