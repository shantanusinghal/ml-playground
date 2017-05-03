from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
  'numpy==1.12.1',
  'tensorflow==1.1.0',
]

setup(
    name='dcgan-mnist',
    version='0.1',
    author = 'Shantanu Singhal',
    author_email = 'shantanusinghal2709@gmail.com',
    install_requires=REQUIRED_PACKAGES,
    packages=['trainer'],
    include_package_data=True,
    description='Generative Adversarial Network capable of generating images using the MNIST handwritten character dataset as training data',
    requires=[]
)
