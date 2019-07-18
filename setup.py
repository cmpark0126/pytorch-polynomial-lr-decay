from setuptools import setup

with open('README.md') as f:
    long_description=f.read()  

setup(
    name='torch_poly_lr_decay',
    version='0.0.1',
    author='Chunmyong Park',
    description='Polynomial Learning Rate Decay Scheduler for PyTorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    maintainer='Chunmyong Park',
    zip_safe=False,
    packages=['torch_poly_lr_decay'],
    install_requires=['torch'],
)
