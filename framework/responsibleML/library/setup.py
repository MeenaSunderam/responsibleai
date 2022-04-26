import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name='spectre',
    packages=setuptools.find_packages(),
    version='0.1.3',
    description='SPECTRE',
    author='Meenakshisundaram.t@gmail.com',
    license='MIT',
    install_requires=[],
    setup_requires=['pytest-runner', 'pandas', 'numpy', 'codecarbon', 'opacus', 'captum'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
    python_requires='>=3.6',
)