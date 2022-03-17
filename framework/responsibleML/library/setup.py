import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name='responsibleML',
    packages=setuptools.find_packages(),
    version='0.1.1',
    description='Responsible ML',
    author='Meenakshisundaram.t@gmail.com',
    license='MIT',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
    python_requires='>=3.6',
)