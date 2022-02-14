# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()
test_requirements = []
setup(
    author = "Francois Vanderseypen",
    author_email = 'swa@orbifold.net',
    python_requires = '>=3.9',
    name = 'sugiyama',
    version = '0.1.0',
    description = 'Sugiyama Graph Layout',
    long_description = readme,
    url = 'https://github.com/Orbifold/sugiyama',
    license = license,
    packages = find_packages(exclude = ('sugiyama', 'sugiyama.*')),
    test_suite = 'tests',
    tests_require = test_requirements,
)
