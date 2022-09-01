from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='AMEX_default_prediction',
      version="0.0.0",
      description="AMEX default prediction model",
      license="MIT",
      author="AMEX default prediction team",
      author_email="yuzhexiao93@gmail.com",
      install_requires=requirements,
      packages=find_packages(),
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
