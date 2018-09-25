from distutils.core import setup

setup(
    name='UCSFRadClassify',
    version='0.0.1',
    author='Scott Werwath',
    author_email='sbw@berkeley.edu',
    packages=['rad_classify'],
    description='Tools for classification of UCSF radiology reports',
    long_description=open('README.md').read(),
)
