from setuptools import setup, find_packages

setup(name='MPE',
      version='1.0.0',
      description='MPE',
      url='not update',
      author='Jiantao Qiu',
      author_email='qjt15@mails.tsinghua.edu.cn',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=['gym', 'numpy-stl']
)
