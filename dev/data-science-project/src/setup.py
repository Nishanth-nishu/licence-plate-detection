from setuptools import setup, find_packages

setup(
    name='licence-plate-detection-project',
    version='0.1.0',
    author='R Nishanth',
    author_email='nishanth0962333@gmail.com',
    description='A project for license plate detection and recognition using YOLOv8.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch',
        'torchvision',
        'easyocr',
        'opencv-python-headless',
        'ultralytics',
        'pandas',
        'numpy',
        'Pillow',
        'PyYAML',
    ],
    entry_points={
        'console_scripts': [
            'train=src.train:main',
            'validate=src.validate:main',
            'test=src.test:main',
        ],
    },
)