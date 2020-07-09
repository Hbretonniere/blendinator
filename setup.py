from setuptools import find_packages, setup

setup(
    name="blendinator",
    version="0.0.1",
    description="Deep learning toolbox for galaxy segmentation and deblending",
    author="Hubert Bretonniere",
    author_email="hbreton@apc.in2p3.fr",
    packages=find_packages(),
    license="BSD",
    install_requires = [
        "numpy",
        "tensorflow>=2.0",
        "keras",
        "h5py",
        "pandas",
        "tqdm",
        "matplotlib",
        "tensorflow_probability"
    ],
    python_requires='>=3.6',
)