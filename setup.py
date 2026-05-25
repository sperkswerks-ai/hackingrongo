"""
hackingrongo — Rongorongo Hybrid Decipherment Pipeline
setup.py (PEP 517-compatible; also usable with plain `pip install -e .`)

All version constraints here must stay in sync with requirements.txt.
"""

from setuptools import find_packages, setup

# Read the long description from README so PyPI / local installs display it.
with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hackingrongo",
    version="0.1.0",
    description="Hybrid CNN + analytical decipherment pipeline for the rongorongo script",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Sarah Perkins",
    python_requires=">=3.11,<3.13",
    packages=find_packages(
        # Only include the inner `hackingrongo/` package tree, not the tests or notebooks.
        where=".",
        include=["hackingrongo", "hackingrongo.*"],
    ),
    install_requires=[
        "torch>=2.2,<3.0",
        "torchvision>=0.17,<1.0",
        "numpy>=1.26",
        "scipy>=1.12,<2.0",
        "hydra-core>=1.3,<2.0",
        "omegaconf>=2.3,<3.0",
        "Pillow>=10.3,<11.0",
        "click>=8.1,<9.0",
        "jinja2>=3.1,<4.0",
        # Layer 4Q: QUBO annealing
        # dimod is the framework; SA uses a pure numpy fallback so no D-Wave
        # runtime packages are required. For QPU access install
        # dwave-ocean-sdk>=6.0 separately (requires a Leap account).
        "dimod>=0.12",
    ],
    extras_require={
        # Install with: pip install -e ".[dev]"
        "dev": [
            "pytest>=8.1,<9.0",
            "pytest-cov>=5.0,<6.0",
            "mypy>=1.9,<2.0",
            "jupyter>=1.0,<2.0",
            "ipykernel>=6.29,<7.0",
        ],
    },
    entry_points={
        # CLI entry points defined in hackingrongo/cli.py via @click.group().
        "console_scripts": [
            "hackingrongo=hackingrongo.cli:cli",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    # Include the Hydra config directory in the installed package so that
    # `hydra.initialize(config_path="conf")` resolves correctly after install.
    package_data={
        "hackingrongo": [],
    },
    include_package_data=True,
)
