from setuptools import setup

extras_require = {
    "test": ["pytest", "dask[dataframe]", "hist"],
    "docs": [
        "sphinx>=4.0.0",
        "dask[dataframe]",
        "dask-sphinx-theme>=2.0.0",
        "autodocsumm",
    ],
}

extras_require["complete"] = sorted(set(sum(extras_require.values(), [])))

setup(
    extras_require=extras_require,
)
