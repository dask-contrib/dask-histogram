from setuptools import setup


extras_require = {
    "test": ["pytest", "dask[dataframe]"],
    "docs": ["sphinx>=4.0.0", "dask-sphinx-theme", "autodocsumm"],
}

extras_require["complete"] = sorted(set(sum(extras_require.values(), [])))

setup(
    extras_require=extras_require,
)
