try:
    from dask.layers import DataFrameTreeReduction
except ImportError:
    try:
        from dask_awkward.layers import AwkwardTreeReductionLayer as DataFrameTreeReduction
    except ImportError:
        DataFrameTreeReduction = None

if DataFrameTreeReduction is None:
    raise ImportError("DataFrameReduction is unimportable - either downgrade dask to <2025"
                      "or install dask-awkward >=2025.")


class MockableDataFrameTreeReduction(DataFrameTreeReduction):
    def mock(self):
        return MockableDataFrameTreeReduction(
            name=self.name,
            name_input=self.name_input,
            npartitions_input=1,
            concat_func=self.concat_func,
            tree_node_func=self.tree_node_func,
            finalize_func=self.finalize_func,
            split_every=self.split_every,
            tree_node_name=self.tree_node_name,
        )
