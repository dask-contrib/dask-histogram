from dask.layers import DataFrameTreeReduction


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
