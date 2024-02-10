from __future__ import annotations

import math
import operator
from typing import Any, Callable

import numpy as np
import toolz
from dask.highlevelgraph import Layer
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


class MockableMultiSourceTreeReduction(Layer):
    """Tree reduction over multiple input collections known to be the
       same type. This layer can be summed to join over input collections.
       Mocking is achieved by taking the first partition of each input.

    Parameters
    ----------
    name : str
        Name to use for the constructed layer.
    names_inputs : tuple[str]
        Name of the input layers that are being reduced together.
    npartitions_inputs : tuple[int]
        Number of partitions in each input layer.
    concat_func : callable
        Function used by each tree node to reduce a list of inputs
        into a single output value. This function must accept only
        a list as its first positional argument.
    tree_node_func : callable
        Function used on the output of ``concat_func`` in each tree
        node. This function must accept the output of ``concat_func``
        as its first positional argument.
    finalize_func : callable, optional
        Function used in place of ``tree_node_func`` on the final tree
        node(s) to produce the final output for each split. By default,
        ``tree_node_func`` will be used.
    split_every : int, optional
        This argument specifies the maximum number of input nodes
        to be handled by any one task in the tree. Defaults to 32.
    split_out : int, optional
        This argument specifies the number of output nodes in the
        reduction tree. If ``split_out`` is set to an integer >=1, the
        input tasks must contain data that can be indexed by a ``getitem``
        operation with a key in the range ``[0, split_out)``.
    output_partitions : list, optional
        List of required output partitions. This parameter is used
        internally by Dask for high-level culling.
    tree_node_name : str, optional
        Name to use for intermediate tree-node tasks.
    """

    name: str
    names_inputs: tuple[str]
    npartitions_inputs: tuple[int]
    concat_func: Callable
    tree_node_func: Callable
    finalize_func: Callable | None
    split_every: int
    split_out: int
    output_partitions: list[int]
    tree_node_name: str
    widths: list[int]
    height: int

    def __init__(
        self,
        name: str,
        names_inputs: tuple[str],
        npartitions_inputs: tuple[int],
        concat_func: Callable,
        tree_node_func: Callable,
        finalize_func: Callable | None = None,
        split_every: int = 32,
        split_out: int | None = None,
        output_partitions: list[int] | None = None,
        tree_node_name: str | None = None,
        annotations: dict[str, Any] | None = None,
    ):
        if len(names_inputs) == 0:
            raise ValueError("Must specify at least one input!")
        if len(names_inputs) != len(npartitions_inputs):
            raise ValueError(
                "Must specify same number of input collections in names_inputs and npartitions_inputs"
            )

        super().__init__(annotations=annotations)
        self.name = name
        self.names_inputs = names_inputs
        self.npartitions_inputs = npartitions_inputs
        self.concat_func = concat_func
        self.tree_node_func = tree_node_func
        self.finalize_func = finalize_func
        self.split_every = split_every
        self.split_out = split_out  # type: ignore
        self.output_partitions = (
            list(range(self.split_out or 1))
            if output_partitions is None
            else output_partitions
        )
        self.tree_node_name = tree_node_name or "tree_node-" + self.name

        # Calculate tree widths and height
        # (Used to get output keys without materializing)
        parts = sum(self.npartitions_inputs)
        self.widths = [parts]
        while parts > 1:
            parts = math.ceil(parts / self.split_every)
            self.widths.append(int(parts))
        self.height = len(self.widths)

    def _make_key(self, *name_parts, split=0):
        # Helper function construct a key
        # with a "split" element when
        # bool(split_out) is True
        return name_parts + (split,) if self.split_out else name_parts

    def _define_task(self, input_keys, final_task=False):
        # Define nested concatenation and func task
        if final_task and self.finalize_func:
            outer_func = self.finalize_func
        else:
            outer_func = self.tree_node_func
        return (toolz.pipe, input_keys, self.concat_func, outer_func)

    def _construct_graph(self):
        """Construct graph for a tree reduction."""

        dsk = {}
        if not self.output_partitions:
            return dsk

        # Deal with `bool(split_out) == True`.
        # These cases require that the input tasks
        # return a type that enables getitem operation
        # with indices: [0, split_out)
        # Therefore, we must add "getitem" tasks to
        # select the appropriate element for each split
        names_inputs_use = self.names_inputs
        if self.split_out:
            names_inputs_use = (
                names_inputs_use + "-split" for names_inputs_use in names_inputs_use
            )
            for s in self.output_partitions:
                for ikey, name_input_use in enumerate(names_inputs_use):
                    for p in range(self.npartitions_inputs[ikey]):
                        dsk[self._make_key(name_input_use, p, split=s)] = (
                            operator.getitem,
                            (self.names_inputs[ikey], p),
                            s,
                        )

        # np.int32 so nothing funny happens on windows
        input_part_counts = np.array(self.npartitions_inputs, dtype=np.int32)
        input_names_indices = np.arange(0, len(input_part_counts), dtype=np.int32)
        input_part_indices = np.concatenate(
            [
                np.arange(0, npartitions_input, dtype=np.int32)
                for npartitions_input in self.npartitions_inputs
            ],
            axis=0,
        )
        input_names_parents = np.repeat(input_names_indices, input_part_counts)
        if self.height >= 2:
            # Loop over output splits
            for s in self.output_partitions:
                # Loop over reduction levels
                for depth in range(1, self.height):
                    # Loop over reduction groups
                    for group in range(self.widths[depth]):
                        # Calculate inputs for the current group
                        p_max = self.widths[depth - 1]
                        lstart = self.split_every * group
                        lstop = min(lstart + self.split_every, p_max)
                        if depth == 1:
                            # Input nodes are from input layers
                            input_keys = [
                                self._make_key(
                                    names_inputs_use[input_names_parents[p]],
                                    input_part_indices[p],
                                    split=s,
                                )
                                for p in range(lstart, lstop)
                            ]
                        else:
                            # Input nodes are tree-reduction nodes
                            input_keys = [
                                self._make_key(
                                    self.tree_node_name, p, depth - 1, split=s
                                )
                                for p in range(lstart, lstop)
                            ]

                        # Define task
                        if depth == self.height - 1:
                            # Final Node (Use fused `self.tree_finalize` task)
                            assert (
                                group == 0
                            ), f"group = {group}, not 0 for final tree reduction task"
                            dsk[(self.name, s)] = self._define_task(
                                input_keys, final_task=True
                            )
                        else:
                            # Intermediate Node
                            dsk[
                                self._make_key(
                                    self.tree_node_name, group, depth, split=s
                                )
                            ] = self._define_task(input_keys, final_task=False)
        else:
            # Deal with single-partition case
            for s in self.output_partitions:
                input_keys = [self._make_key(names_inputs_use[0], 0, split=s)]
                dsk[(self.name, s)] = self._define_task(input_keys, final_task=True)

        return dsk

    def __repr__(self):
        return "MockableMultiSourceTreeReduction<name='{}', input_names={}, split_out={}>".format(
            self.name, self.names_inputs, self.split_out
        )

    def _output_keys(self):
        return {(self.name, s) for s in self.output_partitions}

    def get_output_keys(self):
        if hasattr(self, "_cached_output_keys"):
            return self._cached_output_keys
        else:
            output_keys = self._output_keys()
            self._cached_output_keys = output_keys
        return self._cached_output_keys

    def is_materialized(self):
        return hasattr(self, "_cached_dict")

    @property
    def _dict(self):
        """Materialize full dict representation"""
        if hasattr(self, "_cached_dict"):
            return self._cached_dict
        else:
            dsk = self._construct_graph()
            self._cached_dict = dsk
        return self._cached_dict

    def __getitem__(self, key):
        return self._dict[key]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        # Start with "base" tree-reduction size
        tree_size = (sum(self.widths[1:]) or 1) * (self.split_out or 1)
        if self.split_out:
            # Add on "split-*" tasks used for `getitem` ops
            return tree_size + sum(self.npartitions_inputs, 0) * len(
                self.output_partitions
            )
        return tree_size

    def _keys_to_output_partitions(self, keys):
        """Simple utility to convert keys to output partition indices."""
        splits = set()
        for key in keys:
            try:
                _name, _split = key
            except ValueError:
                continue
            if _name != self.name:
                continue
            splits.add(_split)
        return splits

    def _cull(self, output_partitions):
        return MockableMultiSourceTreeReduction(
            self.name,
            self.names_inputs,
            self.npartitions_inputs,
            self.concat_func,
            self.tree_node_func,
            finalize_func=self.finalize_func,
            split_every=self.split_every,
            split_out=self.split_out,
            output_partitions=output_partitions,
            tree_node_name=self.tree_node_name,
            annotations=self.annotations,
        )

    def cull(self, keys, all_keys):
        """Cull a MockableMultiSourceTreeReduction HighLevelGraph layer"""
        input_names = set()
        for i in range(len(self.names_inputs)):
            input_names.update(
                {(self.names_inputs[i], k) for k in range(self.npartitions_inputs[i])}
            )
        deps = {(self.name, 0): input_names}
        output_partitions = self._keys_to_output_partitions(keys)
        if output_partitions != set(self.output_partitions):
            culled_layer = self._cull(output_partitions)
            return culled_layer, deps
        else:
            return self, deps

    def mock(self):
        return MockableMultiSourceTreeReduction(
            name=self.name,
            names_inputs=self.names_inputs,
            npartitions_inputs=tuple(1 for _ in self.npartitions_inputs),
            concat_func=self.concat_func,
            tree_node_func=self.tree_node_func,
            finalize_func=self.finalize_func,
            split_every=self.split_every,
            tree_node_name=self.tree_node_name,
        )
