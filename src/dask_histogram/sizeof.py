import dask


def register(sizeof):
    import boost_histogram as bh

    @sizeof.register(bh.Histogram)
    def register_bh_Histogram(data):
        return dask.sizeof.sizeof(data.view(flow=True))

    @sizeof.register_lazy("hist")
    def lazy_register_hist_Hist():
        import hist

        @sizeof.register(hist.Hist)
        def register_hist_Hist(data):
            return dask.sizeof.sizeof(data.view(flow=True))
