def register(sizeof):
    @sizeof.register_lazy("boost_histogram")
    def lazy_register_boost_histogram_Histogram():
        import boost_histogram as bh
        import dask

        @sizeof.register(bh.Histogram)
        def register_boost_histogram_Histogram(data):
            return dask.sizeof.sizeof(data.view(flow=True))

    @sizeof.register_lazy("hist")
    def lazy_register_hist_Hist():
        import dask
        import hist

        @sizeof.register(hist.Hist)
        def register_hist_Hist(data):
            return dask.sizeof.sizeof(data.view(flow=True))
