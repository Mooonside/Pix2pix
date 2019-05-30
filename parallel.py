from gluoncv.utils.parallel import DataParallelModel, \
    parallel_apply, tuple_map, DataParallelCriterion, \
    criterion_parallel_apply


class DataParallelModelAug(DataParallelModel):
    def __init__(self, module, ctx_list=None, sync=False):
        super(DataParallelModelAug, self).__init__(module, ctx_list, sync)

    def _split_call(self, inputs, **kwargs):
        assert(len(inputs) == len(self.ctx_list))
        kwargs = []
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])

        if len(self.ctx_list) == 1:
            return tuple([tuple_map(self.module(*inputs[0], **kwargs[0]))])

        inputs = tuple(inputs)
        kwargs = tuple(kwargs)

        return parallel_apply(self.module, inputs, kwargs, self.sync)


class DataParallelCriterionAug(DataParallelCriterion):
    def __init__(self, module, ctx_list, sync):
        super(DataParallelCriterionAug, self).__init__(module, ctx_list, sync)

    def __call__(self, inputs, targets, **kwargs):
        # the inputs should be the outputs of DataParallelModel
        if not self.ctx_list:
            return self.module(inputs, *targets, **kwargs)

        assert(len(inputs) == len(self.ctx_list))
        assert(len(targets) == len(self.ctx_list))
        kwargs = []
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])

        if len(self.ctx_list) == 1:
            return tuple_map(self.module(*(inputs[0] + targets[0]), **kwargs[0]))
        return criterion_parallel_apply(self.module, inputs, targets, kwargs, self.sync)