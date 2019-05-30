import sys
sys.path.insert(0, './gluon-cv/')

import mxnet as mx
from mxnet import ndarray as F
from generator import defineG_unet
from discriminator import defineD_n_layers
from mxnet import gluon

from datasets.fascades import Fascades, _transformer

dataset = Fascades(
    root='/home/chenyifeng/pix2pix/datasets/facades/',
    split='train',
    mode='train',
    transform=_transformer(),
    crop_size=256)

ngpus = len(mx.test_utils.list_gpus())
batch_size = 1 * ngpus
loader = gluon.data.DataLoader(dataset, batch_size, shuffle=False, last_batch='rollover', num_workers=4)

load_iter = loader.__iter__()
image, label = load_iter.next()

gen = defineG_unet(input_nc=3, output_nc=3, ngf=64, ctx=mx.gpu(0))
dis = defineD_n_layers(input_nc=3, output_nc=3, ndf=64, n_layers=3, ctx=mx.gpu(0))

from gluoncv.utils.parallel import split_and_load
from parallel import DataParallelCriterionAug, DataParallelModelAug
from gluoncv.model_zoo.segbase import SegEvalModel

gen_net = DataParallelModelAug(gen, [mx.gpu(i) for i in range(ngpus)], True)
dis_net = DataParallelModelAug(dis, [mx.gpu(i) for i in range(ngpus)], True)
# criterion =  DataParallelCriterion(_criterion, [mx.gpu(i) for i in range(ngpus)] , True)
# evaluator = DataParallelModel(SegEvalModel(model), [ mx.gpu(i) for i in range(ngpus)])

from gluoncv.utils.lr_scheduler import LRScheduler

base_lr = 1e-3

optimizer_params = {
    'learning_rate': base_lr,
    'wd': 0.0,
    #     'lr_scheduler': scheduler,
    'momentum': 0.9
}

gen_opt = mx.gluon.Trainer(gen.collect_params(), 'sgd', optimizer_params, kvstore=mx.kv.create('device'))
dis_opt = mx.gluon.Trainer(dis.collect_params(), 'sgd', optimizer_params, kvstore=mx.kv.create('device'))

from mxnet import autograd
from criterion import GeneratorCriterion, DiscriminatorCriterion

_GeneratorCriterion = GeneratorCriterion(weight=100)
_DiscriminatorCriterion = DiscriminatorCriterion()

g_criterion = DataParallelCriterionAug(_GeneratorCriterion, [mx.gpu(i) for i in range(ngpus)], True)
d_criterion = DataParallelCriterionAug(_DiscriminatorCriterion, [mx.gpu(i) for i in range(ngpus)], True)

real_A = image.as_in_context (mx.cpu (0))
real_B = label.as_in_context (mx.cpu (0))
real_AB = F.concat (*[real_A, real_B], dim=1)

with autograd.record ():
    fake_B = gen_net (real_A)
    real_A = split_and_load (real_A, [mx.gpu (i) for i in range (ngpus)])
    real_B = split_and_load (real_B, [mx.gpu (i) for i in range (ngpus)])

    fake_AB = [[F.concat (*[i, j[0]], dim=1), ] for (i, j) in zip (real_A, fake_B)]

    output_fake = dis_net._split_call (fake_AB)
    output_real = dis_net (real_AB)
    dloss = d_criterion (output_real, output_fake)
    autograd.backward (dloss)
    mx.nd.waitall ()
dis_opt.step (batch_size)

with autograd.record ():
    output_fake = dis_net._split_call (fake_AB)
    fake_real_B = [[i[0], j] for (i, j) in zip (fake_B, real_B)]
    gloss = g_criterion (output_fake, fake_real_B)
    gloss.backward ()
    mx.nd.waitall ()
gen_opt.step (batch_size)
