from mxnet import cpu, initializer
from mxnet.gluon import HybridBlock
from mxnet.gluon import nn


@initializer.register
class ShiftNormal(initializer.Initializer):
    def __init__(self, mean=0.0, sigma=0.01):
        super(ShiftNormal, self).__init__(mean=mean, sigma=sigma)
        self.sigma = sigma
        self.mean = mean

    def _init_weight(self, _, arr):
        initializer.random.normal(self.mean, self.sigma, out=arr)


def _encoder_module(in_channesl, out_channels, norm_layer=nn.BatchNorm, norm_kwargs=None):
    """
    :param in_channesl:
    :param out_channels:
    :param act_fn:
    :param norm_layer: None
    :param norm_kwargs:
    :return:
    """
    module = nn.HybridSequential()
    module.add(nn.LeakyReLU(0.2))
    module.add(nn.Conv2D(channels=out_channels, kernel_size=4, strides=2, padding=1, in_channels=in_channesl))
    if norm_layer is not None:
        norm_kwargs = {} if norm_kwargs is None else {**norm_kwargs}
        norm_kwargs['in_channels'] = out_channels
        module.add(nn.BatchNorm(in_channels=out_channels))
    return module


def _decoder_module(in_channesl, out_channels, norm_layer=nn.BatchNorm, norm_kwargs=None, dp=0.0):
    """
    :param in_channesl:
    :param out_channels:
    :param act_fn:
    :param norm_layer: None
    :param norm_kwargs:
    :return:
    """
    module = nn.HybridSequential()
    module.add(nn.Activation('relu'))
    module.add(nn.Conv2DTranspose(channels=out_channels, kernel_size=4, strides=2, padding=1, in_channels=in_channesl))
    if norm_layer is not None:
        norm_kwargs = {} if norm_kwargs is None else {**norm_kwargs}
        norm_kwargs['in_channels'] = out_channels
        module.add(nn.BatchNorm(in_channels=out_channels))
    if dp > 0:
        module.add(nn.Dropout(dp))
    return module


class defineG_encoder_decoder(HybridBlock):
    def __init__(self, input_nc, output_nc, ngf, ctx=cpu(0)):
        super(defineG_encoder_decoder, self).__init__()
        with self.name_scope():
            self.encoder = nn.HybridSequential()
            encoder_1 = nn.Conv2D(channels=ngf, kernel_size=4, strides=2, padding=1, in_channels=input_nc)
            encoder_2 = _encoder_module(ngf * 1, ngf * 2)
            encoder_3 = _encoder_module(ngf * 2, ngf * 4)
            encoder_4 = _encoder_module(ngf * 4, ngf * 8)
            encoder_5 = _encoder_module(ngf * 8, ngf * 8)
            encoder_6 = _encoder_module(ngf * 8, ngf * 8)
            encoder_7 = _encoder_module(ngf * 8, ngf * 8)
            encoder_8 = _encoder_module(ngf * 8, ngf * 8, norm_layer=None)
            self.encoder.add(*[encoder_1, encoder_2, encoder_3, encoder_4,
                               encoder_5, encoder_6, encoder_7, encoder_8])

            self.decoder = nn.HybridSequential()
            decoder_1 = _decoder_module(ngf * 8, ngf * 8, dp=0.5)
            decoder_2 = _decoder_module(ngf * 8, ngf * 8, dp=0.5)
            decoder_3 = _decoder_module(ngf * 8, ngf * 8, dp=0.5)
            decoder_4 = _decoder_module(ngf * 8, ngf * 8)
            decoder_5 = _decoder_module(ngf * 8, ngf * 4)
            decoder_6 = _decoder_module(ngf * 4, ngf * 2)
            decoder_7 = _decoder_module(ngf * 2, ngf * 1)
            decoder_8 = _decoder_module(ngf * 1, output_nc, norm_layer=None)
            self.decoder.add(*[decoder_1, decoder_2, decoder_3, decoder_4,
                               decoder_5, decoder_6, decoder_7, decoder_8])
            self.decoder.add(nn.Activation('tanh'))

        self.encoder.collect_params('.*weight').initialize(ctx=ctx, init=ShiftNormal(0, 0.02))
        self.encoder.collect_params('.*gamma|.*running_var').initialize(ctx=ctx, init=ShiftNormal(1.0, 0.02))
        self.encoder.collect_params('.*bias|.*running_mean|.*beta').initialize(ctx=ctx, init=initializer.Zero())
        self.decoder.collect_params('.*weight').initialize(ctx=ctx, init=ShiftNormal(0, 0.02))
        self.decoder.collect_params('.*gamma|.*running_var').initialize(ctx=ctx, init=ShiftNormal(1.0, 0.02))
        self.decoder.collect_params('.*bias|.*running_mean|.*beta').initialize(ctx=ctx, init=initializer.Zero())

    def hybrid_forward(self, F, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class defineG_unet(HybridBlock):
    def __init__(self, input_nc, output_nc, ngf, ctx=cpu(0)):
        super(defineG_unet, self).__init__()
        with self.name_scope():
            self.encoder = nn.HybridSequential()
            encoder_1 = nn.Conv2D(channels=ngf, kernel_size=4, strides=2, padding=1, in_channels=input_nc)
            encoder_2 = _encoder_module(ngf * 1, ngf * 2)
            encoder_3 = _encoder_module(ngf * 2, ngf * 4)
            encoder_4 = _encoder_module(ngf * 4, ngf * 8)
            encoder_5 = _encoder_module(ngf * 8, ngf * 8)
            encoder_6 = _encoder_module(ngf * 8, ngf * 8)
            encoder_7 = _encoder_module(ngf * 8, ngf * 8)
            encoder_8 = _encoder_module(ngf * 8, ngf * 8, norm_layer=None)
            self.encoder.add(*[encoder_1, encoder_2, encoder_3, encoder_4,
                               encoder_5, encoder_6, encoder_7, encoder_8])

            self.decoder = nn.HybridSequential()
            decoder_1 = _decoder_module(ngf * 8, ngf * 8, dp=0.5)
            decoder_2 = _decoder_module(ngf * 8 * 2, ngf * 8, dp=0.5)
            decoder_3 = _decoder_module(ngf * 8 * 2, ngf * 8, dp=0.5)
            decoder_4 = _decoder_module(ngf * 8 * 2, ngf * 8)
            decoder_5 = _decoder_module(ngf * 8 * 2, ngf * 4)
            decoder_6 = _decoder_module(ngf * 4 * 2, ngf * 2)
            decoder_7 = _decoder_module(ngf * 2 * 2, ngf * 1)
            decoder_8 = _decoder_module(ngf * 1 * 2, output_nc, norm_layer=None)
            self.decoder.add(*[decoder_1, decoder_2, decoder_3, decoder_4,
                               decoder_5, decoder_6, decoder_7, decoder_8])

        self.encoder.collect_params('.*weight').initialize(ctx=ctx, init=ShiftNormal(0, 0.02))
        self.encoder.collect_params('.*gamma|.*running_var').initialize(ctx=ctx, init=ShiftNormal(1.0, 0.02))
        self.encoder.collect_params('.*bias|.*running_mean|.*beta').initialize(ctx=ctx, init=initializer.Zero())
        self.decoder.collect_params('.*weight').initialize(ctx=ctx, init=ShiftNormal(0, 0.02))
        self.decoder.collect_params('.*gamma|.*running_var').initialize(ctx=ctx, init=ShiftNormal(1.0, 0.02))
        self.decoder.collect_params('.*bias|.*running_mean|.*beta').initialize(ctx=ctx, init=initializer.Zero())

    def hybrid_forward(self, F, x):
        encodings = []

        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            encodings.append(x)

        for i in range(len(self.decoder) - 1):
            x = self.decoder[i](x)
            x = F.concat(*[x, encodings[-i - 2]], dim=1)
        x = self.decoder[-1](x)
        x = F.tanh(x)
        return x


if __name__ == '__main__':
    generater = defineG_unet(3, 3, 16)
    # print(generater._collect_params_with_prefix().keys())
    # x = mx.nd.random.randn(1, 3, 256, 256)
    # y = generater(x)
    # print(y.shape)
    # print(generater)
