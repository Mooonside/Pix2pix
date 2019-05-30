from mxnet.gluon import HybridBlock
from mxnet.gluon import nn
from models.generator import ShiftNormal, initializer
from mxnet import cpu


class defineD_pixelGAN(HybridBlock):
    """discriminator with rf = 1
    """
    def __init__(self, input_nc, output_nc, ndf, norm_layer=nn.BatchNorm, norm_kwargs=None, ctx=cpu(0)):
        super(defineD_pixelGAN, self).__init__()
        with self.name_scope():
            if norm_kwargs is None:
                norm_kwargs = {}
            self.conv1 = nn.Conv2D(channels=ndf, kernel_size=1, strides=1, padding=0, in_channels=input_nc + output_nc)
            self.conv2 = nn.HybridSequential()
            self.conv2.add(
                nn.LeakyReLU(0.2),
                nn.Conv2D(channels=2 * ndf, kernel_size=1, strides=1, padding=0, in_channels=ndf * 2),
                norm_layer(in_channels=ndf, **norm_kwargs)
            )
            self.conv3 = nn.HybridSequential()
            self.conv3.add(
                nn.LeakyReLU(0.2),
                nn.Conv2D(channels=1, kernel_size=1, strides=1, padding=0, in_channels=ndf * 2),
                nn.Activation('sigmoid')
            )
        self.collect_params('.*weight').initialize(ctx=ctx, init=ShiftNormal(0, 0.02))
        self.collect_params('.*gamma|.*running_var').initialize(ctx=ctx, init=ShiftNormal(1.0, 0.02))
        self.collect_params('.*bias|.*running_mean|.*beta').initialize(ctx=ctx, init=initializer.Zero())


    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class defineD_n_layers(defineD_pixelGAN):
    """
    -- if n=0, then use pixelGAN (rf=1)
    -- else rf is 16 if n=1
    --            34 if n=2
    --            70 if n=3
    --            142 if n=4
    --            286 if n=5
    --            574 if n=6
    --      with maximum channels * 2 ^ min(n_layers, 8)
    """
    def __init__(self, input_nc, output_nc, ndf, n_layers=3, norm_layer=nn.BatchNorm, norm_kwargs=None, ctx=cpu(0)):
        if n_layers == 0:
            super(defineD_n_layers, self).__init__(input_nc, output_nc, ndf, norm_layer, norm_kwargs, ctx)
        else:
            super(defineD_pixelGAN, self).__init__()
            if norm_kwargs is None:
                norm_kwargs = {}
            with self.name_scope():
                self.conv1 = nn.Conv2D(channels=ndf, kernel_size=4, strides=2, padding=1, in_channels=input_nc + output_nc)

                self.conv2 = nn.HybridSequential()
                self.conv2.add(nn.LeakyReLU(0.2))
                in_channels, out_channels = ndf, ndf
                for i in range(n_layers - 1):
                    in_channels = out_channels
                    out_channels = ndf * min(1 << (i + 1), 8)
                    self.conv2.add(
                        nn.Conv2D(channels=out_channels, kernel_size=4, strides=2, padding=1, in_channels=in_channels),
                        norm_layer(in_channels=out_channels, **norm_kwargs),
                        nn.LeakyReLU(0.2),
                    )

                in_channels = out_channels
                out_channels = ndf * min(1 << n_layers, 8)
                self.conv3 = nn.HybridSequential()
                self.conv3.add(
                    nn.Conv2D(channels=out_channels, kernel_size=4, strides=1, padding=1, in_channels=in_channels),
                    norm_layer(in_channels=out_channels, **norm_kwargs),
                    nn.LeakyReLU(0.2),
                    nn.Conv2D(channels=1, kernel_size=4, strides=1, padding=1, in_channels=out_channels),
                    nn.Activation('sigmoid')
                )
            self.collect_params('.*weight').initialize(ctx=ctx, init=ShiftNormal(0, 0.02))
            self.collect_params('.*gamma|.*running_var').initialize(ctx=ctx, init=ShiftNormal(1.0, 0.02))
            self.collect_params('.*bias|.*running_mean|.*beta').initialize(ctx=ctx, init=initializer.Zero())

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


if __name__ == '__main__':
    d = defineD_n_layers(3, 3, 16, n_layers=2)
    import mxnet as mx
    x = mx.nd.random.randn(1, 6, 256, 256)
    y = d(x)
    print(y.shape)
