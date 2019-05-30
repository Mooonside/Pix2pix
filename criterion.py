from mxnet.gluon.loss import SigmoidBinaryCrossEntropyLoss, L1Loss, Loss
from gluoncv.utils.parallel import DataParallelCriterion


class GeneratorCriterion(Loss):
    def __init__(self, weight=100, batch_axis=0, **kwargs):
        """
        :param weight: for l1 loss
        :param batch_axis:
        :param kwargs:
        """
        super(GeneratorCriterion, self).__init__(weight, batch_axis, **kwargs)
        self.bce_loss = SigmoidBinaryCrossEntropyLoss(from_sigmoid=True, batch_axis=batch_axis)
        self.l1_loss = L1Loss(weight=weight, batch_axis=0)


    def hybrid_forward(self, F, fake_pred, fake, real):
        real_label = F.ones_like(fake_pred)
        fake_loss = self.bce_loss(fake_pred, real_label)
        diff = self.l1_loss(fake, real)
        loss = fake_loss + diff
        return loss


class DiscriminatorCriterion(Loss):
    def __init__(self, weight=1, batch_axis=0, **kwargs):
        """
        :param weight: for l1 loss
        :param batch_axis:
        :param kwargs:
        """
        super(DiscriminatorCriterion, self).__init__(weight, batch_axis, **kwargs)
        self.bce_loss = SigmoidBinaryCrossEntropyLoss(from_sigmoid=True, batch_axis=batch_axis)


    def hybrid_forward(self, F, real_pred, fake_pred):
        real_label = F.ones_like(real_pred)
        fake_label = F.zeros_like(fake_pred)

        real_loss = self.bce_loss(real_pred, real_label)
        fake_loss = self.bce_loss(fake_pred, fake_label)
        loss = real_loss + fake_loss
        return loss