import os
os.environ["KERAS_BACKEND"] = "torch"
import keras as ks


class R2Loss(ks.losses.Loss):
    def __init__(self, use_mask=False, name="r2_loss"):
        super().__init__(name=name)
        self.use_mask = use_mask

    def call(self, y_true, y_pred):
        if self.use_mask:
            mask = (y_true != -1)
            y_true = ks.ops.where(mask, y_true, 0.0)
            y_pred = ks.ops.where(mask, y_pred, 0.0)
        SS_residue = ks.ops.sum(ks.ops.square(y_true - y_pred), axis=0)
        SS_total = ks.ops.sum(ks.ops.square(y_true - ks.ops.mean(y_true, axis=0)), axis=0)
        r2_loss = SS_residue / (SS_total + 1e-6)
        return ks.ops.mean(r2_loss)


class R2Metric(ks.metrics.Metric):
    def __init__(self, name="r2", **kwargs):
        super(R2Metric, self).__init__(name=name, )
        self.SS_residual = self.add_weight(name='SS_residual', shape=(6,), initializer='zeros')
        self.SS_total = self.add_weight(name='SS_total', shape=(6,), initializer='zeros')
        self.num_samples = self.add_weight(name='num_samples', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        SS_residual = ks.ops.sum(ks.ops.square(y_true - y_pred), axis=0)
        SS_total = ks.ops.sum(ks.ops.square(y_true - y_pred), axis=0)
        self.SS_residual.assign_add(SS_residual)
        self.SS_total.assign_add(SS_total)
        self.num_samples.assign_add(ks.ops.cast(ks.ops.shape(y_true)[0], "float32"))

    def result(self):
        r2 = 1 - self.SS_residual / (self.SS_total + 1e-6)
        return ks.ops.mean(r2)

    def reset_states(self):
        self.total_SS_residual.assign(0)
        self.total_SS_total.assign(0)
        self.num_samples.assign(0)

