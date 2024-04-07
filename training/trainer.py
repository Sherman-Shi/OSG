from .train_utils import batch_to_device
class Trainer(object):
    def __init__(self, config):
        pass

    def train(self, n_train_steps):
        for i in range(self.gradient_accumulate_every):
            batch = next(self.dataloader)
            batch = batch_to_device(batch, device=self.device)
            loss, infos = self.model.loss(*batch)
            loss = loss / self.gradient_accumulate_every
            loss.backward()