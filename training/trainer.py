import torch 
import os 
import wandb
from .train_utils import batch_to_device, cycle

class Trainer(object):
    def __init__(self, diffusion_model, dataset, config):

        #training
        self.save_checkpoints = config["training"]["save_checkpoints"]
        self.log_freq = config["training"]["log_freq"]
        self.save_freq = config["training"]["save_freq"]
        self.batch_size = config["training"]["batch_size"]
        self.gradient_accumulate_every = config["training"]["gradient_accumulate_every"]
        self.device = config["training"]["device"]

        #data 
        self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=self.train_batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))

        #model 
        self.model = diffusion_model


        #save & load 
        self.load_path = config["training"]["load_path"]

    def train(self, n_train_steps):
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch, device=self.device)
                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()
        
            if self.step % self.save_freq == 0:
                self.save()

            if self.step % self.log_freq == 0:
                wandb.log(loss.detach().item())

                
    def save(self):
        data = {
            'step': self.step,
            'model': self.model.state_dict()
        }
        savepath = os.path.join('..', 'weights', 'checkpoint')
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        if self.save_checkpoints:
            savepath = os.path.join(savepath, f'state_{self.step}.pt')
        else:
            savepath = os.path.join(savepath, 'state.pt')
        torch.save(data, savepath)

    def load(self):
        data = torch.load(self.loadpath)
        self.step = data['step']
        self.model.load_state_dict(data['model'])