import torch 
import os 
import wandb
from .train_utils import batch_to_device, cycle

class Trainer(object):
    def __init__(self, dataset, diffusion_model, config):

        self.env_name = config["dataset"]["env_name"]

        #training
        self.save_checkpoints = config["training"]["save_checkpoints"]
        self.log_freq = config["training"]["log_freq"]
        self.save_freq = config["training"]["save_freq"]
        self.batch_size = config["training"]["batch_size"]
        self.gradient_accumulate_every = config["training"]["gradient_accumulate_every"]
        self.device = config["training"]["device"]
        self.train_lr = config["training"]["learning_rate"]
        self.log_to_wandb = config["wandb"]["log_to_wandb"]
    
        #data 
        self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))

        #model 
        self.model = diffusion_model
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=self.train_lr)

        # Initialize for loading checkpoints
        self.loaded_epoch = 0  # Default value if no checkpoint is loaded
        if config["training"]["load_checkpoint"]:
            self.load_path = config["training"]["load_path"]
            self.load()


    def train(self, n_train_steps, current_epoch):
        current_epoch = current_epoch + self.loaded_epoch + 1 
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch, device=self.device)
                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()
        

            if step % self.log_freq == 0:
                print(f"step: {step}: loss: {loss.detach().item()} \n")
                if self.log_to_wandb:
                    wandb.log({"loss": loss.detach().item()})

        if current_epoch % self.save_freq == self.save_freq - 1:
            self.save(current_epoch)
                
    def save(self, epoch):
        data = {
            'epoch': epoch,
            'model': self.model.state_dict()
        }
        current_working_directory = os.getcwd()
        savepath = os.path.join(current_working_directory, 'weights', f'{self.env_name}_checkpoint')
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        savepath = os.path.join(savepath, f'state_{epoch}.pt')
        torch.save(data, savepath)

    def load(self):
        data = torch.load(self.loadpath)
        self.loaded_epoch = data['epoch']
        self.model.load_state_dict(data['model'])