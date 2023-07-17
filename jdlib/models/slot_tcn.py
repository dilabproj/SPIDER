from tqdm import tqdm
import wandb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from .slot_attention import build_grid, SoftPositionEmbed, SlotAttention


class SlotFormer(nn.Module):
    def __init__(self, 
                 encoder, encoder_args,
                 global_branch, global_branch_args,
                 local_branch, local_branch_args,  
                 augmentation=False,  # augmentation settings
                 intermediate_loss_func=None, global_loss_func=None, local_loss_func=None, # loss settings
                 intermediate_learning_rate=1e-4, global_learning_rate=1e-4, local_learning_rate=1e-4, epochs=10,  # training parameters
                 warmup_epochs=0,
                 device="cpu",  # other parameters
                ):
        super(SlotFormer, self).__init__()
        
        # encoder parameters
        self.out_channels = encoder_args['output_dims']
        
        # augmentation
        self.augmentation = augmentation
        
        # loss functions
        self.intermediate_loss_func = intermediate_loss_func
        self.global_loss_func = global_loss_func
        self.local_loss_func = local_loss_func
        
        # general parameters
        self.device = device
        
        # training parameters
        self.intermediate_learning_rate = intermediate_learning_rate
        self.local_learning_rate = local_learning_rate
        self.global_learning_rate = global_learning_rate
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        
        # encoder settings
        self.encoder = encoder(**encoder_args)
        self._encoder = torch.optim.swa_utils.AveragedModel(self.encoder)
        self._encoder.update_parameters(self.encoder)
        
        # custom position encoding
        
        # global branch settings
        self.global_branch = global_branch(**global_branch_args)
        self.global_projector = nn.Sequential(nn.Linear(self.out_channels, self.out_channels * 2**2),
                                              nn.BatchNorm1d(self.out_channels * 2**2), 
                                              nn.ReLU(),
                                              nn.Linear(self.out_channels * 2**2, self.out_channels), 
                                             )
        
        self._global_branch = torch.optim.swa_utils.AveragedModel(self.global_branch)
        self._global_branch.update_parameters(self.global_branch)
                                 
        self._global_projector = torch.optim.swa_utils.AveragedModel(self.global_projector)
        self._global_projector.update_parameters(self.global_projector)
        
        # local branch settings
        self.local_branch = local_branch(**local_branch_args)
        self.local_projector = nn.Sequential(nn.Linear(self.out_channels, self.out_channels * 2**2),
                                             nn.BatchNorm1d(local_branch_args['num_slots']), 
                                             nn.ReLU(),
                                             nn.Linear(self.out_channels * 2**2, self.out_channels), 
                                            )

        self._local_branch = torch.optim.swa_utils.AveragedModel(self.local_branch)
        self._local_branch.update_parameters(self.local_branch)
                      
        self._local_projector = torch.optim.swa_utils.AveragedModel(self.local_projector)
        self._local_projector.update_parameters(self.local_projector)
        
        # optimizer
        self.backbone_scheduler, self.branch_scheduler = self._set_optimizer()
        self.to(self.device)
        
    def _set_optimizer(self):
        backbone_optimizer = optim.AdamW([
            {"params": self.encoder.parameters(), "lr": self.intermediate_learning_rate}
        ])
        
        branch_optimizer = optim.AdamW([
            {"params": self.encoder.parameters(), "lr": max(self.local_learning_rate, self.global_learning_rate)},
            {"params": self.global_branch.parameters(), "lr": self.global_learning_rate},
            {"params": self.global_projector.parameters(), "lr": self.global_learning_rate},
            {"params": self.local_branch.parameters(), "lr": self.local_learning_rate},
            {"params": self.local_projector.parameters(), "lr": self.local_learning_rate},
        ])
        
        lr_lambda = lambda e: (e + 1) /  max(self.warmup_epochs, (e + 1))
        backbone_scheduler = optim.lr_scheduler.LambdaLR(backbone_optimizer, lr_lambda=lr_lambda)
        branch_scheduler = optim.lr_scheduler.LambdaLR(branch_optimizer, lr_lambda=lr_lambda)
        
        return backbone_scheduler, branch_scheduler
        
    def forward(self, x):
        # input shape: (N, Cin, T)
        # intermediate shape: (N, Cout, T)
        # global shape: (N, Cout)
        # local shappe: (N, N_S, Cout)
        if self.training:
            intermediate = self.encoder(x)
        else:
            intermediate = self.encoder(x, mask="all_true")

        global_repr = self.global_branch(intermediate, ecg=None)
        local_repr = self.local_branch(intermediate, ecg=None)
            
        return intermediate, global_repr, local_repr
    
    def fit(self, train_loader, valid_loader=None, wandb_logger=None, saved_path=None):  
        self.train()
        
        best_loss = 1e6
        losses = {}
            
        for e in tqdm(range(self.epochs), desc="Epochs", position=0, leave=True):            
            train_loss_dict = self.train_epoch(train_loader)
            valid_loss_dict = self.valid_epoch(valid_loader) if valid_loader is not None else {}
            
            epoch_loss_dict = train_loss_dict | valid_loss_dict
            epoch_lr_dict = {"backbone_lr": self.backbone_scheduler.get_last_lr()[0],
                             "branch_lr": self.branch_scheduler.get_last_lr()[0]}
            
            if wandb_logger is not None:
                wandb.log(epoch_loss_dict | epoch_loss_dict)
            
            # print(f"Epoch: {e}, loss: {epoch_loss:.6f} (g_loss: {epoch_g_loss:.6f}, l_loss: {epoch_l_loss:.6f})")
            # print(f"Epoch: {e}, backbone_lr: {backbone_lr:.6f}, branch_lr: {branch_lr:.6f}, warmup_epochs={self.warmup_epochs}")
            
            for k in epoch_loss_dict:
                if k in losses:
                    losses[k].append(epoch_loss_dict[k])
                else:
                    losses[k] = [epoch_loss_dict[k]]
            
            if (saved_path is not None) and (epoch_loss_dict['valid_total_loss'] < best_loss):
                best_loss = epoch_loss_dict['valid_total_loss']
                self.save(saved_path)
        
        return losses
    
    
    def encode(self, dataloader):
        self.eval()
        
        all_g_reprs, all_l_reprs = [[], []]
        with torch.no_grad():
            for x, y in tqdm(dataloader, desc="Repr. encoding: ", leave=False):
                x = x.to(self.device)
                intermediate, global_repr, local_repr = self(x)
                all_g_reprs.append(global_repr.cpu())
                all_l_reprs.append(local_repr.cpu())
        
        if len(all_g_reprs[-1].shape)==1:
            all_g_reprs[-1] = all_g_reprs[-1].unsqueeze(0)

        if len(all_l_reprs[-1].shape)==2:
            all_l_reprs[-1] = all_l_reprs[-1].unsqueeze(0)
        
        all_g_reprs = torch.cat(all_g_reprs, dim=0)
        all_l_reprs = torch.cat(all_l_reprs, dim=0)
        
        return all_g_reprs, all_l_reprs
    
    def train_epoch(self, data_loader): 
        self.train()
        epoch_i_loss = 0
        epoch_g_loss = 0
        epoch_l_loss = 0

        for x, y in tqdm(data_loader, desc="Training steps", position=1, leave=False):
            x = x.to(self.device)

            # Generating views
            x_aug = self.augmentation(x) if self.augmentation is not None else x

            # calculating intermediate loss and updating the weights of the encoder                
            intermediate = self.encoder(x_aug)
            with torch.no_grad():
                _intermediate = self._encoder(x, mask="all_true")

            if self.intermediate_loss_func is not None:
                i_loss = self.intermediate_loss_func(intermediate, _intermediate)
                epoch_i_loss += i_loss.item()

                self.backbone_scheduler.optimizer.zero_grad()
                i_loss.backward()
                self.backbone_scheduler.optimizer.step()

                # updating boostraping models
                self._encoder.update_parameters(self.encoder)

            # calculating final losses and updating the weights of the branches
            intermediate = self.encoder(x_aug)
            global_repr = self.global_branch(intermediate, ecg=None)
            global_projection = self.global_projector(global_repr)  # shape = (N, out_channels)
            local_repr = self.local_branch(intermediate, ecg=None, return_k_iter=True)
            
            if len(local_repr.shape)==4:
                local_projection = []
                for i in range(local_repr.shape[0]):
                    local_projection.append(self.local_projector(local_repr[i]))
                local_projection = torch.stack(local_projection)                        
            else:
                local_projection = self.local_projector(local_repr)
            
            # target representations
            with torch.no_grad():
                _intermediate = self._encoder(x, mask="all_true")
                _global_repr = self._global_branch(_intermediate, ecg=None)
                _global_projection = self._global_projector(_global_repr)
                _local_repr = self._local_branch(_intermediate, ecg=None, return_k_iter=True)
                if len(_local_repr.shape)==4:
                    _local_projection = []
                    for i in range(_local_repr.shape[0]):
                        _local_projection.append(self._local_projector(_local_repr[i]))
                    _local_projection = torch.stack(_local_projection)    
                else:
                    _local_projection = self._local_projector(_local_repr)
            
            # losses  
            update_branch = False
            branch_loss = 0
            if self.global_loss_func is not None:
                g_loss = self.global_loss_func(global_projection, _global_projection)
                branch_loss += g_loss
                epoch_g_loss += g_loss.item()
                update_branch = True

            if self.local_loss_func is not None:
                if len(local_projection.shape)==4:  # slot.shape = (n_iter, batch_size, n_slot, dim)
                    l_loss = 0
                    k = local_projection.shape[0]
                    for i in range(k): 
                        l_loss += self.local_loss_func(local_projection[i], _local_projection[i])
                    l_loss /= k
                else:
                    l_loss = self.local_loss_func(local_projection, _local_projection)
                
                branch_loss += l_loss
                epoch_l_loss += l_loss.item()
                update_branch = True

            # Rhythm-heartbeat independent loss

            # updating models
            if update_branch:
                self.branch_scheduler.optimizer.zero_grad()
                branch_loss.backward()
                self.branch_scheduler.optimizer.step()

                # updating boostraping models
                self._global_branch.update_parameters(self.global_branch)
                self._global_projector.update_parameters(self.global_projector)

                self._local_branch.update_parameters(self.local_branch)
                self._local_projector.update_parameters(self.local_projector)

        epoch_i_loss /= len(data_loader)
        epoch_g_loss /= len(data_loader)
        epoch_l_loss /= len(data_loader)
        epoch_loss = epoch_i_loss + epoch_g_loss + epoch_l_loss
        
        if self.intermediate_loss_func is not None: self.backbone_scheduler.step()
        if update_branch: self.branch_scheduler.step()
        
        return {"train_i_loss": epoch_i_loss, 
                "train_g_loss": epoch_g_loss, 
                "train_l_loss": epoch_l_loss, 
                "train_total_loss": epoch_loss}
    
    def train_epoch2(self, data_loader): # 2-step training with whole epoch
        self.train()
        epoch_i_loss = 0
        epoch_g_loss = 0
        epoch_l_loss = 0

        for x, y in tqdm(data_loader, desc="Training phase 1", position=1, leave=False):
            x = x.to(self.device)

            # Generating views
            x_aug = self.augmentation(x) if self.augmentation is not None else x

            # calculating intermediate loss and updating the weights of the encoder                
            intermediate = self.encoder(x_aug)
            with torch.no_grad():
                _intermediate = self._encoder(x, mask="all_true")

            if self.intermediate_loss_func is not None:
                i_loss = self.intermediate_loss_func(intermediate, _intermediate)
                epoch_i_loss += i_loss.item()

                self.backbone_scheduler.optimizer.zero_grad()
                i_loss.backward()
                self.backbone_scheduler.optimizer.step()

                # updating boostraping models
                self._encoder.update_parameters(self.encoder)
                
        for x, y in tqdm(data_loader, desc="Training phase 2", position=1, leave=False):
            x = x.to(self.device)
            
            # calculating final losses and updating the weights of the branches
            intermediate = self.encoder(x_aug)
            global_repr = self.global_branch(intermediate, ecg=None)
            global_projection = self.global_projector(global_repr)  # shape = (N, out_channels)
            local_repr = self.local_branch(intermediate, ecg=None)
            local_projection = self.local_projector(local_repr)
            
            # target representations
            with torch.no_grad():
                _intermediate = self._encoder(x, mask="all_true")
                _global_repr = self._global_branch(_intermediate, ecg=None)
                _global_projection = self._global_projector(_global_repr)
                _local_repr = self._local_branch(_intermediate, ecg=None)
                _local_projection = self._local_projector(_local_repr)
            
            # losses  
            update_branch = False
            branch_loss = 0
            if self.global_loss_func is not None:
                g_loss = self.global_loss_func(global_projection, _global_projection)
                branch_loss += g_loss
                epoch_g_loss += g_loss.item()
                update_branch = True

            if self.local_loss_func is not None:
                l_loss = self.local_loss_func(local_projection, _local_projection)
                branch_loss += l_loss
                epoch_l_loss += l_loss.item()
                update_branch = True

            # Rhythm-heartbeat independent loss

            # updating models
            if update_branch:
                self.branch_scheduler.optimizer.zero_grad()
                branch_loss.backward()
                self.branch_scheduler.optimizer.step()

                # updating boostraping models
                self._global_branch.update_parameters(self.global_branch)
                self._global_projector.update_parameters(self.global_projector)

                self._local_branch.update_parameters(self.local_branch)
                self._local_projector.update_parameters(self.local_projector)
        
        epoch_i_loss /= len(data_loader)
        epoch_g_loss /= len(data_loader)
        epoch_l_loss /= len(data_loader)
        epoch_loss = epoch_i_loss + epoch_g_loss + epoch_l_loss
        
        if self.intermediate_loss_func is not None: self.backbone_scheduler.step()
        if update_branch: self.branch_scheduler.step()
        
        return {"train_i_loss": epoch_i_loss, 
                "train_g_loss": epoch_g_loss, 
                "train_l_loss": epoch_l_loss, 
                "train_total_loss": epoch_loss}
    
    def valid_epoch(self, data_loader):
        self.eval()
        epoch_i_loss = 0
        epoch_g_loss = 0
        epoch_l_loss = 0
        
        for x, y in tqdm(data_loader, desc="Validation steps", position=1, leave=False):
            x = x.to(self.device)

            # Generating views
            x_aug = self.augmentation(x) if self.augmentation is not None else x

            # calculating intermediate loss and updating the weights of the encoder                
            with torch.no_grad():
                intermediate = self.encoder(x, mask="all_true")
                _intermediate = self._encoder(x, mask="all_true")

                if self.intermediate_loss_func is not None:
                    i_loss = self.intermediate_loss_func(intermediate, _intermediate)
                    epoch_i_loss += i_loss.item()

                # calculating final losses and updating the weights of the branches
                # target representations
                _global_repr = self._global_branch(_intermediate, ecg=None)
                _global_projection = self._global_projector(_global_repr)
                _local_repr = self._local_branch(_intermediate, ecg=None)
                _local_projection = self._local_projector(_local_repr)

                global_repr = self.global_branch(intermediate, ecg=None)
                global_projection = self.global_projector(global_repr)  # shape = (N, out_channels)
                local_repr = self.local_branch(intermediate, ecg=None)
                local_projection = self.local_projector(local_repr)

                # losses  
                branch_loss = 0
                if self.global_loss_func is not None:
                    g_loss = self.global_loss_func(global_projection, _global_projection)
                    branch_loss += g_loss
                    epoch_g_loss += g_loss.item()

                if self.local_loss_func is not None:
                    l_loss = self.local_loss_func(local_projection, _local_projection)
                    branch_loss += l_loss
                    epoch_l_loss += l_loss.item()

            # Rhythm-heartbeat independent loss

        epoch_i_loss /= len(data_loader)
        epoch_g_loss /= len(data_loader)
        epoch_l_loss /= len(data_loader)
        epoch_loss = epoch_i_loss + epoch_g_loss + epoch_l_loss
        
        return {"valid_i_loss": epoch_i_loss, 
                "valid_g_loss": epoch_g_loss, 
                "valid_l_loss": epoch_l_loss, 
                "valid_total_loss": epoch_loss}
    
    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save(self.state_dict(), fn)
    
    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn, map_location=self.device)
        self.load_state_dict(state_dict)
      
    
class SlotFormerPos(nn.Module):
    def __init__(self, 
                 encoder, encoder_args,
                 global_branch, global_branch_args,
                 local_branch, local_branch_args,  
                 augmentation=False,  # augmentation settings
                 intermediate_loss_func=None, global_loss_func=None, local_loss_func=None, # loss settings
                 intermediate_learning_rate=1e-4, global_learning_rate=1e-4, local_learning_rate=1e-4, epochs=10,  # training parameters
                 warmup_epochs=0,
                 device="cpu",  # other parameters
                ):
        super(SlotFormerPos, self).__init__()
        
        # encoder parameters
        self.out_channels = encoder_args['output_dims']
        
        # augmentation
        self.augmentation = augmentation
        
        # loss functions
        self.intermediate_loss_func = intermediate_loss_func
        self.global_loss_func = global_loss_func
        self.local_loss_func = local_loss_func
        
        # general parameters
        self.device = device
        
        # training parameters
        self.intermediate_learning_rate = intermediate_learning_rate
        self.local_learning_rate = local_learning_rate
        self.global_learning_rate = global_learning_rate
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        
        # encoder settings
        self.encoder = encoder(**encoder_args)
        self._encoder = torch.optim.swa_utils.AveragedModel(self.encoder)
        self._encoder.update_parameters(self.encoder)
        
        # custom position encoding
        
        # global branch settings
        self.global_branch = global_branch(**global_branch_args)
        self.global_projector = nn.Sequential(nn.Linear(self.out_channels, self.out_channels * 2**2),
                                              nn.BatchNorm1d(self.out_channels * 2**2), 
                                              nn.ReLU(),
                                              nn.Linear(self.out_channels * 2**2, self.out_channels), 
                                             )
        
        self._global_branch = torch.optim.swa_utils.AveragedModel(self.global_branch)
        self._global_branch.update_parameters(self.global_branch)
                                 
        self._global_projector = torch.optim.swa_utils.AveragedModel(self.global_projector)
        self._global_projector.update_parameters(self.global_projector)
        
        # local branch settings
        self.local_branch = local_branch(**local_branch_args)
        self.local_projector = nn.Sequential(nn.Linear(self.out_channels, self.out_channels * 2**2),
                                             nn.BatchNorm1d(local_branch_args['num_slots']), 
                                             nn.ReLU(),
                                             nn.Linear(self.out_channels * 2**2, self.out_channels), 
                                            )

        self._local_branch = torch.optim.swa_utils.AveragedModel(self.local_branch)
        self._local_branch.update_parameters(self.local_branch)
                      
        self._local_projector = torch.optim.swa_utils.AveragedModel(self.local_projector)
        self._local_projector.update_parameters(self.local_projector)
        
        # optimizer
        self.backbone_scheduler, self.branch_scheduler = self._set_optimizer()
        self.to(self.device)
        
    def _set_optimizer(self):
        backbone_optimizer = optim.Adam([
            {"params": self.encoder.parameters(), "lr": self.intermediate_learning_rate},
        ], weight_decay=0.0001)
        
        branch_optimizer = optim.Adam([
            {"params": self.encoder.parameters(), "lr": max(self.local_learning_rate, self.global_learning_rate)},
            {"params": self.global_branch.parameters(), "lr": self.global_learning_rate},
            {"params": self.global_projector.parameters(), "lr": self.global_learning_rate},
            {"params": self.local_branch.parameters(), "lr": self.local_learning_rate},
            {"params": self.local_projector.parameters(), "lr": self.local_learning_rate},
        ], weight_decay=0.0001)
        
        lr_lambda = lambda e: (e + 1) /  max(self.warmup_epochs, (e + 1))
        backbone_scheduler = optim.lr_scheduler.LambdaLR(backbone_optimizer, lr_lambda=lr_lambda)
        branch_scheduler = optim.lr_scheduler.LambdaLR(branch_optimizer, lr_lambda=lr_lambda)
        
        return backbone_scheduler, branch_scheduler
        
    def forward(self, x):
        # input shape: (N, Cin, T)
        # intermediate shape: (N, Cout, T)
        # global shape: (N, Cout)
        # local shappe: (N, N_S, Cout)
        if self.training:
            intermediate = self.encoder(x)
        else:
            intermediate = self.encoder(x, mask="all_true")

        global_repr = self.global_branch(intermediate, ecg=x)
        local_repr = self.local_branch(intermediate, ecg=x)
            
        return intermediate, global_repr, local_repr
    
    def fit(self, train_loader, valid_loader=None, wandb_logger=None, saved_path=None):  
        self.train()
        
        best_loss = 1e6
        losses = {}
            
        for e in tqdm(range(self.epochs), desc="Epochs", position=0, leave=True):            
            train_loss_dict = self.train_epoch(train_loader)
            valid_loss_dict = self.valid_epoch(valid_loader) if valid_loader is not None else {}
            
            epoch_loss_dict = train_loss_dict | valid_loss_dict
            epoch_lr_dict = {"backbone_lr": self.backbone_scheduler.get_last_lr()[0],
                             "branch_lr": self.branch_scheduler.get_last_lr()[0]}
            
            if wandb_logger is not None:
                wandb.log(epoch_loss_dict)
            
            # print(f"Epoch: {e}, loss: {epoch_loss:.6f} (g_loss: {epoch_g_loss:.6f}, l_loss: {epoch_l_loss:.6f})")
            # print(f"Epoch: {e}, backbone_lr: {backbone_lr:.6f}, branch_lr: {branch_lr:.6f}, warmup_epochs={self.warmup_epochs}")
            
            for k in epoch_loss_dict:
                if k in losses:
                    losses[k].append(epoch_loss_dict[k])
                else:
                    losses[k] = [epoch_loss_dict[k]]
            
            if (saved_path is not None) and (epoch_loss_dict['valid_total_loss'] < best_loss):
                best_loss = epoch_loss_dict['valid_total_loss']
                self.save(saved_path)
        
        return losses
    
    
    def encode(self, dataloader):
        self.eval()
        
        all_g_reprs, all_l_reprs = [[], []]
        with torch.no_grad():
            for x, y in tqdm(dataloader, desc="Repr. encoding: ", leave=False):
                x = x.to(self.device)
                intermediate, global_repr, local_repr = self(x)
                all_g_reprs.append(global_repr.cpu())
                all_l_reprs.append(local_repr.cpu())
        
        if len(all_g_reprs[-1].shape)==1:
            all_g_reprs[-1] = all_g_reprs[-1].unsqueeze(0)

        if len(all_l_reprs[-1].shape)==2:
            all_l_reprs[-1] = all_l_reprs[-1].unsqueeze(0)
        
        all_g_reprs = torch.cat(all_g_reprs, dim=0)
        all_l_reprs = torch.cat(all_l_reprs, dim=0)
        
        return all_g_reprs, all_l_reprs
    
    def train_epoch(self, data_loader): 
        self.train()
        epoch_i_loss = 0
        epoch_g_loss = 0
        epoch_l_loss = 0

        for x, y in tqdm(data_loader, desc="Training steps", position=1, leave=False):
            x = x.to(self.device)
            
            # Generating views 
            x_aug = self.augmentation(x) if self.augmentation is not None else x
            
            if self.intermediate_loss_func is not None:
                # calculating intermediate loss and updating the weights of the encoder                
                intermediate = self.encoder(x_aug)
                with torch.no_grad():
                    _intermediate = self._encoder(x, mask="all_true")
                
                i_loss = self.intermediate_loss_func(intermediate, _intermediate)
                epoch_i_loss += i_loss.item()

                self.backbone_scheduler.optimizer.zero_grad()
                i_loss.backward()
                self.backbone_scheduler.optimizer.step()

                # updating boostraping models
                self._encoder.update_parameters(self.encoder)

            # calculating final losses and updating the weights of the branches
            intermediate = self.encoder(x_aug)
            with torch.no_grad():
                _intermediate = self._encoder(x, mask="all_true")
                
            # losses  
            update_branch = False
            branch_loss = 0
            if self.global_loss_func is not None:
                # generating global representations
                global_repr = self.global_branch(intermediate, ecg=x)
                global_projection = self.global_projector(global_repr)  # shape = (N, out_channels)
                with torch.no_grad():
                    _global_repr = self._global_branch(_intermediate, ecg=x)
                    _global_projection = self._global_projector(_global_repr)
                
                # calculating global loss
                g_loss = self.global_loss_func(global_projection, _global_projection)
                branch_loss += g_loss
                epoch_g_loss += g_loss.item()
                update_branch = True

            if self.local_loss_func is not None:
                # generating local representations
                local_repr = self.local_branch(intermediate, ecg=x, return_k_iter=True)
                if len(local_repr.shape)==4:
                    local_projection = []
                    for i in range(local_repr.shape[0]):
                        local_projection.append(self.local_projector(local_repr[i]))
                    local_projection = torch.stack(local_projection)                        
                else:
                    local_projection = self.local_projector(local_repr)
                
                with torch.no_grad():
                    _local_repr = self._local_branch(_intermediate, ecg=x, return_k_iter=True)
                    if len(_local_repr.shape)==4:
                        _local_projection = []
                        for i in range(_local_repr.shape[0]):
                            _local_projection.append(self._local_projector(_local_repr[i]))
                        _local_projection = torch.stack(_local_projection)    
                    else:
                        _local_projection = self._local_projector(_local_repr)
                    
                # calculating local loss
                if len(local_projection.shape)==4:  # slot.shape = (n_iter, batch_size, n_slot, dim)
                    l_loss = 0
                    k = local_projection.shape[0]
                    for i in range(k): 
                        l_loss += self.local_loss_func(local_projection[i], _local_projection[i])
                    l_loss /= k
                else:
                    l_loss = self.local_loss_func(local_projection, _local_projection)
                
                branch_loss += l_loss
                epoch_l_loss += l_loss.item()
                update_branch = True

            # Rhythm-heartbeat independent loss

            # updating models
            if update_branch:
                self.branch_scheduler.optimizer.zero_grad()
                branch_loss.backward()
                self.branch_scheduler.optimizer.step()

                # updating boostraping models
                self._global_branch.update_parameters(self.global_branch)
                self._global_projector.update_parameters(self.global_projector)

                self._local_branch.update_parameters(self.local_branch)
                self._local_projector.update_parameters(self.local_projector)

        epoch_i_loss /= len(data_loader)
        epoch_g_loss /= len(data_loader)
        epoch_l_loss /= len(data_loader)
        epoch_loss = epoch_i_loss + epoch_g_loss + epoch_l_loss
        
        if self.intermediate_loss_func is not None: self.backbone_scheduler.step()
        if update_branch: self.branch_scheduler.step()
        
        return {"train_i_loss": epoch_i_loss, 
                "train_g_loss": epoch_g_loss, 
                "train_l_loss": epoch_l_loss, 
                "train_total_loss": epoch_loss}
    
    def train_epoch2(self, data_loader): # 2-step training with whole epoch
        self.train()
        epoch_i_loss = 0
        epoch_g_loss = 0
        epoch_l_loss = 0

        for x, y in tqdm(data_loader, desc="Training phase 1", position=1, leave=False):
            x = x.to(self.device)

            # Generating views
            x_aug = self.augmentation(x) if self.augmentation is not None else x

            # calculating intermediate loss and updating the weights of the encoder                
            intermediate = self.encoder(x_aug)
            with torch.no_grad():
                _intermediate = self._encoder(x, mask="all_true")

            if self.intermediate_loss_func is not None:
                i_loss = self.intermediate_loss_func(intermediate, _intermediate)
                epoch_i_loss += i_loss.item()

                self.backbone_scheduler.optimizer.zero_grad()
                i_loss.backward()
                self.backbone_scheduler.optimizer.step()

                # updating boostraping models
                self._encoder.update_parameters(self.encoder)
                
        for x, y in tqdm(data_loader, desc="Training phase 2", position=1, leave=False):
            x = x.to(self.device)
            
            # calculating final losses and updating the weights of the branches
            intermediate = self.encoder(x_aug)
            global_repr = self.global_branch(intermediate, ecg=x)
            global_projection = self.global_projector(global_repr)  # shape = (N, out_channels)
            local_repr = self.local_branch(intermediate, ecg=x)
            local_projection = self.local_projector(local_repr)
            
            # target representations
            with torch.no_grad():
                _intermediate = self._encoder(x, mask="all_true")
                _global_repr = self._global_branch(_intermediate, ecg=x)
                _global_projection = self._global_projector(_global_repr)
                _local_repr = self._local_branch(_intermediate, ecg=x)
                _local_projection = self._local_projector(_local_repr)
            
            # losses  
            update_branch = False
            branch_loss = 0
            if self.global_loss_func is not None:
                g_loss = self.global_loss_func(global_projection, _global_projection)
                branch_loss += g_loss
                epoch_g_loss += g_loss.item()
                update_branch = True

            if self.local_loss_func is not None:
                l_loss = self.local_loss_func(local_projection, _local_projection)
                branch_loss += l_loss
                epoch_l_loss += l_loss.item()
                update_branch = True

            # Rhythm-heartbeat independent loss

            # updating models
            if update_branch:
                self.branch_scheduler.optimizer.zero_grad()
                branch_loss.backward()
                self.branch_scheduler.optimizer.step()

                # updating boostraping models
                self._global_branch.update_parameters(self.global_branch)
                self._global_projector.update_parameters(self.global_projector)

                self._local_branch.update_parameters(self.local_branch)
                self._local_projector.update_parameters(self.local_projector)
        
        epoch_i_loss /= len(data_loader)
        epoch_g_loss /= len(data_loader)
        epoch_l_loss /= len(data_loader)
        epoch_loss = epoch_i_loss + epoch_g_loss + epoch_l_loss
        
        if self.intermediate_loss_func is not None: self.backbone_scheduler.step()
        if update_branch: self.branch_scheduler.step()
        
        return {"train_i_loss": epoch_i_loss, 
                "train_g_loss": epoch_g_loss, 
                "train_l_loss": epoch_l_loss, 
                "train_total_loss": epoch_loss}
    
    def valid_epoch(self, data_loader):
        self.eval()
        epoch_i_loss = 0
        epoch_g_loss = 0
        epoch_l_loss = 0
        
        for x, y in tqdm(data_loader, desc="Validation steps", position=1, leave=False):
            x = x.to(self.device)

            # Generating views
            x_aug = self.augmentation(x) if self.augmentation is not None else x

            # calculating intermediate loss and updating the weights of the encoder                
            with torch.no_grad():
                intermediate = self.encoder(x, mask="all_true")
                _intermediate = self._encoder(x, mask="all_true")

                if self.intermediate_loss_func is not None:
                    i_loss = self.intermediate_loss_func(intermediate, _intermediate)
                    epoch_i_loss += i_loss.item()

                # calculating final losses and updating the weights of the branches
                # target representations
                _global_repr = self._global_branch(_intermediate, ecg=x)
                _global_projection = self._global_projector(_global_repr)
                _local_repr = self._local_branch(_intermediate, ecg=x)
                _local_projection = self._local_projector(_local_repr)

                global_repr = self.global_branch(intermediate, ecg=x)
                global_projection = self.global_projector(global_repr)  # shape = (N, out_channels)
                local_repr = self.local_branch(intermediate, ecg=x)
                local_projection = self.local_projector(local_repr)

                # losses  
                branch_loss = 0
                if self.global_loss_func is not None:
                    g_loss = self.global_loss_func(global_projection, _global_projection)
                    branch_loss += g_loss
                    epoch_g_loss += g_loss.item()

                if self.local_loss_func is not None:
                    l_loss = self.local_loss_func(local_projection, _local_projection)
                    branch_loss += l_loss
                    epoch_l_loss += l_loss.item()

            # Rhythm-heartbeat independent loss

        epoch_i_loss /= len(data_loader)
        epoch_g_loss /= len(data_loader)
        epoch_l_loss /= len(data_loader)
        epoch_loss = epoch_i_loss + epoch_g_loss + epoch_l_loss
        
        return {"valid_i_loss": epoch_i_loss, 
                "valid_g_loss": epoch_g_loss, 
                "valid_l_loss": epoch_l_loss, 
                "valid_total_loss": epoch_loss}
    
    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save(self.state_dict(), fn)
    
    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn, map_location=self.device)
        self.load_state_dict(state_dict)

class SlotFormerPos2(nn.Module):
    def __init__(self, 
                 encoder, encoder_args,
                 global_branch, global_branch_args,
                 local_branch, local_branch_args,  
                 augmentation=False,  # augmentation settings
                 intermediate_loss_func=None, global_loss_func=None, local_loss_func=None, # loss settings
                 intermediate_learning_rate=1e-4, global_learning_rate=1e-4, local_learning_rate=1e-4, epochs=10,  # training parameters
                 warmup_epochs=0,
                 device="cpu",  # other parameters
                ):
        super(SlotFormerPos2, self).__init__()
        
        # encoder parameters
        self.out_channels = encoder_args['output_dims']
        
        # augmentation
        self.augmentation = augmentation
        
        # loss functions
        self.intermediate_loss_func = intermediate_loss_func
        self.global_loss_func = global_loss_func
        self.local_loss_func = local_loss_func
        
        # general parameters
        self.device = device
        
        # training parameters
        self.intermediate_learning_rate = intermediate_learning_rate
        self.local_learning_rate = local_learning_rate
        self.global_learning_rate = global_learning_rate
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        
        # encoder settings
        self.encoder = encoder(**encoder_args)
        self._encoder = torch.optim.swa_utils.AveragedModel(self.encoder)
        self._encoder.update_parameters(self.encoder)
        
        # custom position encoding
        
        # global branch settings
        self.global_branch = global_branch(**global_branch_args)
        self.global_projector = nn.Sequential(nn.Linear(self.out_channels, self.out_channels * 2**2),
                                              nn.BatchNorm1d(self.out_channels * 2**2), 
                                              nn.ReLU(),
                                              nn.Linear(self.out_channels * 2**2, self.out_channels), 
                                             )
        
        self._global_branch = torch.optim.swa_utils.AveragedModel(self.global_branch)
        self._global_branch.update_parameters(self.global_branch)
                                 
        self._global_projector = torch.optim.swa_utils.AveragedModel(self.global_projector)
        self._global_projector.update_parameters(self.global_projector)
        
        # local branch settings
        self.local_branch = local_branch(**local_branch_args)
        self.local_projector = nn.Sequential(nn.Linear(self.out_channels, self.out_channels * 2**2),
                                             nn.BatchNorm1d(local_branch_args['num_slots']), 
                                             nn.ReLU(),
                                             nn.Linear(self.out_channels * 2**2, self.out_channels), 
                                            )

        self._local_branch = torch.optim.swa_utils.AveragedModel(self.local_branch)
        self._local_branch.update_parameters(self.local_branch)
                      
        self._local_projector = torch.optim.swa_utils.AveragedModel(self.local_projector)
        self._local_projector.update_parameters(self.local_projector)
        
        # optimizer
        self.backbone_scheduler, self.branch_scheduler = self._set_optimizer()
        self.to(self.device)
        
    def _set_optimizer(self):
        backbone_optimizer = optim.AdamW([
            {"params": self.encoder.parameters(), "lr": self.intermediate_learning_rate},
        ])
        
        branch_optimizer = optim.AdamW([
            {"params": self.encoder.parameters(), "lr": max(self.local_learning_rate, self.global_learning_rate)},
            {"params": self.global_branch.parameters(), "lr": self.global_learning_rate},
            {"params": self.global_projector.parameters(), "lr": self.global_learning_rate},
            {"params": self.local_branch.parameters(), "lr": self.local_learning_rate},
            {"params": self.local_projector.parameters(), "lr": self.local_learning_rate},
        ])
        
        lr_lambda = lambda e: (e + 1) /  max(self.warmup_epochs, (e + 1))
        backbone_scheduler = optim.lr_scheduler.LambdaLR(backbone_optimizer, lr_lambda=lr_lambda)
        branch_scheduler = optim.lr_scheduler.LambdaLR(branch_optimizer, lr_lambda=lr_lambda)
        
        return backbone_scheduler, branch_scheduler
        
    def forward(self, x):
        # input shape: (N, Cin, T)
        # intermediate shape: (N, Cout, T)
        # global shape: (N, Cout)
        # local shappe: (N, N_S, Cout)
        if self.training:
            intermediate = self.encoder(x)
        else:
            intermediate = self.encoder(x, mask="all_true")

        global_repr = self.global_branch(intermediate, ecg=x)
        local_repr = self.local_branch(intermediate, ecg=x)
            
        return intermediate, global_repr, local_repr
    
    def fit(self, train_loader, valid_loader=None, wandb_logger=None, saved_path=None):  
        self.train()
        
        best_loss = 1e6
        losses = {}
            
        for e in tqdm(range(self.epochs), desc="Epochs", position=0, leave=True):            
            train_loss_dict = self.train_epoch(train_loader)
            valid_loss_dict = self.valid_epoch(valid_loader) if valid_loader is not None else {}
            
            epoch_loss_dict = train_loss_dict | valid_loss_dict
            epoch_lr_dict = {"backbone_lr": self.backbone_scheduler.get_last_lr()[0],
                             "branch_lr": self.branch_scheduler.get_last_lr()[0]}
            
            if wandb_logger is not None:
                wandb.log(epoch_loss_dict)
            
            # print(f"Epoch: {e}, loss: {epoch_loss:.6f} (g_loss: {epoch_g_loss:.6f}, l_loss: {epoch_l_loss:.6f})")
            # print(f"Epoch: {e}, backbone_lr: {backbone_lr:.6f}, branch_lr: {branch_lr:.6f}, warmup_epochs={self.warmup_epochs}")
            
            for k in epoch_loss_dict:
                if k in losses:
                    losses[k].append(epoch_loss_dict[k])
                else:
                    losses[k] = [epoch_loss_dict[k]]
            
            if (saved_path is not None) and (epoch_loss_dict['valid_total_loss'] < best_loss):
                best_loss = epoch_loss_dict['valid_total_loss']
                self.save(saved_path)
        
        return losses
    
    
    def encode(self, dataloader):
        self.eval()
        
        all_g_reprs, all_l_reprs = [[], []]
        with torch.no_grad():
            for x, y in tqdm(dataloader, desc="Repr. encoding: ", leave=False):
                x = x.to(self.device)
                intermediate, global_repr, local_repr = self(x)
                all_g_reprs.append(global_repr.cpu())
                all_l_reprs.append(local_repr.cpu())
        
        if len(all_g_reprs[-1].shape)==1:
            all_g_reprs[-1] = all_g_reprs[-1].unsqueeze(0)

        if len(all_l_reprs[-1].shape)==2:
            all_l_reprs[-1] = all_l_reprs[-1].unsqueeze(0)
        
        all_g_reprs = torch.cat(all_g_reprs, dim=0)
        all_l_reprs = torch.cat(all_l_reprs, dim=0)
        
        return all_g_reprs, all_l_reprs
    
    def train_epoch(self, data_loader): 
        self.train()
        epoch_i_loss = 0
        epoch_g_loss = 0
        epoch_l_loss = 0

        for x, y in tqdm(data_loader, desc="Training steps", position=1, leave=False):
            x = x.to(self.device)
            
            # Generating views 
            x_aug = self.augmentation(x) if self.augmentation is not None else x
            
            if self.intermediate_loss_func is not None:
                # calculating intermediate loss and updating the weights of the encoder                
                intermediate = self.encoder(x_aug)
                with torch.no_grad():
                    _intermediate = self._encoder(x, mask="all_true")
                
                i_loss = self.intermediate_loss_func(intermediate, _intermediate)
                epoch_i_loss += i_loss.item()

                self.backbone_scheduler.optimizer.zero_grad()
                i_loss.backward()
                self.backbone_scheduler.optimizer.step()

                # updating boostraping models
                self._encoder.update_parameters(self.encoder)

            # calculating final losses and updating the weights of the branches
            intermediate = self.encoder(x_aug)
            with torch.no_grad():
                _intermediate = self._encoder(x, mask="all_true")
                
            # losses  
            update_branch = False
            branch_loss = 0
            if self.global_loss_func is not None:
                # generating global representations
                global_repr = self.global_branch(intermediate, ecg=x, agg=False)
                # global_projection = self.global_projector(global_repr)  # shape = (N, out_channels)
                with torch.no_grad():
                    _global_repr = self._global_branch(_intermediate, ecg=x, agg=False)
                    # _global_projection = self._global_projector(_global_repr)
                
                # calculating global loss
                g_loss = self.global_loss_func(global_repr, _global_repr)
                branch_loss += g_loss
                epoch_g_loss += g_loss.item()
                update_branch = True

            if self.local_loss_func is not None:
                # generating local representations
                local_repr = self.local_branch(intermediate, ecg=x, return_k_iter=True)
                if len(local_repr.shape)==4:
                    local_projection = []
                    for i in range(local_repr.shape[0]):
                        local_projection.append(self.local_projector(local_repr[i]))
                    local_projection = torch.stack(local_projection)                        
                else:
                    local_projection = self.local_projector(local_repr)
                
                with torch.no_grad():
                    _local_repr = self._local_branch(_intermediate, ecg=x, return_k_iter=True)
                    if len(_local_repr.shape)==4:
                        _local_projection = []
                        for i in range(_local_repr.shape[0]):
                            _local_projection.append(self._local_projector(_local_repr[i]))
                        _local_projection = torch.stack(_local_projection)    
                    else:
                        _local_projection = self._local_projector(_local_repr)
                    
                # calculating local loss
                if len(local_projection.shape)==4:  # slot.shape = (n_iter, batch_size, n_slot, dim)
                    l_loss = 0
                    k = local_projection.shape[0]
                    for i in range(k): 
                        l_loss += self.local_loss_func(local_projection[i], _local_projection[i])
                    l_loss /= k
                else:
                    l_loss = self.local_loss_func(local_projection, _local_projection)
                
                branch_loss += l_loss
                epoch_l_loss += l_loss.item()
                update_branch = True

            # Rhythm-heartbeat independent loss

            # updating models
            if update_branch:
                self.branch_scheduler.optimizer.zero_grad()
                branch_loss.backward()
                self.branch_scheduler.optimizer.step()

                # updating boostraping models
                self._global_branch.update_parameters(self.global_branch)
                self._global_projector.update_parameters(self.global_projector)

                self._local_branch.update_parameters(self.local_branch)
                self._local_projector.update_parameters(self.local_projector)

        epoch_i_loss /= len(data_loader)
        epoch_g_loss /= len(data_loader)
        epoch_l_loss /= len(data_loader)
        epoch_loss = epoch_i_loss + epoch_g_loss + epoch_l_loss
        
        if self.intermediate_loss_func is not None: self.backbone_scheduler.step()
        if update_branch: self.branch_scheduler.step()
        
        return {"train_i_loss": epoch_i_loss, 
                "train_g_loss": epoch_g_loss, 
                "train_l_loss": epoch_l_loss, 
                "train_total_loss": epoch_loss}
    
    def train_epoch2(self, data_loader): # 2-step training with whole epoch
        self.train()
        epoch_i_loss = 0
        epoch_g_loss = 0
        epoch_l_loss = 0

        for x, y in tqdm(data_loader, desc="Training phase 1", position=1, leave=False):
            x = x.to(self.device)

            # Generating views
            x_aug = self.augmentation(x) if self.augmentation is not None else x

            # calculating intermediate loss and updating the weights of the encoder                
            intermediate = self.encoder(x_aug)
            with torch.no_grad():
                _intermediate = self._encoder(x, mask="all_true")

            if self.intermediate_loss_func is not None:
                i_loss = self.intermediate_loss_func(intermediate, _intermediate)
                epoch_i_loss += i_loss.item()

                self.backbone_scheduler.optimizer.zero_grad()
                i_loss.backward()
                self.backbone_scheduler.optimizer.step()

                # updating boostraping models
                self._encoder.update_parameters(self.encoder)
                
        for x, y in tqdm(data_loader, desc="Training phase 2", position=1, leave=False):
            x = x.to(self.device)
            
            # calculating final losses and updating the weights of the branches
            intermediate = self.encoder(x_aug)
            global_repr = self.global_branch(intermediate, ecg=x)
            global_projection = self.global_projector(global_repr)  # shape = (N, out_channels)
            local_repr = self.local_branch(intermediate, ecg=x)
            local_projection = self.local_projector(local_repr)
            
            # target representations
            with torch.no_grad():
                _intermediate = self._encoder(x, mask="all_true")
                _global_repr = self._global_branch(_intermediate, ecg=x)
                _global_projection = self._global_projector(_global_repr)
                _local_repr = self._local_branch(_intermediate, ecg=x)
                _local_projection = self._local_projector(_local_repr)
            
            # losses  
            update_branch = False
            branch_loss = 0
            if self.global_loss_func is not None:
                g_loss = self.global_loss_func(global_projection, _global_projection)
                branch_loss += g_loss
                epoch_g_loss += g_loss.item()
                update_branch = True

            if self.local_loss_func is not None:
                l_loss = self.local_loss_func(local_projection, _local_projection)
                branch_loss += l_loss
                epoch_l_loss += l_loss.item()
                update_branch = True

            # Rhythm-heartbeat independent loss

            # updating models
            if update_branch:
                self.branch_scheduler.optimizer.zero_grad()
                branch_loss.backward()
                self.branch_scheduler.optimizer.step()

                # updating boostraping models
                self._global_branch.update_parameters(self.global_branch)
                self._global_projector.update_parameters(self.global_projector)

                self._local_branch.update_parameters(self.local_branch)
                self._local_projector.update_parameters(self.local_projector)
        
        epoch_i_loss /= len(data_loader)
        epoch_g_loss /= len(data_loader)
        epoch_l_loss /= len(data_loader)
        epoch_loss = epoch_i_loss + epoch_g_loss + epoch_l_loss
        
        if self.intermediate_loss_func is not None: self.backbone_scheduler.step()
        if update_branch: self.branch_scheduler.step()
        
        return {"train_i_loss": epoch_i_loss, 
                "train_g_loss": epoch_g_loss, 
                "train_l_loss": epoch_l_loss, 
                "train_total_loss": epoch_loss}
    
    def valid_epoch(self, data_loader):
        self.eval()
        epoch_i_loss = 0
        epoch_g_loss = 0
        epoch_l_loss = 0
        
        for x, y in tqdm(data_loader, desc="Validation steps", position=1, leave=False):
            x = x.to(self.device)

            # Generating views
            x_aug = self.augmentation(x) if self.augmentation is not None else x

            # calculating intermediate loss and updating the weights of the encoder                
            with torch.no_grad():
                intermediate = self.encoder(x, mask="all_true")
                _intermediate = self._encoder(x, mask="all_true")

                if self.intermediate_loss_func is not None:
                    i_loss = self.intermediate_loss_func(intermediate, _intermediate)
                    epoch_i_loss += i_loss.item()

                # calculating final losses and updating the weights of the branches
                # target representations
                _global_repr = self._global_branch(_intermediate, ecg=x)
                # _global_projection = self._global_projector(_global_repr)
                _local_repr = self._local_branch(_intermediate, ecg=x)
                _local_projection = self._local_projector(_local_repr)

                global_repr = self.global_branch(intermediate, ecg=x)
                # global_projection = self.global_projector(global_repr)  # shape = (N, out_channels)
                local_repr = self.local_branch(intermediate, ecg=x)
                local_projection = self.local_projector(local_repr)

                # losses  
                branch_loss = 0
                if self.global_loss_func is not None:
                    g_loss = self.global_loss_func(global_repr, _global_repr)
                    branch_loss += g_loss
                    epoch_g_loss += g_loss.item()

                if self.local_loss_func is not None:
                    l_loss = self.local_loss_func(local_projection, _local_projection)
                    branch_loss += l_loss
                    epoch_l_loss += l_loss.item()

            # Rhythm-heartbeat independent loss

        epoch_i_loss /= len(data_loader)
        epoch_g_loss /= len(data_loader)
        epoch_l_loss /= len(data_loader)
        epoch_loss = epoch_i_loss + epoch_g_loss + epoch_l_loss
        
        return {"valid_i_loss": epoch_i_loss, 
                "valid_g_loss": epoch_g_loss, 
                "valid_l_loss": epoch_l_loss, 
                "valid_total_loss": epoch_loss}
    
    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save(self.state_dict(), fn)
    
    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn, map_location=self.device)
        self.load_state_dict(state_dict)