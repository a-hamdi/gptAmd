def train_model(
    config: GPT2Config,
    train_dataset: torch.utils.data.Dataset,
    output_dir: str,
    tokenizer=None,
    num_gpus: int = torch.cuda.device_count(),
    **training_args
):
    """Main training function."""
    # Ensure CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your PyTorch installation.")
    
    device = torch.device("cuda:0")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    
    training_config = {
        'output_dir': output_dir,
        'tokenizer': tokenizer,
        'num_epochs': training_args.get('num_epochs', 3),
        'batch_size': training_args.get('batch_size', 8),
        'learning_rate': training_args.get('learning_rate', 5e-5),
        'weight_decay': training_args.get('weight_decay', 0.01),
        'max_grad_norm': training_args.get('max_grad_norm', 1.0),
        'use_wandb': training_args.get('use_wandb', False),
        'wandb_project': training_args.get('wandb_project', 'gpt2-implementation'),
        'steps_per_epoch': training_args.get('steps_per_epoch', 1000)
    }
    
    model = GPT2(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    if num_gpus > 1:
        mp.spawn(
            train_distributed,
            args=(num_gpus, model, train_dataset, training_config),
            nprocs=num_gpus,
            join=True
        )
    else:
        model = model.to(device)
        print("Model moved to GPU")
        
        trainer = GPT2Trainer(
            model=model,
            train_dataset=train_dataset,
            config=training_config,
            device=device
        )
        
        trainer.train() 

class GPT2Trainer:
    """Trainer class for GPT-2 model."""
    
    def __init__(
        self,
        model: GPT2,
        train_dataset: torch.utils.data.Dataset,
        config: Dict[str, Any],
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
    ):
        print(f"\nInitializing trainer on device: {device}")
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size
        
        # Setup distributed training if needed
        if self.world_size > 1:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=True
            )
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Initialize learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('steps_per_epoch', 1000) * config['num_epochs'],
            eta_min=config['learning_rate'] * 0.1
        )
        
        # Initialize mixed precision training
        self.use_amp = torch.cuda.is_available()
        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.scaler = amp.GradScaler(enabled=self.use_amp)
        
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Using mixed precision: {self.use_amp}")
        
        # Initialize wandb if main process
        if self.rank == 0 and config.get('use_wandb', False):
            wandb.init(
                project=config.get('wandb_project', 'gpt2-implementation'),
                config=config
            )
        
        # Add tokenizer to the trainer
        self.tokenizer = config.get('tokenizer', None)

    def train(self):
        """Main training loop."""
        print("\nStarting training loop...")
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'] // self.world_size,
            num_workers=0,  # Must be 0 for IterableDataset
            pin_memory=True
        )
        
        best_loss = float('inf')
        steps_per_epoch = self.config.get('steps_per_epoch', 1000)
        
        for epoch in range(self.config['num_epochs']):
            self.model.train()
            total_loss = 0
            step = 0
            
            if self.rank == 0:
                pbar = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch + 1}/{self.config['num_epochs']}")
            
            for batch in train_loader:
                # Move batch to device and verify
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                if step == 0:  # Print debug info for first batch
                    print(f"\nFirst batch shapes - Input: {input_ids.shape}, Labels: {labels.shape}")
                    print(f"Input device: {input_ids.device}, Labels device: {labels.device}")
                    print(f"Model device: {next(self.model.parameters()).device}")
                
                # Forward pass with mixed precision
                with amp.autocast(device_type=self.device_type, enabled=self.use_amp):
                    logits = self.model(input_ids)
                    loss = nn.CrossEntropyLoss()(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1)
                    )
                
                # Backward pass with gradient scaling
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get('max_grad_norm', 1.0)
                )
                
                # Update weights with gradient scaling
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Update learning rate
                self.scheduler.step()
                
                # Update progress
                total_loss += loss.item()
                step += 1
                
                if self.rank == 0:
                    pbar.update(1)
                    pbar.set_postfix({'loss': loss.item()})
                    
                    if self.config.get('use_wandb', False):
                        wandb.log({
                            'loss': loss.item(),
                            'learning_rate': self.scheduler.get_last_lr()[0]
                        })
                
                # Break if we've reached steps_per_epoch
                if step >= steps_per_epoch:
                    break
            
            # Calculate average loss for epoch
            avg_loss = total_loss / step
            
            if self.rank == 0:
                pbar.close()
                print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")
                
                # Save checkpoint if best loss
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self.save_checkpoint(
                        Path(self.config['output_dir']) / 'best_model.pt',
                        epoch,
                        best_loss
                    ) 