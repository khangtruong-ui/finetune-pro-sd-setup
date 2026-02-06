# training/trainer.py
import jax
import jax.numpy as jnp
import math
from tqdm.auto import tqdm
import logging

from diffusers import FlaxDDPMScheduler
from flax.training import train_state
import optax

from models.components import load_models
from data.dataset import get_dataloader
from utils.sharding import distribute_device, sharding, no_sharding
from utils.checkpoint import save_checkpoint
from training.train_step import create_train_step

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config):
        self.config = config
        
        # Weight dtype
        self.weight_dtype = {
            "no": jnp.float32,
            "fp16": jnp.float16,
            "bf16": jnp.bfloat16,
        }[config.mixed_precision]

        # Load models
        (self.tokenizer, self.text_encoder, self.vae,
         self.vae_params, self.unet, self.unet_params) = load_models(config)

        # Shard UNet params (main trainable part)
        self.unet_params = distribute_device(self.unet_params, no_sharding, replicate=True)

        # Replicate frozen components
        self.vae_params = distribute_device(self.vae_params, no_sharding, replicate=True)
        self.text_encoder_params = self.text_encoder.params = distribute_device(self.text_encoder.params, no_sharding, replicate=True)

        # Optimizer & TrainState
        total_batch_size = config.train_batch_size * jax.device_count()
        lr = config.learning_rate * total_batch_size if config.scale_lr else config.learning_rate

        optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adamw(
                learning_rate=lr,
                b1=0.9,
                b2=0.999,
                weight_decay=0.01,
            ),
        )

        self.state = train_state.TrainState.create(
            apply_fn=self.unet.apply,
            params=self.unet_params,
            tx=optimizer,
        )

        # Noise scheduler
        self.noise_scheduler = FlaxDDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )

        # Create compiled train step
        self.train_step = create_train_step(self.unet, self.vae, self.noise_scheduler)

        # Data loader
        loader_length = get_dataloader(config, self.tokenizer)
        self.dataloader = loader_length['loader']
        # Compute steps
        self.steps_per_epoch = loader_length['length']
        self.max_steps = config.max_train_steps or (config.num_train_epochs * self.steps_per_epoch)
        self._compiled_precompute = None

    def precompute_text_embeddings(self, batch):
        """Precompute text embeddings (text encoder is frozen)"""
        input_ids = batch["input_ids"]
        
        @jax.jit
        def _compiled_precompute(input_ids):
            return self.text_encoder(
                input_ids,
                params=self.text_encoder_params,
                train=False,
            )[0]

        self._compiled_precompute = self._compiled_precompute or _compiled_precompute
        encoder_hidden_states = self._compiled_precompute(input_ids)
        return encoder_hidden_states

    def get_init_epoch(self):
        return 0

    def train(self):
        logger.info("***** Starting Training *****")
        logger.info(f"  Num examples: ~{self.steps_per_epoch * self.config.train_batch_size * jax.device_count()}")
        logger.info(f"  Steps per epoch: {self.steps_per_epoch}")
        logger.info(f"  Total steps: {self.max_steps}")

        rng = jax.random.PRNGKey(self.config.seed)
        global_step = 0

        init_epoch = self.get_init_epoch()
        loader = iter(self.dataloader)
        with open('progress.log', 'w') as f:
            for epoch in range(init_epoch, self.config.num_train_epochs):
                progress_bar = tqdm(
                    zip(range(self.steps_per_epoch), loader),
                    total=self.steps_per_epoch,
                    desc=f"Epoch {epoch + 1}/{self.config.num_train_epochs}",
                    file=f,
                    leave=False
                )
                
                for step, batch in progress_bar:
                    # Move batch to devices and shard data
                    with open('ready.log', 'w') as f:
                        f.write('Here')
                    batch = distribute_device(batch, sharding)
    
                    # Precompute text embeddings (frozen text encoder)
                    print('Precompute embedding:')
                    encoder_hidden_states = self.precompute_text_embeddings(batch)
                    batch["encoder_hidden_states"] = encoder_hidden_states
    
                    # Train step
                    rng, dropout_rng = jax.random.split(rng)
                    print('Train step:')
                    self.state, loss, rng = self.train_step(
                        self.state,
                        self.vae_params,
                        self.text_encoder_params,
                        batch,
                        dropout_rng,
                    )
    
                    global_step += 1
                    progress_bar.set_postfix({"loss": f"{loss:.4f}"})
    
                    if global_step >= self.max_steps:
                        break
                # Checkpoint every 50 epochs or at the end
                if (epoch + 1) % 50 == 0 or (epoch + 1) == self.config.num_train_epochs:
                    save_checkpoint(
                        self.config,
                        epoch,
                        self.state,
                        self.text_encoder_params,
                        self.vae_params,
                        self.tokenizer,
                        self.text_encoder,
                        self.vae,
                        self.unet,
                    )
    
                if global_step >= self.max_steps:
                    break

        logger.info("Training completed!")
