from typing import Any, Optional

import PIL
import torch
import wandb
from src.models.components.sdes import VEReverseSDE
from src.models.components.temperature_schedules import BaseInverseTempSchedule
from src.models.components.utils import sample_from_tensor
from src.models.components.energy_net import EnergyNet


from .dem_module import *


class energyTempModule(DEMLitModule):
    def __init__(
        self,
        lower_temperature: float,
        higher_temperature: float,
        d_temp: float,
        num_eval_samples: int,
        scale_diffusion: bool,
        test_batch_size: int,
        inference_batch_size: int,
        start_resampling_step: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def generate_samples(
        self,
        reverse_sde: VEReverseSDE,
        inverse_temp: float,
        prior, 
        energy_function: BaseEnergyFunction,
        num_samples: int,
        annealing_factor: Optional[float] = 1.0,
        return_full_trajectory: bool = False,
        diffusion_scale=None,
        resampling_interval=None,
        return_logweights=False,
        num_langevin_steps=1,
        batch_size=None,
        num_negative_time_steps=-1,
        start_resampling_step=0,
        return_num_unique_idxs=False,
        resample_at_end=False,
    ) -> torch.Tensor:
        diffusion_scale = diffusion_scale or self.hparams.diffusion_scale
        reverse_sde = reverse_sde or self.reverse_sde

        if num_negative_time_steps == -1:
            num_negative_time_steps = self.hparams.num_negative_time_steps

        prior_samples = prior.sample(num_samples)

        samples, _, num_unique_idxs = self.integrate(
            reverse_sde=reverse_sde,
            samples=prior_samples.clone(),
            reverse_time=True,
            energy_function=energy_function,
            return_full_trajectory=return_full_trajectory,
            diffusion_scale=diffusion_scale,
            resampling_interval=resampling_interval,
            inverse_temperature=inverse_temp,
            annealing_factor=annealing_factor,
            num_langevin_steps=num_langevin_steps,
            batch_size=batch_size,
            num_negative_time_steps=num_negative_time_steps,
            start_resampling_step=start_resampling_step,
        )
        if return_logweights:
            # reintegrate without resampling to get logweights, don't need as many samples
            _, logweights, _ = self.integrate(
                reverse_sde=reverse_sde,
                samples=prior_samples.clone()[:batch_size],
                reverse_time=True,
                energy_function=energy_function,
                return_full_trajectory=return_full_trajectory,
                diffusion_scale=diffusion_scale,
                resampling_interval=self.num_integration_steps + 1,
                inverse_temperature=inverse_temp,
                annealing_factor=annealing_factor,
                num_langevin_steps=1,
                batch_size=batch_size,
                num_negative_time_steps=num_negative_time_steps,
                start_resampling_step=start_resampling_step,
            )
            return samples, logweights, num_unique_idxs

        if return_num_unique_idxs:
            return samples, num_unique_idxs
        return samples

    def integrate(
        self,
        reverse_sde: VEReverseSDE,
        samples: torch.Tensor,
        reverse_time: bool,
        energy_function: BaseEnergyFunction,
        diffusion_scale: float,
        resampling_interval: int,
        inverse_temperature: float,
        annealing_factor: float,
        num_langevin_steps: int,
        num_negative_time_steps: int,
        start_resampling_step: int,
        batch_size=None,
        no_grad=True,
        return_full_trajectory=False,
    ) -> torch.Tensor:
        trajectory, logweights, num_unique_idxs = integrate_sde(
            sde=reverse_sde,
            x1=samples,
            num_integration_steps=self.num_integration_steps,
            energy_function=energy_function,
            start_resampling_step=start_resampling_step,
            reverse_time=reverse_time,
            diffusion_scale=diffusion_scale,
            time_range=1.0,
            resampling_interval=resampling_interval,
            inverse_temperature=inverse_temperature,
            annealing_factor=annealing_factor,
            num_negative_time_steps=num_negative_time_steps,
            num_langevin_steps=num_langevin_steps,
            batch_size=batch_size,
            no_grad=no_grad,
        )
        if return_full_trajectory:
            trajectory, logweights, num_unique_idxs

        return trajectory[-1], logweights, num_unique_idxs
    

    def training_step(self, batch, batch_idx):
        loss = 0.0
        
        x0_samples, _, _ = self.buffer.sample(self.num_samples_to_sample_from_buffer)

        P_mean = -1.2
        P_std = 1.2

        ln_sigmat = torch.randn(len(x0_samples)).to(x0_samples.device) * P_std + P_mean
        ht = torch.exp(2 * ln_sigmat)

        # select inverse_temp from self.inverse_temps randomly
        temp_index = torch.randint(0, len(self.inverse_temperatures) - 1, (1,))
        inverse_temp  = self.inverse_temperatures[temp_index]
        
        sm_loss = self.get_loss(ht, x0_samples, inverse_temp)

        # self.log_dict(
        #     t_stratified_loss(times, dem_loss, loss_name="train/stratified/dem_loss")
        # )
        loss = sm_loss.mean()

        # update and log metrics
        self.dem_train_loss(sm_loss)
        self.log(
            "train/sm_loss",
            self.dem_train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss
    

    def get_loss(self, ht: torch.Tensor, x0: torch.Tensor, inverse_temp) -> torch.Tensor:
        x0.requires_grad = True
        z = torch.randn_like(x0)
        xt = x0 + z * ht[:, None] ** 0.5

        if self.energy_function.is_molecule:
            xt = remove_mean(
                xt,
                self.energy_function.n_particles,
                self.energy_function.n_spatial_dim,
            )

        predicted_scores = -self.net.forward(ht, xt, inverse_temp)
        epsilon = -z

        lambda_t = (ht + 1) / ht
        score_loss = torch.sum(
            (predicted_scores * ht[:, None] ** 0.5 - epsilon) ** 2, dim=(-1)
        )
        score_loss = score_loss.mean()
        return score_loss


    def on_train_epoch_end(self) -> None:
        print("On train epoch end")
        for temp_index, inverse_temp in enumerate(self.inverse_temperatures[:-1]):
            if temp_index == 0:
                self.last_samples[temp_index] = self.generate_samples(
                    reverse_sde=self.reverse_sde,
                    return_logweights=False,
                    prior = self.priors[temp_index],
                    energy_function = self.energy_functions[temp_index],
                    num_samples=self.num_samples_to_generate_per_epoch,
                    batch_size=self.hparams.inference_batch_size,
                    resampling_interval=self.hparams.resampling_interval,
                    inverse_temp = inverse_temp,
                    annealing_factor= 1.0,
                    resample_at_end = False,
                )
                self.last_energies[temp_index] = self.energy_function(self.last_samples[temp_index])
                self.buffers[temp_index].add(self.last_samples[temp_index],
                                                 self.last_energies[temp_index])
                
            inverse_lower_temp = self.inverse_temperatures[temp_index+1]
            "Lightning hook that is called when a training epoch ends."
            self.last_samples[temp_index+1]= self.generate_samples(
                reverse_sde=self.reverse_sde,
                return_logweights=False,
                prior = self.priors[temp_index+1],
                energy_function = self.energy_functions[temp_index+1],
                num_samples=self.num_samples_to_generate_per_epoch,
                batch_size=self.hparams.inference_batch_size,
                resampling_interval=self.hparams.resampling_interval,
                inverse_temp = inverse_temp,
                annealing_factor= (inverse_lower_temp/ inverse_temp),
                resample_at_end = False,
            )
            self.last_energies[temp_index+1] = self.energy_function(self.last_samples[temp_index+1])

            self.buffers[temp_index+1].add(self.last_samples[temp_index+1],
                                                 self.last_energies[temp_index+1])
            
        
        print("On train epoch end end")
    

    def eval_step(self, prefix: str, batch: torch.Tensor, batch_idx: int) -> None:
        """Perform a single eval step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        print("eval step")
        for temp_index, inverse_temp in enumerate(self.inverse_temperatures):
            print("temp_index: ", temp_index)
            energy_function = self.energy_functions[temp_index]

            if prefix == "test":
                true_x0_samples = energy_function.sample_test_set(self.hparams.num_eval_samples)
            elif prefix == "val":
                true_x0_samples = energy_function.sample_val_set(self.hparams.num_eval_samples)

            generated_x0_samples = self.last_samples[temp_index]

            # generate samples noise --> data if needed

            if generated_x0_samples is None or self.hparams.num_eval_samples > len(generated_x0_samples):
                generated_x0_samples = self.generate_samples(
                    reverse_sde=self.reverse_sde,
                    prior=self.priors[temp_index],
                    energy_function=energy_function,
                    num_samples=self.hparams.num_eval_samples,
                    batch_size=self.hparams.inference_batch_size,
                    resampling_interval=self.hparams.resampling_interval,
                    inverse_temp=inverse_temp,
                    resample_at_end=False,
                )

            # sample num_eval_samples from generated samples from dem to match dimenstions
            # required for distribution metrics
            if len(generated_x0_samples) != self.hparams.num_eval_samples:
                indices = torch.randperm(len(generated_x0_samples))[: self.hparams.num_eval_samples]
                generated_x0_samples = generated_x0_samples[indices]
            
            P_mean = -1.2
            P_std = 1.2
        
            ln_sigmat = torch.randn(len(generated_x0_samples)).to(generated_x0_samples.device) * P_std + P_mean
            ht = torch.exp(2 * ln_sigmat)

            with torch.enable_grad():
                loss = self.get_loss(ht, true_x0_samples, inverse_temp).mean(-1)

            # update and log metrics
            loss_metric = self.val_loss if prefix == "val" else self.test_loss
            loss_metric(loss)

            self.log(f"{prefix}/loss", loss_metric, on_step=True, on_epoch=True, prog_bar=True)

            to_log = {
                "data_0": true_x0_samples,
                "gen_0": generated_x0_samples,
            }

            self.eval_step_outputs.append(to_log)
            print("eval step end")
    
    def _log_energy_w2(self, temp_index, prefix="val", test_generated_samples=None):
        energy_function = self.energy_functions[temp_index]
        buffer = self.buffers[temp_index]
        if "test" in prefix:
            data_set = energy_function.sample_test_set(self.hparams.num_eval_samples)
            assert test_generated_samples is not None
            generated_samples = test_generated_samples
            generated_energies = energy_function(generated_samples)
        else:
            if len(self.buffer) < self.hparams.num_eval_samples:
                return
            data_set = energy_function.sample_val_set(self.hparams.num_eval_samples)
            _, generated_energies = buffer.get_last_n_inserted(self.hparams.num_eval_samples)

        energies = energy_function(energy_function.normalize(data_set))
        energy_w2 = pot.emd2_1d(energies.cpu().numpy(), generated_energies.cpu().numpy())

        self.log(
            f"{prefix}/energy_w2",
            self.val_energy_w2(energy_w2),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def _log_dist_w2(self, temp_index, prefix="val", test_generated_samples=None):
        energy_function = self.energy_functions[temp_index]
        buffer = self.buffers[temp_index]
        if "test" in prefix:
            data_set = energy_function.sample_test_set(self.test_batch_size)
            assert test_generated_samples is not None
            generated_samples = test_generated_samples
        else:
            if len(self.buffer) < self.hparams.num_eval_samples:
                return
            data_set = energy_function.sample_val_set(self.hparams.num_eval_samples)
            generated_samples, _ = buffer.get_last_n_inserted(self.hparams.num_eval_samples)

        dist_w2 = pot.emd2_1d(
            energy_function.interatomic_dist(generated_samples).cpu().numpy().reshape(-1),
            energy_function.interatomic_dist(data_set).cpu().numpy().reshape(-1),
        )
        self.log(
            f"{prefix}/dist_w2",
            self.val_dist_w2(dist_w2),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def eval_epoch_end(self, prefix: str):
        if len(self.eval_step_outputs) == 0:
            return
        
        wandb_logger = get_wandb_logger(self.loggers)
        for temp_index, inverse_temp in enumerate(self.inverse_temperatures[:-1]):
            inverse_lower_temp = self.inverse_temperatures[temp_index+1]

            self._log_energy_w2(prefix="val",
                                temp_index=temp_index+1)
            energy_function = self.energy_functions[temp_index]

            if energy_function.is_molecule:
                self._log_dist_w2(prefix="val",
                                  temp_index=temp_index+1)

            # generate samples without annealing for visualization 
            samples, logweights, num_unique_idxs = self.generate_samples(
                reverse_sde=self.reverse_sde,
                prior = self.priors[temp_index],
                energy_function=energy_function,
                num_samples=self.hparams.num_eval_samples,
                return_logweights=True,
                return_num_unique_idxs=True,
                batch_size=self.hparams.inference_batch_size,
                resampling_interval=self.hparams.resampling_interval,
                inverse_temp = inverse_temp,
                annealing_factor = inverse_lower_temp / inverse_temp,
                resample_at_end = False,
            )

            samples_energy = energy_function(samples)

            energy_function.log_on_epoch_end(
                samples,
                samples_energy,
                wandb_logger,
                prefix=fr"val/samples at $\beta$= {inverse_temp}",
            )

            self._log_energy_mean(samples_energy, prefix="val/temp_annealed")

            if self.hparams.resampling_interval != -1:
                self._log_logweights(logweights,
                                     prefix=fr"val / $\beta$= {inverse_temp:0.3f}, $\gamma$= {inverse_lower_temp:0.3f}")
                self._log_std_logweights(logweights,
                                         prefix=fr"val / $\beta$= {inverse_temp:0.3f}, $\gamma$= {inverse_lower_temp:0.3f}")
                self._log_num_unique_idxs(num_unique_idxs,
                                          prefix=fr"val / $\beta$= {inverse_temp:0.3f}, $\gamma$= {inverse_lower_temp:0.3f}")


                if wandb_logger is not None:
                    self.logger.experiment.log(
                        {
                            "1D Array Plot": wandb.plot.line_series(
                                xs=torch.linspace(
                                    1, 0, len(num_unique_idxs)
                                ).tolist(), 
                                ys=[num_unique_idxs],
                                keys=["Number of Unique Indices"],
                                title=fr"val / $\beta$= {inverse_temp:0.3f}, $\gamma$= {inverse_lower_temp:0.3f}",
                                xname="Time", 
                            )
                        }
                    )

    def on_test_epoch_end(self) -> None:
        wandb_logger = get_wandb_logger(self.loggers)

        for temp_index, inverse_temp in enumerate(self.inverse_temperatures[:-1]):
            inverse_lower_temp = self.inverse_temperatures[temp_index + 1]
            final_samples = []

            batch_size = self.hparams.num_eval_samples
            n_batches = self.num_samples_to_save // batch_size
            print(f"Generating {n_batches} batches of annealed samples of size {batch_size}.")
            print(f"Resampling interval is {self.hparams.resampling_interval}")

            for i in range(n_batches):
                start = time.time()
                samples = self.generate_samples(
                    reverse_sde=self.reverse_sde,
                    prior = self.priors[temp_index + 1],
                    energy_function=self.energy_functions[temp_index + 1],
                    num_samples=self.hparams.num_samples_to_save,
                    batch_size=self.hparams.inference_batch_size,
                    resampling_interval=self.hparams.resampling_interval,
                    inverse_temp=inverse_temp,
                    annealing_factor=inverse_lower_temp / inverse_temp,
                    resample_at_end=False,
                )
                final_samples.append(samples)
                end = time.time()
                print(f"batch {i} took {end - start:0.2f}s")

            final_samples = torch.cat(final_samples, dim=0)
            output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            # append time to avoid overwriting
            path = f"{output_dir}/samples_temperature_{inverse_lower_temp}_{self.num_samples_to_save}.pt"
            torch.save(final_samples, path)
            print(f"Saving samples to {path}")

            # compute metrics on a subset of the generated samples
            batch_generated_samples = sample_from_tensor(
                final_samples, self.hparams.test_batch_size
            )

            # log energy w2
            self._log_energy_w2(inverse_temp, prefix="test", test_generated_samples=batch_generated_samples)
            self._log_dist_w2(inverse_temp, prefix="test", test_generated_samples=batch_generated_samples)


    def _log_logweights(self, logweights, prefix="val"):
        wandb_logger = get_wandb_logger(self.loggers)
        if wandb_logger is None:
            return
        fig, axs = plt.subplots(1, 1, figsize=(8, 4))
        # sample a few at random from the logweights
        idx = torch.randint(0, logweights.shape[1], (15,))
        logweights = logweights[:, idx].cpu().numpy()
        integration_times = torch.linspace(1, 0, logweights.shape[0])
        axs.plot(integration_times, logweights)

        # limit yaxis
        axs.set_xlabel("Integration time")
        fig.canvas.draw()
        img = PIL.Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        wandb_logger.log_image(f"{prefix}/annealing_logweights", [img])

    def _log_std_logweights(self, logweights, prefix="val"):
        wandb_logger = get_wandb_logger(self.loggers)
        if wandb_logger is None:
            return

        fig, axs = plt.subplots(1, 1, figsize=(8, 4))
        std_logweights = logweights.std(dim=1).cpu().numpy()
        integration_times = torch.linspace(1, 0, std_logweights.shape[0])
        axs.plot(integration_times, std_logweights)
        axs.set_xlabel("Integration time")
        fig.canvas.draw()
        img = PIL.Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        wandb_logger.log_image(f"{prefix}/std_logweights", [img])

    def _log_num_unique_idxs(self, num_unique_idxs, prefix="val"):
        wandb_logger = get_wandb_logger(self.loggers)
        if wandb_logger is None:
            return
        fig, axs = plt.subplots(1, 1, figsize=(8, 4))
        integration_times = torch.linspace(1, 0, len(num_unique_idxs))
        axs.plot(integration_times, num_unique_idxs)
        axs.set_xlabel("Integration time")
        fig.canvas.draw()
        img = PIL.Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        wandb_logger.log_image(f"{prefix}/num_unique_idxs", [img])

    def setup(self, stage: str) -> None:
        super().setup(stage)
        self.energy_functions = {}
        self.priors = {}
        self.buffers = {}
        self.last_samples = {}
        self.last_energies = {}

        num_temps = int((self.hparams.higher_temperature - self.hparams.lower_temperature) / self.hparams.d_temp) + 1

        temperatures = torch.linspace(self.hparams.lower_temperature,
                                        self.hparams.higher_temperature,
                                        num_temps)

        print("Temperatures: ", temperatures)
        self.inverse_temperatures = torch.flip(torch.round(self.hparams.higher_temperature / temperatures,
                                                           decimals=2).to(self.device), dims=(0,))


        print("Inverse Temperatures: ", self.inverse_temperatures)



        self.score_net = self.hparams.net()
        self.net = EnergyNet(score_net=self.score_net)

        self.reverse_sde = VEReverseSDE(energy_net=self.net,
                                        noise_schedule=self.hparams.noise_schedule,
                                        score_net=None,
                                        pin_energy=True
                                        )
        
        for temp_index, inverse_temp in enumerate(self.inverse_temperatures):
            self.energy_functions[temp_index] = self.hparams.energy_function(device=self.device,
                                                                               temperature=1/inverse_temp)
            self.priors[temp_index] = self.partial_prior(device=self.device,
                                                           scale=(self.noise_schedule.h(1) / inverse_temp) ** 0.5)
            self.buffers[temp_index] = self.partial_buffer(device=self.device)
            self.last_samples[temp_index] = None
            self.last_energies[temp_index] = None


            if self.init_from_prior:
                init_states = self.priors[temp_index].sample(self.num_init_samples)

            else:
                init_states = self.energy_functions[0].sample(self.num_init_samples)
            init_energies = self.energy_functions[temp_index](init_states)
            self.buffers[temp_index].add(init_states, init_energies)




if __name__ == "__main__":
    _ = DEMLitModule(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
