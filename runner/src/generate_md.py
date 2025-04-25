"""

Script to generate MD data as in timewarp.

Benchmarking on Mila Cluster March 10 2025

on AL6 (uncapped)
A100: 0.99 s/it
L40s: 0.59 s/it
RTX8000: 0.66 s/it
CPU (cn-h001):
    1x: 8.05 s/it
    2x: 5.95 s/it
    4x: 6.22 s/it
    8x: 6.40 s/it
    16x: 8.00 s/it
    32x: 18.49 s/it
"""

import logging
import os
from sys import stdout
from typing import Optional

import hydra
import mdtraj as md
import numpy as np
import openmm
import tqdm
from omegaconf import DictConfig
from openmm import Platform, unit
from openmm.app import ForceField, Simulation, StateDataReporter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="md.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    pdb_path = os.path.join(cfg.pdb_dir, (cfg.pdb_filename + ".pdb"))
    topology = md.load_topology(pdb_path).to_openmm()
    positions = md.load_pdb(pdb_path).xyz[0]
    platform_properties = {}
    if hasattr(cfg, "platform_properties"):
        platform_properties = cfg.platform_properties

    forcefield = ForceField("amber14-all.xml", "implicit/obc1.xml")
    system = forcefield.createSystem(
        topology,
        nonbondedMethod=openmm.app.CutoffNonPeriodic,
        nonbondedCutoff=2.0 * openmm.unit.nanometer,
        constraints=None,
    )
    integrator = openmm.LangevinMiddleIntegrator(
        cfg.temperature * openmm.unit.kelvin,
        0.3 / openmm.unit.picosecond,
        1.0 * openmm.unit.femtosecond,
    )
    logger.info(f"Platform name: {cfg.platform_name} {platform_properties}")
    logger.info(f"Temperature: {cfg.temperature}")
    logger.info(f"Running {pdb_path} for {cfg.num_steps} steps")
    platform = Platform.getPlatform(cfg.platform_name)
    simulation = Simulation(
        topology,
        system,
        integrator,
        platform=platform,
        platformProperties=platform_properties,
    )
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy()
    simulation.reporters.append(
        StateDataReporter(
            os.path.join(cfg.output_dir, "output.txt"),
            cfg.log_freq,
            step=True,
            potentialEnergy=True,
            temperature=True,
        )
    )
    logger.info("Minimized. Running simulation...")
    simulation.step(cfg.warmup_steps)
    all_positions = []
    for step in tqdm.tqdm(range(cfg.num_steps)):
        simulation.step(cfg.step_size)
        st = simulation.context.getState(getPositions=True)
        coords = st.getPositions(asNumpy=True) / unit.nanometer
        all_positions.append(coords)
    all_positions = np.array(all_positions, dtype=np.float32)
    save_path = os.path.join(cfg.output_dir, cfg.output_filename)
    logger.info(f"saving to {save_path} with shape {all_positions.shape}")
    os.makedirs(cfg.output_dir, exist_ok=True)
    np.savez_compressed(save_path, all_positions=all_positions)


if __name__ == "__main__":
    main()
