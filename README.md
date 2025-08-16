# Unitree Go-2 Robot Simulation in IsaacSSim

## Getting Started
1. Build the container:
   ```
   cd simulator
   apptainer build container.sif container.def
   ```
2. Run the container:
   ```
   apptainer exec container.sif bash
   ```
3. IsaacLab installation:
   ```
   git clone https://github.com/isaac-sim/IsaacLab.git
   cd IsaacLab
   ln -s $(realpath ../IsaacSim) _isaac_sim
   ./isaaclab.sh --conda isaaclab
   conda activate isaaclab
   ./isaaclab.sh --install