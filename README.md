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

3. IsaacSim installation:
   Download the IsaacSim 4.5 binary and extract in `simulator` directory.
   ```
   wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone%404.5.0-rc.36%2Brelease.19112.f59b3005.gl.linux-x86_64.release.zip
   unzip isaac-sim-standalone@4.5.0-rc.36+release.19112.f59b3005.gl.linux-x86_64.release.zip -d /IsaacSim
   ```
4. IsaacLab installation:
   ```
   git clone https://github.com/isaac-sim/IsaacLab.git
   cd IsaacLab
   ln -s $(realpath ../IsaacSim) _isaac_sim
   ./isaaclab.sh --conda isaaclab
   conda activate isaaclab
   ./isaaclab.sh --install

5. Launch the simulation (Make sure to launch from apptainer instance):
   ```
   cd src
   python isaac_go2_ros2.py
   ```