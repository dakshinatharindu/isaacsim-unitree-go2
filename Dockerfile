# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim
FROM nvcr.io/nvidia/isaac-sim:4.5.0 as isaac-sim

# Configure interactive settings and locale for container
# (required for auto-accepting licenses)
ENV DEBIAN_FRONTEND=noninteractive
# ENV DEBCONF_NONINTERACTIVE_SEEN=true
# ENV LANG=C.UTF-8
ENV TZ=UTC

# Source Isaac Sim's internal ROS2 Distro
# ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# ENV LD_LIBRARY_PATH=/isaac-sim/exts/isaacsim.ros2.bridge/humble/lib


# Install any necessary packages for running simulations. This list
# can be appended to by adding a forward slash (\) and then entering the
# package name on the next line. The final line must always NOT have a
# foward slash in it.
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    wget \
    git \
    cmake \
    git-lfs \
    sudo \
    gnupg2 \
    lsb-release \
    curl \
    locales \
    tzdata

# Set up locale
RUN locale-gen en_US en_US.UTF-8
RUN update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8

# Update the working directiory from what the Isaac Sim container sets
# to root
WORKDIR /root


# ROS2 Humble setup
RUN apt install software-properties-common -y
RUN add-apt-repository universe
RUN apt update && apt install curl -y

RUN export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}') && \
    curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb" && \
    dpkg -i /tmp/ros2-apt-source.deb && \
    rm /tmp/ros2-apt-source.deb
    
# Then proceed with ROS2 installation
RUN apt-get update && apt-get install -y --no-install-recommends --allow-downgrades \
        ros-humble-desktop \
        python3-argcomplete \
        ros-dev-tools \
        ros-humble-rmw-cyclonedds-cpp \
        ros-humble-rosidl-generator-dds-idl \
        libfreetype6-dev \
        libbrotli-dev \
        libbrotli1=1.0.9-2build6
    
# Install additional ROS2 tools
RUN apt-get install -y \
        python3-rosdep \
        python3-rosinstall \
        python3-rosinstall-generator \
        python3-wstool \
        build-essential \
        python3-colcon-common-extensions \
        python3-pip
    
# Initialize rosdep
RUN rosdep init || true
RUN rosdep update

# Install Anaconda for package management
RUN wget -q --show-progress --progres=bar:force:noscroll \
    https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh \
    -O /tmp/Anaconda3-2024.10-1-Linux-x86_64.sh

RUN bash /tmp/Anaconda3-2024.10-1-Linux-x86_64.sh -b

RUN rm /tmp/Anaconda3-2024.10-1-Linux-x86_64.sh

RUN ./anaconda3/bin/conda init && \
    ./anaconda3/bin/conda config --set auto_activate_base false

# Install Isaac Lab environment
RUN bash -c "export TERM=linux && \
    source /opt/ros/humble/setup.bash && \
    git clone https://github.com/isaac-sim/IsaacLab.git && \
    cd IsaacLab && \
    git checkout v2.2.0 && \
    ln -s /isaac-sim _isaac_sim && \
    source /root/anaconda3/etc/profile.d/conda.sh && \
    ./isaaclab.sh --conda isaaclab && \
    conda run -n isaaclab ./isaaclab.sh --install"

# # Source ROS2 Humble on login
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

# # Build arguments: you must pass HOST_UID and HOST_GID when building
# ARG HOST_USER
# ARG HOST_UID
# ARG HOST_GID


# # Create the group and user with same UID/GID as host
# RUN groupadd -g ${HOST_GID} ${HOST_USER} && \
#     useradd -m -u ${HOST_UID} -g ${HOST_GID} -s /bin/bash ${HOST_USER}

# # Give passwordless sudo access
# RUN echo "${HOST_USER} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${HOST_USER} && \
#     chmod 0440 /etc/sudoers.d/${HOST_USER}

# # Switch to the created user
# USER ${HOST_USER}
# WORKDIR /home/${HOST_USER}

# RUN wget -q --show-progress --progres=bar:force:noscroll \
#     https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh \
#     -O /tmp/Anaconda3-2024.10-1-Linux-x86_64.sh

# RUN bash /tmp/Anaconda3-2024.10-1-Linux-x86_64.sh -b

# RUN rm /tmp/Anaconda3-2024.10-1-Linux-x86_64.sh

# RUN ./anaconda3/bin/conda init && \
#     ./anaconda3/bin/conda config --set auto_activate_base false

# RUN source /opt/ros/humble/setup.bash




ENTRYPOINT ["/bin/bash", "-l"]