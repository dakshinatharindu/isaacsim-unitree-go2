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
    sudo

# Add a symlink to /root to easily get to Isaac Sim examples
RUN ln -s /isaac-sim /root/isaac-sim

# Setup Git and clone Habitat Sim
RUN git lfs install

# Update the working directiory from what the Isaac Sim container sets
# to root
WORKDIR /root

# Install and configure Anaconda


# Install ROS2
RUN add-apt-repository universe
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# RUN add-apt-repository universe
# RUN apt update && apt install curl -y

# RUN export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
# RUN curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb"
# RUN dpkg -i /tmp/ros2-apt-source.deb
# RUN rm /tmp/ros2-apt-source.deb
# RUN apt update

RUN apt-get update && apt-get install -y --no-install-recommends --allow-downgrades \
    ros-humble-desktop \
    python3-argcomplete \
    ros-dev-tools \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-rosidl-generator-dds-idl \
    # ros-humble-rviz2 \
    libfreetype6-dev \
    libbrotli-dev \
    libbrotli1=1.0.9-2build6

# RUN apt-get install -y \
#         python3-rosdep \
#         python3-rosinstall \
#         python3-rosinstall-generator \
#         python3-wstool \
#         build-essential \
#         python3-colcon-common-extensions \
#         python3-pip


# Source ROS2 Humble on login
RUN echo "source /opt/ros/humble/setup.bash" >> /etc/bash.bashrc

# Build arguments: you must pass HOST_UID and HOST_GID when building
ARG HOST_USER
ARG HOST_UID
ARG HOST_GID


# Create the group and user with same UID/GID as host
RUN groupadd -g ${HOST_GID} ${HOST_USER} && \
    useradd -m -u ${HOST_UID} -g ${HOST_GID} -s /bin/bash ${HOST_USER}

# Give passwordless sudo access
RUN echo "${HOST_USER} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${HOST_USER} && \
    chmod 0440 /etc/sudoers.d/${HOST_USER}

# Switch to the created user
USER ${HOST_USER}
WORKDIR /home/${HOST_USER}

RUN wget -q --show-progress --progres=bar:force:noscroll \
    https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh \
    -O /tmp/Anaconda3-2024.10-1-Linux-x86_64.sh

RUN bash /tmp/Anaconda3-2024.10-1-Linux-x86_64.sh -b

RUN rm /tmp/Anaconda3-2024.10-1-Linux-x86_64.sh

RUN ./anaconda3/bin/conda init && \
    ./anaconda3/bin/conda config --set auto_activate_base false

# RUN source /opt/ros/humble/setup.bash

# # Install Isaac Lab environment
# RUN git clone https://github.com/isaac-sim/IsaacLab.git
# RUN cd IsaacLab
# RUN git checkout v2.2.0
# RUN ln -s /isaac-sim _isaac_sim
# RUN ./isaaclab.sh --conda isaaclab


ENTRYPOINT ["/bin/bash", "-l"]