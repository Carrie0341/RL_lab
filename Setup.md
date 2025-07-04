# IsaacLab Setup

```bash
# 下載並安裝 Isaac Sim

wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone%404.5.0-rc.36%2Brelease.19112.f59b3005.gl.linux-x86_64.release.zip

export ISAACSIM_PATH="${HOME}/isaacsim"
mkdir -p ${ISAACSIM_PATH}
unzip isaac-sim-standalone*.zip -d ${ISAACSIM_PATH}
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"

# 安裝必要的系統套件
sudo apt install cmake build-essential

# 下載IsaacLab
cd ~
git clone https://github.com/isaac-sim/IsaacLab.git
cd ~/IsaacLab
ln -s ${HOME}/isaacsim _isaac_sim

# 安裝 IsaacLab 及環境
./isaaclab.sh -c
conda activate env_isaaclab
./isaaclab.sh -i
./isaaclab.sh -i rl_games
```

# RL_lab Setup

```bash
git clone https://github.com/Carrie0341/RL_lab
conda activate env_isaaclab
python -m pip install -e source/RL_Lab
```
