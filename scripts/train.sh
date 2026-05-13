# example to run:
#       bash train.sh PERACT_BC 0,1 12345 ${exp_name}

# set the method name
method=${1} # PERACT_BC or BIMANUAL_PERACT

# set the seed number
seed="0"

train_gpu=${2:-"0,1"}
train_gpu_list=(${train_gpu//,/ })

# set the port for ddp training.
port=${3:-"12345"}
# you could enable/disable wandb by this.
use_wandb=False

# cur_dir=$(pwd)
train_demo_path="/home/tappei-m/Project/AnyBimanual_data/rlbench2"

# we set experiment name as method+date. you could specify it as you like.
addition_info="$(date +%Y%m%d)"
exp_name=${4:-"${method}_${addition_info}"}
logdir="/home/tappei-m/Project/AnyBimanual_logs"

# create a tmux window for training
echo "I am going to kill the session ${exp_name}, are you sure? (5s)"
sleep 1s
tmux kill-session -t ${exp_name}
sleep 3s
echo "start new tmux session: ${exp_name}, running main.py"
tmux new-session -d -s ${exp_name}

#######
# override hyper-params in config.yaml
#######
batch_size=1
anybimanual=True
augmentation_type="ab" # "standard"
tasks=[bimanual_pick_laptop]
demo=20
episode_length=25
save_freq=10
log_freq=1
task_folder="multi"
replay_path="/home/tappei-m/Project/AnyBimanual_replay"
training_iterations=5
num_workers=0
transitions_before_train=1
wandb_project="AnyBimanual"

tmux select-pane -t 0 
tmux send-keys "conda activate rlbench; 
CUDA_VISIBLE_DEVICES=${train_gpu} python train.py method=$method \
        rlbench.task_name=${exp_name} \
        framework.logdir=${logdir} \
        rlbench.demo_path=${train_demo_path} \
        framework.start_seed=${seed} \
        framework.use_wandb=${use_wandb} \
        framework.wandb_group=${exp_name} \
        framework.wandb_name=${exp_name} \
        ddp.num_devices=${#train_gpu_list[@]} \
        replay.batch_size=${batch_size} \
        ddp.master_port=${port} \
        rlbench.tasks=${tasks} \
        rlbench.demos=${demo} \
        rlbench.episode_length=${episode_length} \
        framework.save_freq=${save_freq} \
        framework.wandb_project=${wandb_project} \
        replay.path=${replay_path} \
        replay.task_folder=${task_folder} \
        framework.log_freq=${log_freq} \
        framework.training_iterations=${training_iterations} \
        framework.num_workers=${num_workers} \
        framework.transitions_before_train=${transitions_before_train} \
        framework.frozen=${frozen} \
        framework.anybimanual=${anybimanual} \
        framework.augmentation_type=${augmentation_type}
"
# remove 0.ckpt
# rm -rf logs/${exp_name}/seed${seed}/weights/0

tmux -2 attach-session -t ${exp_name}