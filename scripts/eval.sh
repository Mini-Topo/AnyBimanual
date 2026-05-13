# this script is for evaluating a given checkpoint.
#       bash scripts/eval.sh PERACT_BC  0 ${exp_name}


# some params specified by user
method=$1
# set the seed number
seed="0"
# set the gpu id for evaluation. we use one gpu for parallel evaluation.
eval_gpu=${2:-"0"}

test_demo_path="/home/tappei-m/Project/AnyBimanual_data/rlbench2"
logdir="/home/tappei-m/Project/AnyBimanual_logs"

addition_info="$(date +%Y%m%d)"
exp_name=${3:-"${method}_${addition_info}"}

starttime=`date +'%Y-%m-%d %H:%M:%S'`

camera=True
gripper_mode='BimanualDiscrete'
arm_action_mode='BimanualEndEffectorPoseViaPlanning'
action_mode='BimanualMoveArmThenGripper'
tasks=[bimanual_pick_laptop]
eval_type="all"
eval_episodes=1

echo "I am going to kill the session ${exp_name}_${eval_type}, are you sure? (5s)"
sleep 2s
tmux kill-session -t ${exp_name}_${eval_type}
sleep 3s
echo "start new tmux session: ${exp_name}_${eval_type}, running eval.py"
tmux new-session -d -s ${exp_name}_${eval_type}

tmux select-pane -t 0 
tmux send-keys "conda activate rlbench; 
CUDA_VISIBLE_DEVICES=${eval_gpu} xvfb-run -a python eval.py method=$method \
    rlbench.task_name=${exp_name} \
    rlbench.demo_path=${test_demo_path} \
    framework.logdir=${logdir} \
    framework.start_seed=${seed} \
    cinematic_recorder.enabled=${camera} \
    rlbench.gripper_mode=${gripper_mode} \
    rlbench.arm_action_mode=${arm_action_mode} \
    rlbench.action_mode=${action_mode} \
    rlbench.tasks=${tasks}  \
    framework.eval_type=${eval_type} \
    framework.eval_episodes=${eval_episodes}
"
tmux -2 attach-session -t ${exp_name}_${eval_type}