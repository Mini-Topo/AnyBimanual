````md
# AnyBimanual 実行時の修正メモ（RTX 4060 Ti 16GB）

## 環境

- Ubuntu 22.04
- RTX 4060 Ti 16GB
- CUDA 12.2
- Python 3.8
- conda env: `rlbench`

---

# 追加で必要だった pip install

```bash
pip install ftfy
pip install regex
pip install blosc
pip install transformers
pip install plotly
pip install wandb==0.14.0
````

---

# dataset 配置

```text
/home/tappei-m/Project/AnyBimanual_data/rlbench2
```

例：

```text
rlbench2/
├── bimanual_pick_laptop.train.squashfs
└── bimanual_pick_laptop_train/
```

---

# checkpoint 配置

```text
/home/tappei-m/Project/AnyBimanual_checkpoints/PERACT_BC/
```

配置したファイル：

```text
checkpoint_peract_bc_leader_layer_0.pt
checkpoint_peract_bc_follower_layer_0.pt
```

---

# offline_train_runner.py 修正

ファイル：

```text
third_party/YARR/yarr/runners/offline_train_runner.py
```

修正：

```python
file_path = f"/home/tappei-m/Project/AnyBimanual_checkpoints/{method.name}/"
```

---

# batch_size=1 対応（OOM回避）

RTX 4060 Ti 16GB では `batch_size=2` で CUDA OOM が発生したため、`batch_size=1` で実行。

ただし checkpoint 側は batch_size=2 前提の voxelizer buffer を持っているため、追加修正が必要。

---

# voxelizer reshape 修正

ファイル：

```text
agents/peract_bc/qattention_peract_bc_agent.py
```

`load_weights()` 内。

元コード：

```python
if not self._training:
```

を削除し、以下を training 時にも適用：

```python
b = merged_state_dict["_voxelizer._ones_max_coords"].shape[0]

merged_state_dict["_voxelizer._ones_max_coords"] = merged_state_dict[
    "_voxelizer._ones_max_coords"
][0:1]

flat_shape = merged_state_dict["_voxelizer._flat_output"].shape[0]

merged_state_dict["_voxelizer._flat_output"] = merged_state_dict[
    "_voxelizer._flat_output"
][0 : flat_shape // b]

merged_state_dict["_voxelizer._tiled_batch_indices"] = merged_state_dict[
    "_voxelizer._tiled_batch_indices"
][0:1]

merged_state_dict["_voxelizer._index_grid"] = merged_state_dict[
    "_voxelizer._index_grid"
][0:1]
```

---

# cfg が agent に渡っていない問題

ファイル：

```text
agents/peract_bc/launch_utils.py
```

修正前：

```python
anybimanual=cfg.framework.anybimanual,
```

修正後：

```python
anybimanual=cfg.framework.anybimanual,
cfg=cfg,
```

---

# wandb 無効化時の修正

ファイル：

```text
agents/peract_bc/qattention_peract_bc_agent.py
```

修正前：

```python
if self.cfg.framework.use_wandb:
```

修正後：

```python
if wandb.run is not None:
```

また、

```python
wandb.log(...)
```

も同様に `wandb.run is not None` 条件付きに変更。

---

# replay cache 初期化

実行前に replay cache を削除：

```bash
rm -rf /home/tappei-m/Project/AnyBimanual_replay/*
```

---

# 実行コマンド（debug）

```bash
CUDA_VISIBLE_DEVICES=0 python train.py method=PERACT_BC   rlbench.task_name=debug_test   framework.logdir=/home/tappei-m/Project/AnyBimanual_logs   rlbench.demo_path=/home/tappei-m/Project/AnyBimanual_data/rlbench2   framework.start_seed=0   framework.use_wandb=False   framework.wandb_group=debug_test   framework.wandb_name=debug_test   ddp.num_devices=1   replay.batch_size=1   ddp.master_port=12345   rlbench.tasks=[bimanual_pick_laptop]   rlbench.demos=20   rlbench.episode_length=25   framework.save_freq=10   framework.wandb_project=AnyBimanual   replay.path=/home/tappei-m/Project/AnyBimanual_replay   replay.task_folder=multi   framework.log_freq=1   framework.training_iterations=5 framework.num_workers=0  framework.transitions_before_train=1  framework.frozen=   framework.anybimanual=True   framework.augmentation_type=ab   2>&1 | tee debug_train.log
```

---

# 最終確認

以下を確認：

* replay generation 成功
* checkpoint loading 成功
* train loop 実行
* loss 出力
* reconstruction image 保存
* `Stopping envs ...` で正常終了

今回の環境では、`PERACT_BC` の debug training が正常動作することを確認。

```
```

# Evaluation 実行まとめ

## 実行目的

AnyBimanual / PERACT_BC の評価パイプラインが、
RLBench + CoppeliaSim 上で正常動作するか確認。

---

# 実行環境

* Ubuntu 22.04
* RTX 4060 Ti 16GB
* CUDA 12.2
* headless mode (`xvfb-run`)
* RLBench dual panda environment

---

# 実行コマンド

```bash
CUDA_VISIBLE_DEVICES=0 xvfb-run -a python eval.py \
  method=PERACT_BC \
  rlbench.task_name=debug_test \
  framework.logdir=/home/tappei-m/Project/AnyBimanual_logs \
  rlbench.demo_path=/home/tappei-m/Project/AnyBimanual_data/rlbench2 \
  framework.start_seed=0 \
  cinematic_recorder.enabled=True \
  rlbench.gripper_mode=BimanualDiscrete \
  rlbench.arm_action_mode=BimanualEndEffectorPoseViaPlanning \
  rlbench.action_mode=BimanualMoveArmThenGripper \
  rlbench.tasks=[bimanual_pick_laptop] \
  framework.eval_type=all \
  framework.eval_episodes=1
```

---

# 確認できたこと

## 1. RLBench / CoppeliaSim 起動成功

以下を確認：

* dual panda robot 起動
* task environment 起動
* rollout 実行
* headless simulation 実行

ログ：

```text
Using dual panda robot
Starting episode 0
```

---

# 2. checkpoint 読み込み成功

以下 checkpoint を正常ロード：

```text
checkpoint_peract_bc_leader_layer_0.pt
checkpoint_peract_bc_follower_layer_0.pt
```

---

# 3. policy rollout 実行成功

以下が実行された：

* observation取得
* voxelization
* policy inference
* action prediction
* simulator rollout

---

# 4. cinematic recorder 動作確認

動画生成成功：

```text
/home/tappei-m/Project/AnyBimanual_logs/debug_test/PERACT_BC/seed0/videos/bimanual_pick_laptop_w0_s0_fail.mp4
```

---

# 5. evaluation pipeline 完走

ログ最後：

```text
Finished evaluation.
```

まで到達。

---

# 評価結果

```text
Evaluating bimanual_pick_laptop | Episode 0 | Score: 0.0
```

task 成功は確認できなかった。

---

# 動画について

生成された rollout 動画では、ロボットアームはほとんど動作しなかった。

これは：

* training_iterations=5 の debug training のみ実施
* 十分な学習が行われていない

ためと考えられる。

---

# 結論

今回の evaluation により、

* RLBench
* CoppeliaSim
* dual-arm environment
* checkpoint loading
* policy rollout
* video generation

を含む AnyBimanual の評価パイプラインが、
end-to-end で実行可能であることを確認した。

一方で、今回は debug 用の短時間学習のみであり、
論文レベルの manipulation performance は未確認。
