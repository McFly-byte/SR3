# Windows 多GPU训练解决方案

## 问题
Windows 上 PyTorch 分布式训练存在 `USE_LIBUV` 兼容性问题，环境变量设置无法传递到子进程。

## 解决方案

### 方案1：使用单GPU训练（推荐，最稳定）

```powershell
python sr.py -p train -c config/sr3_mrsi_64.json -gpu 0
```

**优点**：稳定可靠，无需处理分布式问题
**缺点**：训练速度较慢

### 方案2：使用 torch.distributed.launch（尝试）

```powershell
python train_multi_gpu_launch.py
```

或者直接：

```powershell
$env:USE_LIBUV="0"; python -m torch.distributed.launch --nproc_per_node=3 --master_port=29500 sr.py -p train -c config/sr3_mrsi_64.json
```

### 方案3：修改配置文件使用单GPU

编辑 `config/sr3_mrsi_64.json`，将 `gpu_ids` 改为单个GPU：

```json
"gpu_ids": [0]
```

然后运行：

```powershell
python sr.py -p train -c config/sr3_mrsi_64.json
```

### 方案4：使用 WSL2（如果支持）

在 WSL2 中运行，可以正常使用多GPU训练。

## 建议

对于开发和测试，**强烈建议使用方案1（单GPU训练）**，因为：
1. 稳定可靠，不会遇到分布式问题
2. 代码逻辑完全相同，只是速度较慢
3. 可以正常完成所有训练流程

如果需要多GPU训练，建议在 Linux 服务器上运行。


