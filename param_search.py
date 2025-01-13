import itertools
import subprocess

# -------------------------------------------- #
#                 定义超参数选项                #
# -------------------------------------------- #

param_options = {
    "learn_rate": [1e-6, 1e-5, 1e-4, 2e-4, 5e-4],  # 学习率
    "batch": [2, 4, 8, 16],  # batch大小
    "epochs": [50, 100, 150, 200],  # 训练轮次
    "beta1": [0.5, 0.6, 0.7, 0.8, 0.9],  # Adam优化器的b1参数
    "beta2": [0.999, 0.98],  # Adam优化器的b2参数
    "decay": [0, 1e-4, 1e-5],  # 权重衰减，防止过拟合
    "moment": [0.9, 0.95],  # 动量参数，用于优化器
    "optim": ["Adam", "AdamW", "RMSprop", "SGD"],  # 不同的优化器
    "drop_out": [0.3, 0.4, 0.5, 0.6, 0.7],  # dropout值，用于防止过拟合
}

# -------------------------------------------- #
#           生成所有可能的参数组合              #
# -------------------------------------------- #

param_combos = list(itertools.product(
    param_options["learn_rate"],  # 学习率
    param_options["batch"],       # batch大小
    param_options["epochs"],      # 训练轮次
    param_options["beta1"],       # Adam优化器b1参数
    param_options["beta2"],       # Adam优化器b2参数
    param_options["decay"],       # 权重衰减
    param_options["moment"],      # 动量参数
    param_options["optim"],       # 优化器类型
    param_options["drop_out"]     # dropout概率
))

# -------------------------------------------- #
#             遍历每种参数组合并训练            #
# -------------------------------------------- #

for combo in param_combos:
    lr, batch, n_epochs, b1, b2, decay, moment, optimizer, dropout = combo

    # 打印当前使用的超参数组合
    print(f"正在使用超参数：lr={lr}, batch_size={batch}, n_epochs={n_epochs}, b1={b1}, b2={b2}, decay={decay}, moment={moment}, optimizer={optimizer}, dropout={dropout}")

    # 调用训练脚本并传入当前超参数组合
    subprocess.run([
        "python", "E:/A_workbench/Object/遮挡识别/遮挡恢复/1.train_2.py",  # 替换为你自己的训练脚本路径
        "--lr", str(lr),               # 学习率
        "--batch_size", str(batch),    # batch大小
        "--n_epochs", str(n_epochs),   # 训练轮次
        "--b1", str(b1),               # Adam的b1参数
        "--b2", str(b2),               # Adam的b2参数
        "--weight_decay", str(decay),  # 权重衰减
        "--momentum", str(moment),     # 动量参数
        "--optimizer", str(optimizer), # 优化器类型
        "--dropout", str(dropout)      # dropout概率
    ])

# -------------------------------------------- #
#               训练完成后的提示                #
# -------------------------------------------- #
print("所有参数组合的训练已完成！")
