import matplotlib.pyplot as plt

alpha = [0.00, 0.005, 0.075, 0.09, 0.10, 0.105, 0.11, 0.125, 0.15]
val_mse = [
    0.142738,  # 0.00
    0.136218,  # 0.05
    0.135953,  # 0.075
    0.135520,  # 0.09
    0.141903,  # 0.095
    0.134880,  # 0.10
    0.135583,  # 0.105
    0.137851,  # 0.11
    0.136231,  # 0.125
    0.140596   # 0.15
]

plt.figure()
plt.plot(alpha, val_mse, marker='o')
plt.axvline(0.10, linestyle='--', alpha=0.5)
plt.xlabel("Log-price scaling α")
plt.ylabel("Validation MSE (epoch 50)")
plt.title("Log-Price Scaling Sweep: Options Nodes")
plt.show()
