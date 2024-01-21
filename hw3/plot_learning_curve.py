import json
import matplotlib.pyplot as plt

# 讀取 JSON 檔案
with open('output/v4/checkpoint-250/trainer_state.json', 'r') as file:
    trainer_state = json.load(file)

# 提取訓練損失和評估損失
train_losses = [entry["loss"] for entry in trainer_state["log_history"] if "loss" in entry]
eval_losses = [entry["eval_loss"] for entry in trainer_state["log_history"] if "eval_loss" in entry]

# 提取步驟數
steps = list(set([entry["step"] for entry in trainer_state["log_history"]]))
steps.sort()


# 繪製 learning curve
plt.figure(figsize=(10, 6))
# plt.plot(steps, train_losses, label="Training Loss", marker='o')
# plt.plot(steps, eval_losses, label="Evaluation Loss", marker='o')
plt.plot(steps, eval_losses, marker='o')

plt.title('Learning Curve')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('learning_curve.png')
plt.show()
