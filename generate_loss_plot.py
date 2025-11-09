import matplotlib.pyplot as plt
import re

# Read the log file
with open('/home/bhanu/pmni/PMNI/logs/train_20251108_121455.log', 'r') as f:
    log_content = f.read()

# Extract loss values using regex
loss_pattern = r'(\d+)/30000.*loss=(\d+\.\d+e[+-]\d+)'
matches = re.findall(loss_pattern, log_content)

iterations = []
losses = []

for match in matches:
    iter_num = int(match[0])
    loss_val = float(match[1])
    iterations.append(iter_num)
    losses.append(loss_val)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(iterations, losses, 'b-', linewidth=2, alpha=0.7)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('PMNI Training Loss Convergence - Bear Object')
plt.grid(True, alpha=0.3)
plt.yscale('log')  # Log scale for better visualization
plt.tight_layout()

# Save the plot
plt.savefig('/home/bhanu/pmni/PMNI/report/images/loss_convergence.png', dpi=300, bbox_inches='tight')
plt.close()

print("Loss convergence plot generated!")