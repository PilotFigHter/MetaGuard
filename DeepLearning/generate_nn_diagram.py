"""
Generate detailed Neural Network diagram with visible neurons
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_facecolor('#fafafa')

# Title
ax.text(8, 9.5, 'Neural Network Architecture - MetaGuard', fontsize=18, fontweight='bold', ha='center', color='#2c3e50')
ax.text(8, 9.1, 'Deep Learning Model for Network Intrusion Detection', fontsize=12, ha='center', color='#7f8c8d')

# Colors
colors = {'input': '#3498db', 'hidden': '#27ae60', 'output': '#e74c3c', 'connection': '#bdc3c7'}

# Function to draw neurons
def draw_neurons(ax, x, y_positions, color, label, count):
    for i, y in enumerate(y_positions):
        circle = Circle((x, y), 0.15, facecolor=color, edgecolor='black', linewidth=1.5, zorder=5)
        ax.add_patch(circle)
    # Layer label
    ax.text(x, y_positions[0] + 1.2, label, fontsize=9, ha='center', fontweight='bold', color=color)
    ax.text(x, y_positions[-1] - 0.5, f'n={count}', fontsize=8, ha='center', color='#7f8c8d')

# Input Layer (6 neurons)
input_y = np.linspace(6.5, 3.5, 6)
for i, y in enumerate(input_y):
    circle = Circle((1.5, y), 0.18, facecolor=colors['input'], edgecolor='black', linewidth=2, zorder=5)
    ax.add_patch(circle)
    # Feature name
    features = ['Init Fwd\nWin Byts', 'Fwd Seg\nSize Min', 'Protocol', 'Fwd Header\nLen', 'Fwd Pkt\nLen Max', 'ACK Flag\nCnt']
    ax.text(0.5, y, features[i][:12], fontsize=7, ha='right', va='center', color='#2c3e50')

ax.text(1.5, 7.5, 'INPUT LAYER', fontsize=10, ha='center', fontweight='bold', color=colors['input'])
ax.text(1.5, 2.8, '(6 features)', fontsize=9, ha='center', color='#7f8c8d')

# Arrow from input to hidden1
for y in input_y:
    ax.plot([1.68, 4.3], [y, y*0.85+0.8], color=colors['connection'], alpha=0.3, linewidth=0.5, zorder=1)

# Hidden Layer 1 (12 representative neurons shown)
hidden1_y = np.linspace(7, 3, 12)
for y in hidden1_y:
    circle = Circle((5.5, y), 0.2, facecolor=colors['hidden'], edgecolor='black', linewidth=1.5, zorder=5)
    ax.add_patch(circle)

ax.text(5.5, 7.7, 'HIDDEN LAYER 1', fontsize=10, ha='center', fontweight='bold', color=colors['hidden'])
ax.text(5.5, 2.8, '(64 neurons)', fontsize=9, ha='center', color='#7f8c8d')
ax.text(5.5, 2.4, 'ReLU activation', fontsize=8, ha='center', style='italic', color='#7f8c8d')

# Arrow from hidden1 to hidden2
for i, y in enumerate(hidden1_y[::2]):
    target_y = 5.5 + (i-2)*0.3
    ax.plot([5.7, 8.3], [y, target_y], color=colors['connection'], alpha=0.3, linewidth=0.5, zorder=1)

# Hidden Layer 2 (8 representative neurons)
hidden2_y = np.linspace(6.2, 4.2, 8)
for y in hidden2_y:
    circle = Circle((9.5, y), 0.2, facecolor=colors['hidden'], edgecolor='black', linewidth=1.5, zorder=5)
    ax.add_patch(circle)

ax.text(9.5, 6.9, 'HIDDEN LAYER 2', fontsize=10, ha='center', fontweight='bold', color=colors['hidden'])
ax.text(9.5, 3.8, '(32 neurons)', fontsize=9, ha='center', color='#7f8c8d')
ax.text(9.5, 3.4, 'ReLU activation', fontsize=8, ha='center', style='italic', color='#7f8c8d')

# Arrow from hidden2 to output
for i, y in enumerate(hidden2_y[::2]):
    target_y = 5.2 + (i-2)*0.3
    ax.plot([9.7, 12.3], [y, target_y], color=colors['connection'], alpha=0.3, linewidth=0.5, zorder=1)

# Output Layer (15 neurons - show as row)
output_x = np.linspace(13, 15.5, 6)
output_y = 5.2
for x in output_x:
    circle = Circle((x, output_y), 0.18, facecolor=colors['output'], edgecolor='black', linewidth=1.5, zorder=5)
    ax.add_patch(circle)
output_x2 = np.linspace(13.5, 14.5, 4)
for x in output_x2:
    circle = Circle((x, output_y-0.5), 0.18, facecolor=colors['output'], edgecolor='black', linewidth=1.5, zorder=5)
    ax.add_patch(circle)
output_x3 = np.linspace(13.8, 14.2, 2)
for x in output_x3:
    circle = Circle((x, output_y-1.0), 0.18, facecolor=colors['output'], edgecolor='black', linewidth=1.5, zorder=5)
    ax.add_patch(circle)
circle = Circle((14, output_y-1.5), 0.18, facecolor=colors['output'], edgecolor='black', linewidth=1.5, zorder=5)
ax.add_patch(circle)

ax.text(14.2, 6.5, 'OUTPUT LAYER', fontsize=10, ha='center', fontweight='bold', color=colors['output'])
ax.text(14.2, 5.8, '(15 classes)', fontsize=9, ha='center', color='#7f8c8d')
ax.text(14.2, 5.3, 'Softmax', fontsize=8, ha='center', style='italic', color='#7f8c8d')

# Arrows between layers
ax.annotate('', xy=(4, 5.5), xytext=(2, 5.5), arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=2))
ax.annotate('', xy=(7.5, 5.5), xytext=(5.7, 5.5), arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=2))
ax.annotate('', xy=(11, 5.5), xytext=(9.7, 5.5), arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=2))

# Info boxes
# Loss box
loss_box = FancyBboxPatch((0.5, 0.3), 5, 1.3, boxstyle="round,pad=0.05",
                         facecolor='#fff3cd', edgecolor='#ffc107', linewidth=1)
ax.add_patch(loss_box)
ax.text(3, 1.05, 'Loss Function: CrossEntropyLoss', fontsize=10, ha='center', fontweight='bold', color='#856404')
ax.text(3, 0.65, 'Measures classification error between predicted and actual labels', 
       fontsize=8, ha='center', style='italic', color='#856404')

# Optimizer box
opt_box = FancyBboxPatch((6, 0.3), 4.5, 1.3, boxstyle="round,pad=0.05",
                        facecolor='#d4edda', edgecolor='#28a745', linewidth=1)
ax.add_patch(opt_box)
ax.text(8.25, 1.05, 'Optimizer: Adam', fontsize=10, ha='center', fontweight='bold', color='#155724')
ax.text(8.25, 0.65, 'lr=0.001, with momentum', fontsize=8, ha='center', style='italic', color='#155724')

# Results box
res_box = FancyBboxPatch((11, 0.3), 4.5, 1.3, boxstyle="round,pad=0.05",
                        facecolor='#f8d7da', edgecolor='#dc3545', linewidth=1)
ax.add_patch(res_box)
ax.text(13.25, 1.05, 'Results', fontsize=10, ha='center', fontweight='bold', color='#721c24')
ax.text(13.25, 0.65, 'Acc: 88.1% | F1: 66.6%', fontsize=8, ha='center', color='#721c24')

plt.tight_layout()
plt.savefig('DeepLearning/results/nn_architecture.png', dpi=200, bbox_inches='tight', facecolor='#fafafa')
plt.close()
print("Done!")