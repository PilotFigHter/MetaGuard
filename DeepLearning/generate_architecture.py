"""
Generate Neural Network Architecture Diagram
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Colors
colors = {
    'input': '#3498db',
    'hidden': '#2ecc71', 
    'output': '#e74c3c',
    'arrow': '#7f8c8d',
    'text': '#2c3e50'
}

# Title
ax.text(7, 9.3, 'Neural Network Architecture - MetaGuard', fontsize=18, fontweight='bold', 
       ha='center', color=colors['text'])
ax.text(7, 8.9, 'Input(6) → 64 → 32 → Output(15)', fontsize=12, 
       ha='center', color='#7f8c8d')

# Input Layer
input_layer = mpatches.FancyBboxPatch((0.5, 4), 1.5, 3, boxstyle="round,pad=0.05",
                                       facecolor=colors['input'], edgecolor='black', linewidth=2)
ax.add_patch(input_layer)
ax.text(1.25, 7.3, 'INPUT', fontsize=11, ha='center', fontweight='bold', color='white')
ax.text(1.25, 6.7, 'Layer', fontsize=11, ha='center', color='white')
ax.text(1.25, 6.0, '(6)', fontsize=10, ha='center', color='white')

# Arrow
ax.annotate('', xy=(2.5, 5.5), xytext=(2, 5.5),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))

# Hidden Layer 1 (64 neurons)
hidden1 = mpatches.FancyBboxPatch((2.5, 2), 2, 5, boxstyle="round,pad=0.05",
                                   facecolor=colors['hidden'], edgecolor='black', linewidth=2)
ax.add_patch(hidden1)
ax.text(3.5, 7.3, 'HIDDEN', fontsize=11, ha='center', fontweight='bold', color='white')
ax.text(3.5, 6.7, 'Layer 1', fontsize=11, ha='center', color='white')
ax.text(3.5, 6.0, '(64)', fontsize=10, ha='center', color='white')
ax.text(3.5, 5.3, 'ReLU', fontsize=9, ha='center', color='white')

# Arrow
ax.annotate('', xy=(5, 5.5), xytext=(4.5, 5.5),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))

# Hidden Layer 2 (32 neurons)
hidden2 = mpatches.FancyBboxPatch((5, 3.5), 2, 3, boxstyle="round,pad=0.05",
                                   facecolor=colors['hidden'], edgecolor='black', linewidth=2)
ax.add_patch(hidden2)
ax.text(6, 5.8, 'HIDDEN', fontsize=10, ha='center', fontweight='bold', color='white')
ax.text(6, 5.3, 'Layer 2', fontsize=10, ha='center', color='white')
ax.text(6, 4.8, '(32)', fontsize=10, ha='center', color='white')
ax.text(6, 4.3, 'ReLU', fontsize=9, ha='center', color='white')

# Arrow
ax.annotate('', xy=(7.5, 5.5), xytext=(7, 5.5),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))

# Output Layer
output_layer = mpatches.FancyBboxPatch((7.5, 4), 2, 3, boxstyle="round,pad=0.05",
                                       facecolor=colors['output'], edgecolor='black', linewidth=2)
ax.add_patch(output_layer)
ax.text(8.5, 6.3, 'OUTPUT', fontsize=11, ha='center', fontweight='bold', color='white')
ax.text(8.5, 5.8, 'Layer', fontsize=11, ha='center', color='white')
ax.text(8.5, 5.3, '(15)', fontsize=10, ha='center', color='white')
ax.text(8.5, 4.8, 'Softmax', fontsize=9, ha='center', color='white')

# Arrows from hidden to output (multiple)
for i in range(3):
    y_pos = 5.5 + (i-1)*0.5
    ax.annotate('', xy=(7.5, y_pos), xytext=(7, y_pos),
              arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1, alpha=0.5))

# Legend box
legend_box = mpatches.FancyBboxPatch((10.5, 1), 3.5, 4, boxstyle="round,pad=0.05",
                                      facecolor='#f8f9fa', edgecolor='#bdc3c7', linewidth=1)
ax.add_patch(legend_box)
ax.text(12.25, 4.7, 'Layer Types', fontsize=11, ha='center', fontweight='bold')
ax.text(11.5, 4.2, '■ Input: 6 features', fontsize=9, color=colors['input'])
ax.text(11.5, 3.7, '■ Hidden: ReLU activation', fontsize=9, color=colors['hidden'])
ax.text(11.5, 3.2, '■ Output: Softmax + CE Loss', fontsize=9, color=colors['output'])
ax.text(11.5, 2.5, 'Optimizer: Adam (lr=0.001)', fontsize=9, color=colors['text'])

# Description box
desc_box = mpatches.FancyBboxPatch((0.5, 0.3), 9, 1.3, boxstyle="round,pad=0.05",
                                   facecolor='#ecf0f1', edgecolor='#bdc3c7', linewidth=1)
ax.add_patch(desc_box)
ax.text(5, 1.35, 'CrossEntropyLoss:Measures classification error between predicted and actual labels', 
       fontsize=9, ha='center', style='italic', color=colors['text'])
ax.text(5, 0.85, 'Adam Optimizer:Adaptive learning rate with momentum for faster convergence', 
       fontsize=9, ha='center', style='italic', color=colors['text'])

plt.tight_layout()
plt.savefig('DeepLearning/results/nn_architecture.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Architecture diagram saved to DeepLearning/results/nn_architecture.png")