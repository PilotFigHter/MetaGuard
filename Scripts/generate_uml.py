"""
Generate UML Class Diagram for MetaGuard Website
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import matplotlib.lines as mlines

def draw_class(ax, x, y, name, attributes, methods, color='#2c3e50'):
    width = 6
    height = 0.4 + len(attributes) * 0.35 + len(methods) * 0.35 + 0.2
    
    # Main box
    box = FancyBboxPatch((x, y - height), width, height,
                          boxstyle="round,pad=0.05,rounding_size=0.2",
                          facecolor='#ecf0f1', edgecolor=color, linewidth=2)
    ax.add_patch(box)
    
    # Header
    header = FancyBboxPatch((x, y - 0.6), width, 0.6,
                             boxstyle="round,pad=0.02,rounding_size=0.1",
                             facecolor=color, edgecolor=color, linewidth=0)
    ax.add_patch(header)
    
    # Class name
    ax.text(x + width/2, y - 0.35, name, ha='center', va='center',
            fontsize=12, fontweight='bold', color='white')
    
    # Attributes
    for i, attr in enumerate(attributes):
        ax.text(x + 0.15, y - 0.7 - i * 0.35, attr, ha='left', va='center',
                fontsize=9, color='#2c3e50')
    
    # Separator line
    sep_y = y - 0.6 - len(attributes) * 0.35 - 0.1
    ax.plot([x, x + width], [sep_y, sep_y], color=color, linewidth=1)
    
    # Methods
    for i, method in enumerate(methods):
        ax.text(x + 0.15, sep_y - 0.15 - i * 0.35, method, ha='left', va='center',
                fontsize=9, color='#2c3e50', style='italic')
    
    return height

def draw_arrow(ax, x1, y1, x2, y2, label=''):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.5))
    if label:
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.2, label, ha='center', fontsize=8, color='#7f8c8d')

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.set_aspect('equal')
ax.axis('off')

# Title
ax.text(8, 11.5, 'MetaGuard - UML Class Diagram', ha='center', va='center',
        fontsize=16, fontweight='bold', color='#2c3e50')
ax.text(8, 11, 'Website Architecture (Flask Application)', ha='center', va='center',
        fontsize=12, color='#7f8c8d')

# Class 1: Flask App (top left)
h1 = draw_class(ax, 0.5, 10, 'Flask App',
    ['- sniffer_active: bool', '- prediction_results: list', '- selected_model: str', '- ensemble: Model', '- scaler: Scaler', '- nn_model_wrapper: NNWrapper'],
    ['+ index()', '+ start()', '+ stop()', '+ demo()', '+ results()', '+ debug()'],
    '#e74c3c')

# Class 2: Sniffer Thread (top right)
h2 = draw_class(ax, 9.5, 10, 'SnifferThread',
    ['- sniffer_active: bool'],
    ['+ sniffer_loop()', '+ process(pkt)', '+ get_features(pkt)'],
    '#3498db')

# Class 3: Feature Extractor (middle)
h3 = draw_class(ax, 5, 7, 'FeatureExtractor',
    ['+ FEATURES: list'],
    ['+ get_features(pkt)', '+ get_proto_name(proto)'],
    '#27ae60')

# Class 4: NN Model Wrapper (middle right)
h4 = draw_class(ax, 9.5, 7, 'NNModelWrapper',
    ['- model: Net', '- scaler: Scaler', '- label_encoder: LabelEncoder'],
    ['+ predict(X)'],
    '#9b59b6')

# Class 5: Scaler (bottom left)
h5 = draw_class(ax, 0.5, 4.5, 'Scaler',
    ['- scaler: StandardScaler'],
    ['+ transform(X)'],
    '#f39c12')

# Class 6: Ensemble Model (bottom center-left)
h6 = draw_class(ax, 4, 4.5, 'EnsembleModel',
    ['- rf: RandomForest', '- xgb: XGBoost', '- lgbm: LightGBM', '- weights: list'],
    ['+ predict(X)'],
    '#1abc9c')

# Class 7: HTML Template (bottom right)
h7 = draw_class(ax, 9.5, 4.5, 'HTMLTemplate',
    ['- index.html'],
    ['+ render()'],
    '#e67e22')

draw_arrow(ax, 4.5, 8.2, 5.7, 7.8, 'extracts features')
draw_arrow(ax, 7.7, 7.6, 9.5, 7.6, 'uses')
draw_arrow(ax, 3, 4.5, 4, 4.8, 'normalizes')
draw_arrow(ax, 6, 4.5, 6.3, 6, 'predicts')
draw_arrow(ax, 7, 9.3, 9.5, 9.3, 'controls')

# Legend
legend_y = 1.2
ax.add_patch(FancyBboxPatch((0.5, legend_y - 0.8), 15, 1,
             boxstyle="round,pad=0.1", facecolor='#f8f9fa', edgecolor='#dee2e6'))
ax.text(8, legend_y + 0.1, 'Legend', ha='center', fontsize=10, fontweight='bold')

# Arrow legend
ax.annotate('', xy=(2.5, legend_y - 0.4), xytext=(1.5, legend_y - 0.4),
            arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.5))
ax.text(2, legend_y - 0.4, 'Association', ha='center', va='center', fontsize=9)

# Color legend items
colors = ['#e74c3c', '#3498db', '#27ae60', '#9b59b6', '#f39c12', '#1abc9c', '#e67e22']
labels_legend = ['Flask App', 'Sniffer', 'Features', 'NN', 'Scaler', 'Ensemble', 'HTML']
for i, (c, l) in enumerate(zip(colors, labels_legend)):
    ax.add_patch(patches.Rectangle((9 + i * 0.9, legend_y - 0.6), 0.3, 0.3,
                                   facecolor=c, edgecolor='black'))
    ax.text(9.4 + i * 0.9, legend_y - 0.45, l, ha='left', va='center', fontsize=7)

# Data flow note
ax.text(8, 0.3, 'Data Flow: Packet → Extract Features → Normalize → Predict → Display',
        ha='center', va='center', fontsize=9, color='#7f8c8d', style='italic')

def layout_uml_grid(ax):
    # Clear area and set up a tidy grid layout (non-overlapping)
    ax.set_xlim(-1, 13)
    ax.set_ylim(-1, 13)
    ax.axis('off')

    # Row 1
    draw_class(ax, 0.5, 11.5, 'Flask App',
        ['- sniffer_active: bool', '- prediction_results: list', '- selected_model: str', '- ensemble: Model', '- scaler: Scaler', '- nn_model_wrapper: NNWrapper'],
        ['+ index()', '+ start()', '+ stop()', '+ demo()', '+ results()', '+ debug()'],
        '#e74c3c')
    draw_class(ax, 7.5, 11.5, 'SnifferThread',
        ['- sniffer_active: bool'],
        ['+ sniffer_loop()', '+ process(pkt)', '+ get_features(pkt)'],
        '#3498db')

    # Row 2
    draw_class(ax, 0.5, 7.0, 'FeatureExtractor',
        ['+ FEATURES: list'],
        ['+ get_features(pkt)', '+ get_proto_name(proto)'],
        '#27ae60')
    draw_class(ax, 7.5, 7.0, 'NNModelWrapper',
        ['- model: Net', '- scaler: Scaler', '- label_encoder: LabelEncoder'],
        ['+ predict(X)'],
        '#9b59b6')

    # Row 3
    draw_class(ax, 0.5, 3.0, 'Scaler',
        ['- scaler: StandardScaler'],
        ['+ transform(X)'],
        '#f39c12')
    draw_class(ax, 6.5, 3.0, 'EnsembleModel',
        ['- rf: RandomForest', '- xgb: XGBoost', '- lgbm: LightGBM', '- weights: list'],
        ['+ predict(X)'],
        '#1abc9c')
    draw_class(ax, 11.5, 3.0, 'HTMLTemplate',
        ['- index.html'],
        ['+ render()'],
        '#e67e22')

    # Optional simple arrows to illustrate flow
    draw_arrow(ax, 4.0, 11.0, 7.0, 9.0, 'extracts → uses')
    draw_arrow(ax, 4.0, 7.5, 4.0, 5.0, 'normalizes')

layout_uml_grid(ax)
plt.tight_layout()
plt.savefig('MetaWeb/UML_Diagram.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('MetaWeb/UML_Diagram.pdf', bbox_inches='tight', facecolor='white')
print("UML diagram rebuilt with improved spacing at MetaWeb/UML_Diagram.png and .pdf")
