"""
Visualization of Trend Analysis Process
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Use A4 landscape size (requirement: dashboards must match A4 dimensions)
A4_FIGSIZE = (11.69, 8.27)

# Set font to support Persian
plt.rcParams['font.family'] = 'Arial'

fig, ax = plt.subplots(figsize=A4_FIGSIZE)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, 'Trend Analysis Process', 
        ha='center', va='top', fontsize=20, fontweight='bold')

# Level 1: Indicators
level1_y = 8
ax.text(5, level1_y + 0.3, 'Level 1: Trend Indicators', 
        ha='center', fontsize=14, fontweight='bold', color='#2c3e50')

indicators = [
    ('SMA\n0.8', 1), ('EMA\n0.85', 1.8), ('WMA\n0.7', 2.6),
    ('DEMA\n0.75', 3.4), ('TEMA\n0.78', 4.2), ('MACD\n0.88', 5),
    ('ADX\n0.9', 5.8), ('SAR\n0.72', 6.6), ('Super\n0.8', 7.4),
    ('Ichi\n0.85', 8.2)
]

for name, x in indicators:
    box = FancyBboxPatch((x-0.3, level1_y-0.5), 0.6, 0.4,
                         boxstyle="round,pad=0.05", 
                         edgecolor='#3498db', facecolor='#ecf0f1',
                         linewidth=2)
    ax.add_patch(box)
    lines = name.split('\n')
    ax.text(x, level1_y-0.15, lines[0], ha='center', fontsize=8, fontweight='bold')
    ax.text(x, level1_y-0.35, lines[1], ha='center', fontsize=7, color='#27ae60')

# Level 2: Category Score
level2_y = 6
ax.text(5, level2_y + 0.5, 'Level 2: Category Score', 
        ha='center', fontsize=14, fontweight='bold', color='#2c3e50')

# Formula box
formula_box = FancyBboxPatch((2, level2_y-0.6), 6, 0.8,
                            boxstyle="round,pad=0.1", 
                            edgecolor='#e74c3c', facecolor='#fadbd8',
                            linewidth=2)
ax.add_patch(formula_box)
ax.text(5, level2_y-0.05, 'Trend Score = Σ(Signal × Confidence) / Σ(Confidence)', 
        ha='center', fontsize=10, fontweight='bold')
ax.text(5, level2_y-0.35, 'Trend Accuracy = Average(All Confidences)', 
        ha='center', fontsize=9, style='italic')

# Arrows from Level 1 to Level 2
for name, x in indicators:
    arrow = FancyArrowPatch((x, level1_y-0.5), (5, level2_y+0.2),
                           arrowstyle='->', mutation_scale=15, 
                           color='#95a5a6', alpha=0.4, linewidth=1)
    ax.add_patch(arrow)

# Result box
result_box = FancyBboxPatch((3.5, level2_y-1.3), 3, 0.5,
                           boxstyle="round,pad=0.05", 
                           edgecolor='#27ae60', facecolor='#d5f4e6',
                           linewidth=2)
ax.add_patch(result_box)
ax.text(5, level2_y-0.95, 'Trend: 0.89 (Bullish)', 
        ha='center', fontsize=11, fontweight='bold', color='#27ae60')
ax.text(5, level2_y-1.15, 'Accuracy: 0.81 (81%)', 
        ha='center', fontsize=9, color='#16a085')

# Level 3: Weight Adjustment
level3_y = 3.5
ax.text(5, level3_y + 0.5, 'Level 3: Weight Adjustment', 
        ha='center', fontsize=14, fontweight='bold', color='#2c3e50')

# Categories
categories = [
    ('Trend\n30%→33%\n0.81 acc', 2),
    ('Momentum\n25%→24%\n0.7 acc', 4),
    ('Cycle\n25%→22%\n0.65 acc', 6),
    ('Volume\n20%→21%\n0.75 acc', 8)
]

for name, x in categories:
    color = '#27ae60' if '→33%' in name else '#95a5a6'
    box = FancyBboxPatch((x-0.6, level3_y-0.6), 1.2, 0.8,
                        boxstyle="round,pad=0.05", 
                        edgecolor=color, facecolor='#ecf0f1',
                        linewidth=2)
    ax.add_patch(box)
    lines = name.split('\n')
    for i, line in enumerate(lines):
        y_offset = 0.2 - i*0.25
        fontsize = 9 if i == 0 else 8
        fontweight = 'bold' if i == 0 else 'normal'
        ax.text(x, level3_y+y_offset, line, ha='center', 
               fontsize=fontsize, fontweight=fontweight)

# Arrow from Level 2 to Level 3
arrow = FancyArrowPatch((5, level2_y-1.3), (5, level3_y+0.3),
                       arrowstyle='->', mutation_scale=20, 
                       color='#e74c3c', linewidth=2)
ax.add_patch(arrow)

# Level 4: Final Signal
level4_y = 1
ax.text(5, level4_y + 0.5, 'Level 4: Final Signal', 
        ha='center', fontsize=14, fontweight='bold', color='#2c3e50')

final_box = FancyBboxPatch((2.5, level4_y-0.7), 5, 1,
                          boxstyle="round,pad=0.1", 
                          edgecolor='#8e44ad', facecolor='#e8daef',
                          linewidth=3)
ax.add_patch(final_box)
ax.text(5, level4_y-0.05, 'Overall Score = (Trend × 33%) + (Mom × 24%) + (Cyc × 22%)', 
        ha='center', fontsize=9, fontweight='bold')
ax.text(5, level4_y-0.25, 'Volume Confirmation Applied', 
        ha='center', fontsize=8, style='italic')
ax.text(5, level4_y-0.45, 'Final Signal: Bullish (0.64)', 
        ha='center', fontsize=12, fontweight='bold', color='#27ae60')
ax.text(5, level4_y-0.62, 'Confidence: 82%', 
        ha='center', fontsize=10, color='#16a085')

# Arrows from Level 3 to Level 4
for name, x in categories:
    arrow = FancyArrowPatch((x, level3_y-0.6), (5, level4_y+0.3),
                           arrowstyle='->', mutation_scale=15, 
                           color='#95a5a6', alpha=0.4, linewidth=1)
    ax.add_patch(arrow)

# Legend
legend_elements = [
    mpatches.Patch(color='#3498db', label='Indicators'),
    mpatches.Patch(color='#e74c3c', label='Calculation'),
    mpatches.Patch(color='#27ae60', label='Higher Weight'),
    mpatches.Patch(color='#8e44ad', label='Final Result')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

# Add note
note_text = ('Note: Higher accuracy increases category weight\n'
            'Trend has 81% accuracy → gets 33% weight (up from 30%)')
ax.text(5, 0.2, note_text, ha='center', fontsize=8, 
       style='italic', color='#7f8c8d',
       bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.5))

plt.tight_layout()
plt.savefig('trend_analysis_flow.png', dpi=150, bbox_inches='tight')
print("✅ Trend analysis flow diagram saved as 'trend_analysis_flow.png'")
plt.close()

# Create a second diagram showing indicator signals
fig, ax = plt.subplots(figsize=A4_FIGSIZE)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9.5, 'Indicator Signal Calculation', 
        ha='center', va='top', fontsize=18, fontweight='bold')

# SMA Example
y = 8
box = FancyBboxPatch((0.5, y-1), 4, 1.5,
                    boxstyle="round,pad=0.1", 
                    edgecolor='#3498db', facecolor='#ebf5fb',
                    linewidth=2)
ax.add_patch(box)
ax.text(2.5, y+0.3, 'SMA Signal Calculation', ha='center', 
       fontsize=12, fontweight='bold', color='#2c3e50')
ax.text(2.5, y, 'Price: $50,000', ha='center', fontsize=9)
ax.text(2.5, y-0.2, 'SMA(20): $48,000', ha='center', fontsize=9)
ax.text(2.5, y-0.4, 'Diff: 4.17%', ha='center', fontsize=9, color='#27ae60')
ax.text(2.5, y-0.65, '→ Signal: Bullish', ha='center', fontsize=10, 
       fontweight='bold', color='#27ae60')
ax.text(2.5, y-0.85, 'Confidence: 0.80', ha='center', fontsize=9)

# MACD Example  
box = FancyBboxPatch((5.5, y-1), 4, 1.5,
                    boxstyle="round,pad=0.1", 
                    edgecolor='#e74c3c', facecolor='#fadbd8',
                    linewidth=2)
ax.add_patch(box)
ax.text(7.5, y+0.3, 'MACD Signal Calculation', ha='center', 
       fontsize=12, fontweight='bold', color='#2c3e50')
ax.text(7.5, y, 'MACD: 120', ha='center', fontsize=9)
ax.text(7.5, y-0.2, 'Signal: 90', ha='center', fontsize=9)
ax.text(7.5, y-0.4, 'Histogram: 30', ha='center', fontsize=9, color='#27ae60')
ax.text(7.5, y-0.65, '→ Signal: Very Bullish', ha='center', fontsize=10, 
       fontweight='bold', color='#27ae60')
ax.text(7.5, y-0.85, 'Confidence: 0.88', ha='center', fontsize=9)

# Signal Strength Scale
y = 5.5
ax.text(5, y+0.5, 'Signal Strength Scale', ha='center', 
       fontsize=14, fontweight='bold', color='#2c3e50')

signals = [
    ('Very Bullish\n+2.0', 1, '#27ae60'),
    ('Bullish\n+1.0', 2.2, '#52be80'),
    ('Bullish Broken\n+0.5', 3.4, '#82e0aa'),
    ('Neutral\n0.0', 4.6, '#95a5a6'),
    ('Bearish Broken\n-0.5', 5.8, '#f1948a'),
    ('Bearish\n-1.0', 7, '#ec7063'),
    ('Very Bearish\n-2.0', 8.2, '#e74c3c')
]

for name, x, color in signals:
    box = FancyBboxPatch((x-0.5, y-0.6), 1, 0.8,
                        boxstyle="round,pad=0.05", 
                        edgecolor=color, facecolor=color,
                        alpha=0.3, linewidth=2)
    ax.add_patch(box)
    lines = name.split('\n')
    ax.text(x, y-0.05, lines[0], ha='center', fontsize=8, fontweight='bold')
    ax.text(x, y-0.35, lines[1], ha='center', fontsize=9, fontweight='bold', color=color)

# Confidence Levels
y = 3.5
ax.text(5, y+0.5, 'Confidence Levels', ha='center', 
       fontsize=14, fontweight='bold', color='#2c3e50')

conf_levels = [
    ('Very High\n0.8-1.0', 1.5, '#27ae60'),
    ('High\n0.6-0.8', 3.5, '#52be80'),
    ('Medium\n0.3-0.6', 5.5, '#f39c12'),
    ('Low\n0.0-0.3', 7.5, '#e74c3c')
]

for name, x, color in conf_levels:
    box = FancyBboxPatch((x-0.8, y-0.5), 1.6, 0.7,
                        boxstyle="round,pad=0.05", 
                        edgecolor=color, facecolor='white',
                        linewidth=2)
    ax.add_patch(box)
    lines = name.split('\n')
    ax.text(x, y-0.05, lines[0], ha='center', fontsize=9, fontweight='bold')
    ax.text(x, y-0.3, lines[1], ha='center', fontsize=10, color=color, fontweight='bold')

# Example Calculation
y = 1.5
box = FancyBboxPatch((1, y-0.8), 8, 1,
                    boxstyle="round,pad=0.1", 
                    edgecolor='#8e44ad', facecolor='#f4ecf7',
                    linewidth=2)
ax.add_patch(box)
ax.text(5, y+0.1, 'Complete Example', ha='center', 
       fontsize=12, fontweight='bold', color='#2c3e50')
ax.text(5, y-0.15, '4 Indicators: Bullish (0.8), Bullish (0.85), Bullish Broken (0.7), Bullish (0.9)', 
       ha='center', fontsize=8)
ax.text(5, y-0.35, 'Score = (1.0×0.8 + 1.0×0.85 + 0.5×0.7 + 1.0×0.9) / 3.25 = 0.89', 
       ha='center', fontsize=8, style='italic')
ax.text(5, y-0.55, '→ Final: Bullish Strong (0.89) with 81% Accuracy', 
       ha='center', fontsize=10, fontweight='bold', color='#27ae60')

plt.tight_layout()
plt.savefig('indicator_signals.png', dpi=150, bbox_inches='tight')
print("✅ Indicator signals diagram saved as 'indicator_signals.png'")
plt.close()

print("\n📊 Both diagrams created successfully!")
print("  1. trend_analysis_flow.png - Shows the complete analysis flow")
print("  2. indicator_signals.png - Shows signal calculation details")
