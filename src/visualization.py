import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Project root directory
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Load the Top-5 features JSON
with open(os.path.join(base, 'models', 'top5_features.json')) as f:
    top5 = json.load(f)

# Prepare radar chart labels
labels = sorted({feat for feats in top5.values() for feat in feats})
N = len(labels)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# Create figure and polar axis
fig, ax = plt.subplots(subplot_kw={'polar': True}, figsize=(8,8))
for role, feats in top5.items():
    # Assign value 5 if feature is in top-5, else 0
    vals = [(5 if lab in feats else 0) for lab in labels]
    vals += vals[:1]
    ax.plot(angles, vals, label=role)
    ax.fill(angles, vals, alpha=0.25)

# Configure axis ticks and labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=9)
ax.set_yticklabels([])
ax.set_title("Top 5 Features by Role", pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# Save the figure
os.makedirs(os.path.join(base, 'figures', 'radar'), exist_ok=True)
plt.tight_layout()
plt.savefig(os.path.join(base, 'figures', 'radar', 'radar_top5.png'), bbox_inches='tight')
plt.show()