import json
import numpy as np
import matplotlib.pyplot as plt


with open('/models/heloc_lgbm_error_analysis.json', 'r') as f:
    errors = json.load(f)

confidences = [error['confidence'] for error in errors]

print(f"number of errors: {len(confidences)}")
print(f"mean confidence: {np.mean(confidences):.4f}")
print(f"median confidence: {np.median(confidences):.4f}")
print(f"confidence range: {np.min(confidences):.4f} - {np.max(confidences):.4f}")

high_conf = sum(1 for c in confidences if c > 0.8)
medium_conf = sum(1 for c in confidences if 0.6 <= c <= 0.8)
low_conf = sum(1 for c in confidences if c < 0.6)

print(f"High confidence errors (>0.8): {high_conf}")
print(f"Medium confidence errors (0.6-0.8): {medium_conf}")  
print(f"Low confidence errors (<0.6): {low_conf}")

plt.hist(confidences, bins=10)
plt.title('Error Sample Confidence Distribution')
plt.show()