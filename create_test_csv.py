import pandas as pd
import random
from pathlib import Path

# Point this at your test_images folder.
folder = Path("test_images2")

# Find all image files.
extensions = [".jpg", ".jpeg", ".png", ".JPG", ".PNG", ".JPEG"]
image_files = [
    f.name for f in folder.iterdir()
    if f.suffix in extensions
]

# Randomly assign metadata.
random.seed(42)  # Fixed seed so results are reproducible.

rows = []
for filename in sorted(image_files):
    rows.append({
        "file_name":  filename,
        "session_id": random.choice(["S01", "S02", "S03"]),
        "trial_id":   f"T{random.randint(1, 20):02d}",
        "condition":  random.choice(["positive", "negative", "neutral"]),
        "timestamp":  round(random.uniform(0.0, 30.0), 3),
    })

df = pd.DataFrame(rows)
output_path = folder / "metadata.csv"
df.to_csv(output_path, index=False)
print(f"Created {output_path} with {len(rows)} rows.")
print(df.head())