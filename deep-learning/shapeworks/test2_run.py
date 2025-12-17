import shapeworks as sw
import numpy as np
import os
import glob

# ★ここが重要：C++のログを画面に出すスイッチ
if hasattr(sw, 'setup_console_logging'):
    sw.setup_console_logging()

print("--- ShapeWorks Optimization (Log Fix) ---")

# --- 1. データ生成 (省略可) ---
output_dir = "test_data"
os.makedirs(output_dir, exist_ok=True)
files = []
try:
    dim = 50
    x, y, z = np.ogrid[-dim//2:dim//2, -dim//2:dim//2, -dim//2:dim//2]
    for i, radius in enumerate([10.0, 12.0]): 
        filename = os.path.join(output_dir, f"sphere_{i}.nrrd")
        files.append(filename)
        if not os.path.exists(filename):
            mask = (x**2 + y**2 + z**2) <= radius**2
            arr = mask.astype(np.float32)
            img = sw.Image(arr)
            img.setSpacing([1.0, 1.0, 1.0])
            img.setOrigin([-dim/2, -dim/2, -dim/2])
            dt = img.computeDT(0.5) 
            dt.write(filename)
except: pass

# --- 2. プロジェクト設定 ---
subjects_list = []
for filename in files:
    s = sw.Subject()
    s.set_number_of_domains(1)
    s.set_groomed_filenames([filename])
    subjects_list.append(s)

project = sw.Project()
project.set_subjects(subjects_list)

params = sw.Parameters()
# パラメータ設定
params.set("number_of_particles", "32") # テスト用に32に戻しました
params.set("initial_relative_weighting", "1.0")
params.set("relative_weighting", "1.0")
params.set("starting_regularization", "10.0")
params.set("ending_regularization", "1.0")
params.set("iterations_per_split", "50")
params.set("optimization_iterations", "50")

# ★ここも "3" になっているか再確認
params.set("verbosity", "3")

project.set_parameters("optimize", params)

# --- 3. 最適化実行 ---
print("Initializing Optimizer...")
opt = sw.Optimize()
opt.SetUpOptimize(project)

print("Running Optimization (Logs should appear below)...")
print("-" * 30)

# ここでログが滝のように流れるはずです
opt.Run()

print("-" * 30)
print("Optimization finished.")