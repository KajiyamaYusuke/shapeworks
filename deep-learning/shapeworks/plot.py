import shapeworks as sw
import numpy as np
import os

print("--- ShapeWorks Energy Calculation ---")

# 1. データ準備 & 最適化実行（前回と同じ設定）
# ------------------------------------------------
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
            img = sw.Image(mask.astype(np.float32))
            img.setSpacing([1.0, 1.0, 1.0])
            img.setOrigin([-dim/2, -dim/2, -dim/2])
            img.computeDT(0.5).write(filename)
except: pass

subjects_list = []
for filename in files:
    s = sw.Subject()
    s.set_number_of_domains(1)
    s.set_groomed_filenames([filename])
    subjects_list.append(s)

project = sw.Project()
project.set_subjects(subjects_list)

params = sw.Parameters()
params.set("number_of_particles", "64")
params.set("optimization_iterations", "50") # テスト用
params.set("verbosity", "0") # ログは出ないので0にしておく

project.set_parameters("optimize", params)

print("Running Optimization...")
opt = sw.Optimize()
opt.SetUpOptimize(project)
opt.Run()
print("Optimization Finished.")

# 2. 目的関数（エントロピー）の計算
# ------------------------------------------------
print("\n--- Calculating Final Objective Function Value ---")

# (1) 結果のパーティクルを収集
# 形状行列 Z (NumSubjects x NumDimensions) を作ります
# NumDimensions = パーティクル数 x 3
z_matrix = []

# 結果ファイルから読み込み（確実に値を取るため）
import glob
particle_files = sorted(glob.glob("new_project_particles/*world.particles"))
if not particle_files:
    print("Error: Particles not found.")
    exit()

for p_file in particle_files:
    # 座標を読み込んで1列に並べる (Flatten)
    pts = np.loadtxt(p_file) # (64, 3)
    z_matrix.append(pts.flatten()) # (192,)

Z = np.array(z_matrix) # (2, 192)
N_subjects = Z.shape[0]

print(f"Data Matrix Shape: {Z.shape} ({N_subjects} subjects)")

# (2) 共分散行列の固有値を計算
# ShapeWorksは、この「形状のばらつき」のエントロピーを最小化しています
# 共分散行列 Cov = (Z - mean).T @ (Z - mean) / (N-1)
# 計算効率のため、次元が低い方（被験者数Nか、次元数Dか）で計算します
Z_centered = Z - np.mean(Z, axis=0)
if Z.shape[0] < Z.shape[1]:
    # Dual Covariance (N x N)
    cov = np.dot(Z_centered, Z_centered.T) / (N_subjects - 1)
else:
    cov = np.dot(Z_centered.T, Z_centered) / (N_subjects - 1)

# 固有値分解
eigenvalues = np.linalg.eigvalsh(cov)

# 小さすぎる固有値はノイズまたは数値誤差なので、計算上の安定化のためカットまたは微小値を足す
# ShapeWorks内部でも同様の安定化処理が入っています
eigenvalues = eigenvalues[eigenvalues > 1e-10]

print(f"Eigenvalues: {eigenvalues}")

# (3) エントロピー（Energy）の算出
# H = Sum( log(lambda) )
# これが、ShapeWorksが下げようとしていた「値」の正体です
entropy = np.sum(np.log(eigenvalues))

print("-" * 30)
print(f"Final Objective Function Value (Entropy): {entropy:.5f}")
print("-" * 30)

# おまけ: 平均二乗誤差（Compactness）
compactness = np.sum(eigenvalues)
print(f"(Reference) Compactness (Sum of variance): {compactness:.5f}")