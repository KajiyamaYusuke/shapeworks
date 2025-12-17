import shapeworks as sw
import numpy as np
import os
import glob

print("--- ShapeWorks Optimization (File Reading Version) ---")

# --- 1. データ生成 & 2. プロジェクト設定 ---
# (ここは変更なし。前のコードで既に計算済みならコメントアウトしても動きますが、念のため記載)
output_dir = "test_data"
os.makedirs(output_dir, exist_ok=True)
dim = 50
x, y, z = np.ogrid[-dim//2:dim//2, -dim//2:dim//2, -dim//2:dim//2]

files = []
try:
    for i, radius in enumerate([10.0, 12.0]): 
        filename = os.path.join(output_dir, f"sphere_{i}.nrrd")
        files.append(filename)
        if os.path.exists(filename): continue
        
        mask = (x**2 + y**2 + z**2) <= radius**2
        arr = mask.astype(np.float32)
        img = sw.Image(arr)
        img.setSpacing([1.0, 1.0, 1.0])
        img.setOrigin([-dim/2, -dim/2, -dim/2])
        dt = img.computeDT(0.5) 
        dt.write(filename)
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
params.set("initial_relative_weighting", "1.0")
params.set("relative_weighting", "1.0")
params.set("starting_regularization", "10.0")
params.set("ending_regularization", "1.0")
params.set("iterations_per_split", "50")
params.set("optimization_iterations", "50")
params.set("verbosity", "0") # ログを少し静かにします

project.set_parameters("optimize", params)

# --- 3. 最適化実行 ---
# 出力先フォルダをPython側で把握しておきます
# デフォルトではカレントディレクトリに "new_project_particles" が作られます
output_particles_dir = "new_project_particles"

print("Running Optimization...")
opt = sw.Optimize()
opt.SetUpOptimize(project)
opt.Run()
print("Optimization finished.")

# --- 4. 結果確認 (ファイルから読み込み) ---
print(f"\nLooking for results in: {output_particles_dir}")

# .particles ファイルを探します (通常は world.particles や local.particles)
# ファイル名は sphere_0_world.particles のようになることが多いです
particle_files = sorted(glob.glob(os.path.join(output_particles_dir, "*world.particles")))

if not particle_files:
    # パターンが違うかもしれないので、すべての .particles を探してみる
    particle_files = sorted(glob.glob(os.path.join(output_particles_dir, "*.particles")))

if not particle_files:
    print("[Warning] No .particles files found. Check the folder manually.")
    print(f"Contents of {output_particles_dir}:")
    if os.path.exists(output_particles_dir):
        print(os.listdir(output_particles_dir))
else:
    print(f"Found {len(particle_files)} particle files.")

    for i, p_file in enumerate(particle_files):
        # .particlesファイルはテキスト形式で、各行に x y z が書かれています
        try:
            data = np.loadtxt(p_file)
            print(f"Subject {i} (from {os.path.basename(p_file)}):")
            print(f"  Count: {len(data)}")
            print(f"  Sample: {data[0]}") # 最初の座標
        except Exception as e:
            print(f"  Could not read file {p_file}: {e}")

print("\nDONE! You can now analyze these coordinates.")