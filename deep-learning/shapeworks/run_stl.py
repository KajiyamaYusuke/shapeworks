import shapeworks as sw
import os
import glob
import numpy as np

# --- 設定項目 ---
INPUT_DIR = "inputs"             # STLが入っているフォルダ
OUTPUT_DT_DIR = "groomed_data"   # 変換後のデータ(nrrd)の保存先
PARTICLE_DIR = "results_particles" # 結果の保存先

# 画像の解像度（spacing）
# 重要: データの単位（mmかcmか）に合わせて調整が必要です
# 骨などのmm単位なら 0.5 ~ 1.0 くらいが目安
SPACING = [1.0, 1.0, 1.0] 

# パーティクル数
NUM_PARTICLES = 128 

# ----------------------------------------
if hasattr(sw, 'setup_console_logging'):
    sw.setup_console_logging()

print("--- ShapeWorks STL Pipeline ---")

# 1. STLファイルの検索
stl_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.stl")))
if not stl_files:
    print(f"Error: No .stl files found in '{INPUT_DIR}'.")
    print("Please create the folder and put your .stl files there.")
    exit()

print(f"Found {len(stl_files)} STL files.")
os.makedirs(OUTPUT_DT_DIR, exist_ok=True)

# 2. 前処理: STL -> 距離場(NRRD) への変換
# ShapeWorksはメッシュ表面ではなく、距離場（ボリューム）で計算を行います
dt_files = []

print("\n--- Step 1: Converting STLs to Distance Transforms ---")
for stl_path in stl_files:
    # 出力ファイル名を作成
    base_name = os.path.basename(stl_path).replace(".stl", "")
    out_path = os.path.join(OUTPUT_DT_DIR, base_name + ".nrrd")
    
    dt_files.append(out_path)
    
    # 既に変換済みならスキップ（時間短縮）
    if os.path.exists(out_path):
        print(f"Skipping (already exists): {base_name}")
        continue
        
    print(f"Processing: {base_name} ...")
    
    try:
        # メッシュ読み込み
        mesh = sw.Mesh(stl_path)
        
        # --- 重要: 距離場への変換 ---
        # toDistanceTransform(spacing, padding, ...)
        # meshの境界ボックスより少し広めに(padding)領域をとって画像化します
        dt_img = mesh.toDistanceTransform(spacing=SPACING, padding=[10, 10, 10])
        
        # 保存
        dt_img.write(out_path)
        
    except Exception as e:
        print(f"Error converting {stl_path}: {e}")
        # 失敗したファイルはリストから除外
        dt_files.pop()

print(f"\nConversion complete. {len(dt_files)} files ready for optimization.")

# 3. プロジェクト設定
print("\n--- Step 2: Setting up Optimization ---")

# Subjectリストの作成
subjects_list = []
for dt_path in dt_files:
    s = sw.Subject()
    s.set_number_of_domains(1)
    s.set_groomed_filenames([dt_path])
    subjects_list.append(s)

project = sw.Project()
project.set_subjects(subjects_list)

# パラメータ設定
params = sw.Parameters()
params.set("number_of_particles", str(NUM_PARTICLES))
params.set("initial_relative_weighting", "0.05") # 初期は少し弱めに
params.set("relative_weighting", "1.0")
params.set("starting_regularization", "100.0")
params.set("ending_regularization", "1.0")
params.set("iterations_per_split", "100")
params.set("optimization_iterations", "200") # しっかり回す
params.set("verbosity", "0") # ログは控えめに

project.set_parameters("optimize", params)

# 4. 最適化実行
# 出力先をPython側で指定してあげると確実です
# (ライブラリのバージョンによっては自動でフォルダを作らない場合があるため)
if not os.path.exists(PARTICLE_DIR):
    os.makedirs(PARTICLE_DIR)
# Projectクラスのヘッダ設定等が必要な場合がありますが、
# ここではデフォルト出力(カレントディレクトリのプロジェクトフォルダ)に任せ、
# 後でファイルを移動/読み込みます。

print("Initializing Optimizer...")
opt = sw.Optimize()
opt.SetUpOptimize(project)

print("Running Optimization...")
opt.Run()
print("Optimization Finished.")

# 5. 結果の確認（ファイルから読み込み）
print("\n--- Step 3: Checking Results ---")

# デフォルトで生成されるフォルダを探す
result_folder = "new_project_particles" # バージョンによるが通常はこれ
target_files = sorted(glob.glob(os.path.join(result_folder, "*world.particles")))

if target_files:
    print(f"Found {len(target_files)} result files in {result_folder}.")
    
    # 最初のデータの座標を表示
    data = np.loadtxt(target_files[0])
    print(f"Sample Result ({os.path.basename(target_files[0])}):")
    print(f"  Total Particles: {len(data)}")
    print(f"  First Point: {data[0]}")
    
    # 目的関数値（エントロピー）の簡易計算
    all_data = []
    for f in target_files:
        all_data.append(np.loadtxt(f).flatten())
    
    if len(all_data) > 1:
        Z = np.array(all_data)
        Z_centered = Z - np.mean(Z, axis=0)
        cov = np.dot(Z_centered, Z_centered.T) / (len(all_data)-1)
        eig = np.linalg.eigvalsh(cov)
        valid_eig = eig[eig > 1e-10]
        entropy = np.sum(np.log(valid_eig))
        print(f"  System Entropy: {entropy:.4f}")
    else:
        print("  (Need more than 1 subject to calculate group entropy)")

else:
    print("Warning: Could not find result files automatically.")
    print("Check the current directory for a folder named 'new_project_particles'.")