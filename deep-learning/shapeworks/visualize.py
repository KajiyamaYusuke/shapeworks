import numpy as np
import matplotlib.pyplot as plt
import glob
import os

print("--- Visualizing Correspondence ---")

# 結果ファイルの場所
particle_dir = "new_project_particles"
files = sorted(glob.glob(os.path.join(particle_dir, "*world.particles")))

if len(files) < 2:
    print("Error: Need at least 2 result files to visualize correspondence.")
    exit()

# データを読み込み
data_list = []
names = []
for f in files:
    data = np.loadtxt(f)
    data_list.append(data)
    # ファイル名から識別用の名前を抽出
    name = os.path.basename(f).replace("_world.particles", "")
    names.append(name)
    print(f"Loaded: {name} ({len(data)} points)")

# --- 可視化 1: 並べて表示 ---
fig = plt.figure(figsize=(12, 6))

# 形状1
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
d1 = data_list[0]
# 色をインデックス順に変えることで「どの点がどこに対応するか」を表現
colors = np.arange(len(d1)) 
ax1.scatter(d1[:,0], d1[:,1], d1[:,2], c=colors, cmap='jet', s=20)
ax1.set_title(names[0])
ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')

# 形状2
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
d2 = data_list[1]
p2 = ax2.scatter(d2[:,0], d2[:,1], d2[:,2], c=colors, cmap='jet', s=20)
ax2.set_title(names[1])
ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')

# 視点を揃える
# データの範囲を取得してスケールを統一
all_pts = np.vstack(data_list)
min_xyz = np.min(all_pts, axis=0)
max_xyz = np.max(all_pts, axis=0)
center = (min_xyz + max_xyz) / 2
range_xyz = np.max(max_xyz - min_xyz) / 2

for ax in [ax1, ax2]:
    ax.set_xlim(center[0]-range_xyz, center[0]+range_xyz)
    ax.set_ylim(center[1]-range_xyz, center[1]+range_xyz)
    ax.set_zlim(center[2]-range_xyz, center[2]+range_xyz)

plt.suptitle("Correspondence Check (Same Color = Same Anatomical Point)", fontsize=14)
plt.savefig("correspondence_check.png")
print("Saved image to 'correspondence_check.png'")

# --- 可視化 2: 差異の計算 (ヒートマップ) ---
# 形状が変わったときに「どこが一番動いたか」を計算
diff = np.linalg.norm(d1 - d2, axis=1) # 対応点ごとの距離
max_diff_idx = np.argmax(diff)

print(f"\nMax difference: {diff[max_diff_idx]:.4f} units")
print(f"At particle index: {max_diff_idx}")

# 差分をヒートマップで表示
fig2 = plt.figure(figsize=(8, 8))
ax3 = fig2.add_subplot(111, projection='3d')
p3 = ax3.scatter(d1[:,0], d1[:,1], d1[:,2], c=diff, cmap='hot', s=30)
fig2.colorbar(p3, label="Displacement Magnitude")
ax3.set_title(f"Shape Difference ({names[0]} vs {names[1]})")

# 視点統一
ax3.set_xlim(center[0]-range_xyz, center[0]+range_xyz)
ax3.set_ylim(center[1]-range_xyz, center[1]+range_xyz)
ax3.set_zlim(center[2]-range_xyz, center[2]+range_xyz)

plt.savefig("shape_difference.png")
print("Saved image to 'shape_difference.png'")
# plt.show()