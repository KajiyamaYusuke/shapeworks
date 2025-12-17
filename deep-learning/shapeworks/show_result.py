import matplotlib.pyplot as plt
import glob
import numpy as np
import os

# 結果ファイルの場所
particle_dir = "new_project_particles"
files = sorted(glob.glob(os.path.join(particle_dir, "*world.particles")))

fig = plt.figure(figsize=(10, 5))

# 2つの球体を並べて表示
for i, p_file in enumerate(files):
    ax = fig.add_subplot(1, 2, i+1, projection='3d')
    data = np.loadtxt(p_file)
    
    # 3D散布図
    ax.scatter(data[:,0], data[:,1], data[:,2], c='r', marker='o')
    
    # 見た目の調整
    ax.set_title(f"Subject {i}\n({len(data)} particles)")
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    
    # スケールを揃える（球体に見えるように）
    limit = 15
    ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit); ax.set_zlim(-limit, limit)

plt.tight_layout()
plt.show()
# ※WSLで画面が出ない場合は plt.savefig("result.png") に変えてください