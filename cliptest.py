import numpy as np
path = r"C:\Users\bates\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\humanoid_amp\motions\humanoid_walk.npz"
d = np.load(path, allow_pickle=True)

print("Keys:", list(d.keys()))
print("FPS:", float(d["fps"]))
print("DoF names:", list(d["dof_names"]))
print("Body names:", list(d["body_names"]))

n_frames = d["dof_positions"].shape[0]
fps = float(d["fps"])
print(f"Number of frames: {n_frames}")
print(f"Duration (s): {n_frames / fps:.4f}")
print(f"dt (s): {1.0 / fps:.6f}")

for k in ["dof_positions", "dof_velocities", "body_positions",
          "body_rotations", "body_linear_velocities", "body_angular_velocities"]:
    print(f"{k}: shape={d[k].shape}, dtype={d[k].dtype}")