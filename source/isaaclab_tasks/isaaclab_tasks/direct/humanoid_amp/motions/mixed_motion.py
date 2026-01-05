import numpy as np
import os

def merge_motion_files(file_list, output_file):
    """
    Merges multiple .npz motion files into a single file for Isaac Lab/AMP.
    """
    if not file_list:
        print("No files provided.")
        return

    # 1. Load the first file to establish the baseline (FPS, names, etc.)
    print(f"Loading base file: {file_list[0]}")
    base_data = np.load(file_list[0])
    second_data = np.load(file_list[1])

    second_fps = second_data['fps']
    second_dof_names = second_data['dof_names']
    second_body_names = second_data['body_names']
    # Store metadata that must be identical across files
    fps = base_data['fps']
    dof_names = base_data['dof_names']
    body_names = base_data['body_names']
    
    if fps != second_fps:
        raise ValueError(f"FPS mismatch in {file_list[0]} and {file_list[1]}")

    # Initialize lists to hold the concatenated data
    # We start with the data from the first file
    merged_dof_pos = [base_data['dof_positions']]
    merged_dof_vel = [base_data['dof_velocities']]
    merged_body_pos = [base_data['body_positions']]
    merged_body_rot = [base_data['body_rotations']]
    merged_body_lin_vel = [base_data['body_linear_velocities']]
    merged_body_ang_vel = [base_data['body_angular_velocities']]

    total_frames = base_data['dof_positions'].shape[0]

    # 2. Iterate through the rest of the files
    for file_path in file_list[1:]:
        print(f"Merging file: {file_path}")
        if not os.path.exists(file_path):
            print(f"Error: File not found {file_path}")
            continue
            
        data = np.load(file_path)

        # Validation checks
        if data['fps'] != fps:
            print(f"WARNING: FPS mismatch in {file_path}. Base: {fps}, Current: {data['fps']}")
            # Ideally resample here, but for now we skip or warn
        
        if not np.array_equal(data['dof_names'], dof_names):
            raise ValueError(f"DOF Names do not match in {file_path}")

        # Append data
        merged_dof_pos.append(data['dof_positions'])
        merged_dof_vel.append(data['dof_velocities'])
        merged_body_pos.append(data['body_positions'])
        merged_body_rot.append(data['body_rotations'])
        merged_body_lin_vel.append(data['body_linear_velocities'])
        merged_body_ang_vel.append(data['body_angular_velocities'])
        
        total_frames += data['dof_positions'].shape[0]

    # 3. Concatenate everything along the time axis (axis 0)
    final_data = {
        'fps': fps,
        'dof_names': dof_names,
        'body_names': body_names,
        'dof_positions': np.concatenate(merged_dof_pos, axis=0),
        'dof_velocities': np.concatenate(merged_dof_vel, axis=0),
        'body_positions': np.concatenate(merged_body_pos, axis=0),
        'body_rotations': np.concatenate(merged_body_rot, axis=0),
        'body_linear_velocities': np.concatenate(merged_body_lin_vel, axis=0),
        'body_angular_velocities': np.concatenate(merged_body_ang_vel, axis=0)
    }

    # 4. Save to new file
    np.savez(output_file, **final_data)
    print(f"\nSuccess! Merged file saved to: {output_file}")
    print(f"Total frames: {total_frames}")

if __name__ == "__main__":
    # --- CONFIGURE YOUR PATHS HERE ---
    
    # List the paths to your individual motion files
    input_files = [
        "C:/Users/bates/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp/motions/humanoid_walk.npz",
        "C:/Users/bates/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp/motions/humanoid_run.npz" 
    ]
    
    # Output path
    output_path = "C:/Users/bates/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp/motions/humanoid_mixed.npz"
    
    merge_motion_files(input_files, output_path)