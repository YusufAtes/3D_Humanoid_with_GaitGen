@echo off
REM ==================================================================
REM  Noisy Plane Demo — sweeps noise amplitude from 0.01 to 0.10 m
REM  for BOTH noise types: random and wave.
REM ==================================================================

setlocal enabledelayedexpansion

REM --- Configurable parameters (edit as needed) ---------------------
set CHECKPOINT=c:/Users/bates/IsaacLab/logs/skrl/humanoid_direct/2026-02-18_10-25-04_ppo_torch/checkpoints/best_agent.pt
set TASK=Isaac-Humanoid-Direct-v0
set NUM_ENVS=1
set ALGORITHM=PPO
REM ------------------------------------------------------------------

echo ==================================================================
echo  Noisy Plane Demo: sweeping amplitude 0.01 to 0.10 m
echo  Noise types: random, wave
echo ==================================================================

for %%T in (random wave) do (
    echo.
    echo ==================================================================
    echo  Noise type: %%T
    echo ==================================================================

    for %%A in (0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 0.11 0.12 0.13 0.14 0.15) do (
        echo.
        echo ==================================================================
        echo  Running noisy_plane_demo.py  noise_type=%%T  amplitude=%%A m
        echo ==================================================================
        call .\isaaclab.bat -p scripts\reinforcement_learning\skrl\noisy_plane_demo.py ^
            --checkpoint %CHECKPOINT% ^
            --task %TASK% ^
            --num_envs %NUM_ENVS% ^
            --algorithm %ALGORITHM% ^
            --noise_amplitude %%A ^
            --noise_type %%T ^
            --headless

        if errorlevel 1 (
            echo [WARN] noisy_plane_demo.py exited with error for type %%T amplitude %%A
        )
    )
)

echo.
echo ==================================================================
echo  All noise types and amplitudes completed!
echo ==================================================================
endlocal
