@echo off
REM =============================================================
REM  Ramp Angle Demo — sweeps test_angle from -12 to +12 degrees
REM  Each iteration launches ramp_demo.py with a different angle.
REM =============================================================

setlocal enabledelayedexpansion

REM --- Configurable parameters (edit as needed) ----------------
set CHECKPOINT=c:/Users/bates/IsaacLab/logs/skrl/humanoid_direct/2026-02-18_10-25-04_ppo_torch/checkpoints/best_agent.pt
set TASK=Isaac-Humanoid-Direct-v0
set NUM_ENVS=1
set ALGORITHM=PPO
set ANGLE_START=-12
set ANGLE_END=12
set ANGLE_STEP=1
REM -------------------------------------------------------------

echo ============================================================
echo  Ramp Angle Demo: sweeping %ANGLE_START% to %ANGLE_END% degrees
echo ============================================================

for /L %%A in (%ANGLE_START%, %ANGLE_STEP%, %ANGLE_END%) do (
    echo.
    echo ============================================================
    echo  Running ramp_demo.py with test_angle = %%A degrees
    echo ============================================================
    call .\isaaclab.bat -p scripts\reinforcement_learning\skrl\ramp_demo.py ^
        --checkpoint %CHECKPOINT% ^
        --task %TASK% ^
        --num_envs %NUM_ENVS% ^
        --algorithm %ALGORITHM% ^
        --test_angle %%A ^
        --headless

    if errorlevel 1 (
        echo [WARN] ramp_demo.py exited with error for angle %%A
    )
)

echo.
echo ============================================================
echo  All angles completed!
echo ============================================================
endlocal

