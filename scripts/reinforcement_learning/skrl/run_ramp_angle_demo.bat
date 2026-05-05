@echo off
REM =============================================================
REM  Ramp Angle Demo — sweeps test_angle from -12 to +12 degrees
REM  Each iteration launches ramp_demo.py with a different angle.
REM =============================================================

setlocal enabledelayedexpansion

REM --- Configurable parameters (edit as needed) ----------------
set CHECKPOINT=C:\Users\bates\IsaacLab\logs\skrl\humanoid_amp_im_walk_v2\2026-04-27_20-16-13_amp_torch_no_curriculum_no_decay_noAMP_noImitation\checkpoints\best_agent.pt
set TASK=Isaac-Humanoid-AMP-Imp-Direct-v0
set NUM_ENVS=1
set ALGORITHM=AMP
set ANGLE_START=-15
set ANGLE_END=15
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

