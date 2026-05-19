@echo off
REM ==================================================================
REM  Noisy Plane Demo sweep
REM    noise_type:   random, wave
REM    seed:         42, 1357
REM    amplitude:    0.01 .. 0.10 m
REM    downsampled_scale: 0.10, 0.50, 1.00  (random only; wave ignores it)
REM ==================================================================

setlocal enabledelayedexpansion

REM --- Configurable parameters --------------------------------------
set CHECKPOINT=C:\Users\bates\IsaacLab\logs\skrl\humanoid_amp_im_walk_v2\2026-05-18_10-54-15_amp_torch_hardcurriculum_no_decay\checkpoints\best_agent.pt
set TASK=Isaac-Humanoid-AMP-Imp-Direct-v0
set NUM_ENVS=1
set ALGORITHM=AMP
REM ------------------------------------------------------------------

echo ==================================================================
echo  Noisy Plane Demo: seeds {42,1357} x amps {0.01..0.10}
echo  Random also sweeps downsampled_scale {0.10, 0.50, 1.00}
echo ==================================================================

REM --- RANDOM noise: sweep seed x downsampled_scale x amplitude -----
for %%S in (42 1357 7919) do (
    for %%N in (0.10 0.50 1.00) do (
        for %%A in (0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10) do (
            echo.
            echo ==================================================================
            echo  RANDOM  seed=%%S  downsampled_scale=%%N  amplitude=%%A m
            echo ==================================================================
            call .\isaaclab.bat -p scripts\reinforcement_learning\skrl\noisy_plane_demo.py ^
                --checkpoint %CHECKPOINT% ^
                --task %TASK% ^
                --num_envs %NUM_ENVS% ^
                --algorithm %ALGORITHM% ^
                --noise_type random ^
                --noise_seed %%S ^
                --downsampled_scale %%N ^
                --noise_amplitude %%A ^
                --headless

            if errorlevel 1 (
                echo [WARN] random run failed: seed=%%S downsampled_scale=%%N amp=%%A
            )
        )
    )
)

REM --- WAVE noise: sweep seed x amplitude (no downsampled_scale) ----
for %%S in (42 1357 7919) do (
    for %%A in (0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10) do (
        echo.
        echo ==================================================================
        echo  WAVE  seed=%%S  amplitude=%%A m
        echo ==================================================================
        call .\isaaclab.bat -p scripts\reinforcement_learning\skrl\noisy_plane_demo.py ^
            --checkpoint %CHECKPOINT% ^
            --task %TASK% ^
            --num_envs %NUM_ENVS% ^
            --algorithm %ALGORITHM% ^
            --noise_type wave ^
            --noise_seed %%S ^
            --noise_amplitude %%A ^
            --headless

        if errorlevel 1 (
            echo [WARN] wave run failed: seed=%%S amp=%%A
        )
    )
)

echo.
echo ==================================================================
echo  All sweeps completed!
echo ==================================================================
endlocal