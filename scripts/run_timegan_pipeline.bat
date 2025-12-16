@echo off
REM Complete TimeGAN pipeline: create stress dataset -> train -> sample
echo ============================================================
echo Step 1: Creating stress-filtered dataset (VIX >= 20)
echo ============================================================
python -c "from src.data.preprocessing import make_sp500_stress_windows; make_sp500_stress_windows(seq_len=50, vix_threshold=20)"

echo.
echo ============================================================
echo Step 2: Training TimeGAN on stress data
echo ============================================================
python src/training/train_baseline_timegan.py --dataset_name sp500_logret_stress20 --run_name timegan_stress20_L50 --seq_len 50 --hidden_dim 24 --latent_dim 24 --num_layers 3 --embedding_epochs 200 --supervisor_epochs 200 --joint_epochs 200 --lr 1e-3 --batch_size 128 --gamma 1.0

echo.
echo ============================================================
echo Step 3: Generating stress scenarios
echo ============================================================
python scripts/sample_timegan.py --timegan_ckpt experiments/checkpoints/timegan_stress20_L50/best_timegan.pt --dataset_name sp500_logret_stress20 --num_samples 16 --stress_scale 1.5 --temperature 1.0 --save_dir experiments/results/timegan_stress_L50

echo.
echo ============================================================
echo Pipeline complete! Check experiments/results/timegan_stress_L50/
echo ============================================================
