# CUDA_VISIBLE_DEVICES=6 python continual_train_noisy.py --logs-dir release_test/pattern-0.3_3407  --noise_ratio 0.3 --noise pattern --seed 3407
CUDA_VISIBLE_DEVICES=6 python continual_train_noisy.py --logs-dir release_test/pattern-0.2  --noise_ratio 0.2 --noise pattern 
CUDA_VISIBLE_DEVICES=6 python continual_train_noisy.py --logs-dir release_test/pattern-0.1  --noise_ratio 0.1 --noise pattern 

