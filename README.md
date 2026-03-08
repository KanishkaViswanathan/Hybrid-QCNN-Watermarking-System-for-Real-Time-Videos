python -m venv venv

pip install torch torchvision torchaudio opencv-python pillow matplotlib tqdm scikit-image pennylane

python 01_make_watermark.py 

python 00_extract_frames.py

python train_hybrid_semifragile.py

python eval_hybrid_semifragile_metrics.py 
