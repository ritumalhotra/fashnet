export TEST_BATCH_SIZE=32
export MODEL_MEAN="(0.485, 0.456, 0.406)"
export MODEL_STD="(0.229, 0.224, 0.225)"
export IMG_HEIGHT=225
export IMG_WIDTH=225
export DEVICE="cpu"

python3 predict.py