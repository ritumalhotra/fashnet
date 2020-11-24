export CUDA_VISIBLE_DEVICES="cpu"
export IMG_HEIGHT=225
export IMG_WIDTH=225
export EPOCHS=50
export TRAIN_BATCH_SIZE=64
export TEST_BATCH_SIZE=8
export MODEL_MEAN="(0.485, 0.456, 0.406)"
export MODEL_STD="(0.229, 0.224, 0.225)"
export BASE_MODEL="resnet18"
#TODO(Sayar): Make this generic
export TRAINING_FOLDS_CSV="/Users/Banner/Downloads/train_full.csv"

export TRAINING_FOLDS="(0,1,2,3)"
export VALIDATION_FOLDS="(4,)"
python3 train_model.py


# export TRAINING_FOLDS="(0,1,2,4)"
# export VALIDATION_FOLDS="(3,)"
# python3 train_model.py

# export TRAINING_FOLDS="(0,1,4,3)"
# export VALIDATION_FOLDS="(2,)"
# python3 train_model.py

# export TRAINING_FOLDS="(0,4,2,3)"
# export VALIDATION_FOLDS="(1,)"
# python3 train_model.py

# export TRAINING_FOLDS="(4,1,2,3)"
# export VALIDATION_FOLDS="(0,)"
# python3 train_model.py
