device=$1
xp_dir=../log/$2

echo "Using device" $device
echo "Writing to directory: " $xp_dir

mkdir $xp_dir;
> $xp_dir/log_mnist.txt;

# Baselines

# MNIST Adagrad training
python baseline.py --dataset mnist --solver adagrad \
    --lr 0.01 --n_epochs 500 --save_at 200 --loss svm --device $device --xp_dir $xp_dir;

# MNIST Adadelta training
python baseline.py --dataset mnist --solver adadelta \
--lr 1.0 --n_epochs 500 --save_at 100 --loss svm --device $device --xp_dir $xp_dir;

# MNIST Adam training
python baseline.py --dataset mnist --solver adam \
    --lr 0.001 --n_epochs 500 --save_at 100 --loss svm --device $device --xp_dir $xp_dir;

# LW-SVM

# MNIST Adagrad tuning
python lwsvm.py --dataset mnist --in_name mnist_adagrad_svm --device $device --xp_dir $xp_dir;

# MNIST Adadelta tuning
python lwsvm.py --dataset mnist --in_name mnist_adadelta_svm --device $device --xp_dir $xp_dir;

# MNIST Adam tuning
python lwsvm.py --dataset mnist --in_name mnist_adam_svm --device $device --xp_dir $xp_dir;


# extract results
python extract.py --xp_dir $xp_dir --device $device;

# plot results and save
python visualize.py --xp_dir $xp_dir --export_pdf $xp_dir/plot.pdf;
