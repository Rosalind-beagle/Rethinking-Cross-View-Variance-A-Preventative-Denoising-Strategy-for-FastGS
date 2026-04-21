CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=toaster python train.py -s ./datasets/shinyblender/toaster --eval --densification_interval 500  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.042 --grad_abs_thresh 0.0015 --dense 0.01 --mult 0.7
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=ball python train.py -s ./datasets/shinyblender/ball --eval --densification_interval 500  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.042 --grad_abs_thresh 0.0015 --dense 0.01 --mult 0.7
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=car python train.py -s ./datasets/shinyblender/car --eval --densification_interval 500  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.042 --grad_abs_thresh 0.0015 --dense 0.01 --mult 0.7
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=coffee python train.py -s ./datasets/shinyblender/coffee --eval --densification_interval 500  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.042 --grad_abs_thresh 0.0015 --dense 0.01 --mult 0.7
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=helmet python train.py -s ./datasets/shinyblender/helmet --eval --densification_interval 500  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.042 --grad_abs_thresh 0.0015 --dense 0.01 --mult 0.7
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=teapot python train.py -s ./datasets/shinyblender/teapot --eval --densification_interval 500  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.042 --grad_abs_thresh 0.0015 --dense 0.01 --mult 0.7

CUDA_VISIBLE_DEVICES=0 python render.py -m output/toaster --skip_train --mult 0.7
CUDA_VISIBLE_DEVICES=0 python render.py -m output/ball --skip_train --mult 0.7
CUDA_VISIBLE_DEVICES=0 python render.py -m output/car --skip_train --mult 0.7
CUDA_VISIBLE_DEVICES=0 python render.py -m output/coffee --skip_train --mult 0.7
CUDA_VISIBLE_DEVICES=0 python render.py -m output/helmet --skip_train --mult 0.7
CUDA_VISIBLE_DEVICES=0 python render.py -m output/teapot --skip_train --mult 0.7

CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/toaster
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/ball
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/car
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/coffee
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/helmet
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/teapot