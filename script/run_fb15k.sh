
python src/pipeline.py --dataset fb15k --model transe --negatives 10 --sampler relational --lr 0.01 --l2 0.0001863 --margin 1
python src/pipeline.py --dataset fb15k --model transh --negatives 10 --sampler relational --lr 0.01 --l2 0.0001863 --margin 1
python src/pipeline.py --dataset fb15k --model transr --negatives 10 --sampler relational --lr 0.01 --l2 0.0001863 --margin 1

python src/pipeline.py --dataset fb15k --model transh --negatives 100 --sampler relational --lr 0.01 --l2 0.0001863 --margin 1
python src/pipeline.py --dataset fb15k --model transr --negatives 100 --sampler relational --lr 0.01 --l2 0.0001863 --margin 1




