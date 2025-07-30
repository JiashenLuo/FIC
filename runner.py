import os

os.system("python train.py --model_name=fic --data_name=ml_100k --params neg_c=7")
os.system("python train.py --model_name=fic --data_name=ml_1m --params neg_c=10")
os.system("python train.py --model_name=fic --data_name=douban_book --params neg_c=51")
os.system("python train.py --model_name=fic --data_name=yelp2018 --params neg_c=42")
os.system("python train.py --model_name=fic --data_name=gowalla --params neg_c=100")