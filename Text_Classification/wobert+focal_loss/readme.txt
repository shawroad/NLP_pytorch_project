focal loss参考:https://zhuanlan.zhihu.com/p/75542467

对于focal loss中两个超参数的设置:  
       alpha: 假设二分类中 正样本:负样本=2:8  我们就可以对正样本的alpha设小一点，即 放大正样本的loss 
	   gamma: 也是放大loss  这里采用默认的2

python data_process.py进行数据处理
python run_classify.py  进行分类模型的训练