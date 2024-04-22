##用于存放部分关键参数
# 数据参数维度数
num_item = 29
# 客户端数量（需要数据分割的份数）
num_user = 10
# 进行数据提取的文件路径
File_Name = 'Data/data_map10/creditcard1_train.csv'
# 少数类样本名称
File_Less = 'Data_Less.csv'
# 数据进行SMOTE过采样后的负样本文件名称
File_Smote = 'dataProcess/Data_Smote.csv'
# 数据在过采样后合并得到的数据集名称
File_Merge_Smote = 'dataProcess/smote.csv'
# 打乱顺序后的文件路径
File_Upset = 'dataProcess/Data_Upset.csv'
# 分割后训练集路径
File_Train = 'dataProcess/Data_Train.csv'
# 分割后测试集路径
File_Check = 'dataProcess/Data_Test.csv'
# 训练集比例
Train_Set = 0.8
User_w = ['0.01']
# 初始权重值
base_score = 0.5

