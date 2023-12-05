
import pandas as pd

def p1_2_1(data_df, col, year):
    p1=data_df.sort_values(by=col, ascending=False)
    p1=p1[p1['year']==year]
    p1=p1[[col,'year']]
    p1=p1.head(10)
    return p1

def p2_2_1(data_df, col, position):
    p1=data_df.sort_values(by=col, ascending=False)
    p1=p1[p1['cp']==position]
    p1=p1[[col, 'cp']]
    p1=p1.head(10)
    return p1

data_df=pd.read_csv("2019_kbo_for_kaggle_v2.csv")
num=2015
for i in range(4):
    print(p1_2_1(data_df, 'H', num))
    num=num+1

arr=['포수', '1루수', '2루수', '3루수', '유격수', '좌익수', '중견수', '우익수']

for i in range(8):
    print(p2_2_1(data_df, 'war', arr[i]))


temp=data_df[['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG', 'salary']]
print(temp.corrwith(temp.salary)) #RBI가 0.547702로 가장 상관관계가 높음.