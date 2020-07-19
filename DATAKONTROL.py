## buraya bakarak yapıldı : https://www.kaggle.com/retroflake/classification-of-car-s-acceptability

## Sample alıp overview görmek için :
data.sample(10)

## Row ve Column sayısı öğrenmek için :
data.shape

## Bir column değeri üzerinden data'yı filtreleyip yeni bir dataframe oluşturmak için
a_df=[]
for i in data.values:
    if i[6] == 'yes':
        a_df.append(i)
yesvoters = pd.DataFrame(a_df)

## Missing Value olup olmadığını kontrol etmek için :
data.isnull().sum()

## Analitik açıdan data'ya bakmak için; count, unique, top, freq :
data.describe

## Mesela 'Vote' column'undaki AKP, CHP, MHP'ye oy verenlerin sayısını görmek için :
data['Vote'].value_counts().sort_index()

## Mesela bir arabanın price'ının pahalı olup olmamasının nümerik hale getirilmesi :
data.Price.replace(('very_high','high','medium','low'),(3,2,1,0), inplace = True)

## Araba'nın Price'ı ve Evalution'ın value'lerindeki kesişimler için. ( mesela hem düşük fiyat hem good kaç tane )
buy = pd.crosstab(data['Buying'], data['Evaluation'])

## daha detaylı şeyler ve grafikler falan için baştakı linke bak.
