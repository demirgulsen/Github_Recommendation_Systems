##################################################
# Association Rule Based Recommender System
##################################################

#########################
# İş Problemi
#########################

# Türkiye’nin en büyük online hizmet platformu olan Armut, hizmet verenler ile hizmet almak isteyenleri buluşturmaktadır.
# Bilgisayarın veya akıllı telefonunun üzerinden birkaç dokunuşla temizlik, tadilat, nakliyat gibi hizmetlere kolayca
# ulaşılmasını sağlamaktadır.
# Hizmet alan kullanıcıları ve bu kullanıcıların almış oldukları servis ve kategorileri içeren veri setini kullanarak
# Association Rule Learning ile ürün tavsiye sistemi oluşturulmak istenmektedir.


#########################
# Veri Seti
#########################
# Veri seti müşterilerin aldıkları servislerden ve bu servislerin kategorilerinden oluşmaktadır.
# Alınan her hizmetin tarih ve saat bilgisini içermektedir.

# UserId: Müşteri numarası
# ServiceId: Her kategoriye ait anonimleştirilmiş servislerdir. (Örnek : Temizlik kategorisi altında koltuk yıkama servisi)
# Bir ServiceId farklı kategoriler altında bulanabilir ve farklı kategoriler altında farklı servisleri ifade eder.
# (Örnek: CategoryId’si 7 ServiceId’si 4 olan hizmet petek temizliği iken CategoryId’si 2 ServiceId’si 4 olan hizmet mobilya montaj)
# CategoryId: Anonimleştirilmiş kategorilerdir. (Örnek : Temizlik, nakliyat, tadilat kategorisi)
# CreateDate: Hizmetin satın alındığı tarih

##############################
# Proje Görevleri
##############################

##############################
# GÖREV 1: Veriyi Hazırlama
##############################
# Adım 1: armut_data.csv dosyasını okutunuz.
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("Github_Recommendation_Systems/CaseStudy_1/datasets/armut_data.csv")
df.head()
df.shape
df.info()
df.isnull().sum()

# Adım 2: ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir. ServiceId ve CategoryId’yi "_" ile birleştirerek bu hizmetleri
# temsil edecek yeni bir değişken oluşturunuz. Elde edilmesi gereken çıktı:

# Hizmet
# 0_7
# 48_5

df['Hizmet'] = df['ServiceId'].astype(str) + '_' + df['CategoryId'].astype(str)
df['Hizmet'].head()

# Adım 3: Veri seti, hizmetlerin alındığı tarih ve saatten oluşmaktadır, herhangi bir sepet tanımı (fatura vb. ) bulunmamaktadır. Association Rule
# Learning uygulayabilmek için bir sepet (fatura vb.) tanımı oluşturulması gerekmektedir.
# Burada sepet tanımı her bir müşterinin aylık aldığı hizmetlerdir.
# Örneğin; 7256 id'li müşteri 2017'in 8.ayında aldığı 9_4, 46_4 hizmetleri bir sepeti; 2017’in 10.ayında aldığı 9_4, 38_4 hizmetleri
# başka bir sepeti ifade etmektedir. Sepetleri unique bir ID ile tanımlanması gerekmektedir. Bunun için öncelikle sadece yıl ve ay içeren yeni bir
# date değişkeni oluşturunuz. UserID ve yeni oluşturduğunuz date değişkenini "_" ile birleştirirek ID adında yeni bir değişkene atayınız.
# Elde edilmesi gereken çıktı:

# SepetId
# # 10000_2017-08

df['CreateDate'] = pd.to_datetime(df['CreateDate'])
df['New_Date'] = df["CreateDate"].dt.strftime("%Y-%m")
df['SepetId'] = df['UserId'].astype(str) + '_' + df['New_Date']

################################################################
# Görev 2: Birliktelik Kuralları Üretiniz ve Öneride bulununuz
################################################################
# Adım 1: Aşağıdaki gibi sepet, hizmet pivot table’i oluşturunuz.

# Hizmet         0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
# SepetID
# 0_2017-08        0     0      0     0      0     0     0     0     0     0..
# 0_2017-09        0     0      0     0      0     0     0     0     0     0..
# 0_2018-01        0     0      0     0      0     0     0     0     0     0..
# 0_2018-04        0     0      0     0      0     1     0     0     0     0..
# 10000_2017-08    0     0      0     0      0     0     0     0     0     0..

service_basket_df = df.groupby(['SepetId', 'Hizmet'])['Hizmet'].count().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
service_basket_df.head()

# applymap' e alternatif çözüm
# service_basket_df = df.groupby(['SepetId', 'Hizmet'])['Hizmet'].count().unstack().fillna(0)
# service_basket_df = service_basket_df.apply(lambda x: x.apply(lambda y: 1 if y > 0 else 0), axis=1)


# Adım 2: Birliktelik kurallarını oluşturunuz.

frequent_itemsets = apriori(service_basket_df, min_support=0.01, use_colnames=True)


rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)

# Adım3: arl_recommender fonksiyonunu kullanarak en son 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))

    recommendation_list = list({item for item_list in recommendation_list for item in item_list})
    return recommendation_list[:rec_count]


arl_recommender(rules, '2_0', 2)
arl_recommender(rules, '13_11', 1)
arl_recommender(rules, '38_4', 3)

