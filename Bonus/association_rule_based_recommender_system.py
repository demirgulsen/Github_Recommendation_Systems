######################################################################################
# Association Rule Based Recommender System
######################################################################################

####################
# İş Problemi
####################
# Aşağıda 3 farklı kullanıcının sepet bilgileri verilmiştir. Bu sepet bilgilerine en
# uygun ürün önerisini birliktelik kuralı kullanarak yapınız. Ürün önerileri 1 tane
# ya da 1'den fazla olabilir. Karar kurallarını 2010-2011 Germany müşterileri
# üzerinden türetiniz.
# Kullanıcı 1’in sepetinde bulunan ürünün id'si: 21987
# Kullanıcı 2’in sepetinde bulunan ürünün id'si : 23235
# Kullanıcı 3’in sepetinde bulunan ürünün id'si : 22747

########################
# Veri Seti Hikayesi
########################
# Online Retail II isimli veri seti İngiltere merkezli bir perakende şirketinin 01/12/2009 - 09/12/2011 tarihleri arasındaki online satış işlemlerini içeriyor.
# Şirketin ürün kataloğunda hediyelik eşyalar yer almaktadır ve çoğu müşterisinin toptancı olduğu bilgisi mevcuttur.

##################
# Değişkenler
##################
# InvoiceNo   : Fatura Numarası ( Eğer bu kod C ile başlıyorsa işlemin iptal edildiğini ifade eder )
# StockCode   : Ürün kodu ( Her bir ürün için eşsiz )
# Description : Ürün ismi
# Quantity    : Ürün adedi ( Faturalardaki ürünlerden kaçar tane satıldığı)
# InvoiceDate : Fatura tarihi
# UnitPrice   : Fatura fiyatı ( Sterlin )
# CustomerID  : Eşsiz müşteri numarası
# Country     : Ülke ismi

###################################
# Proje Görevleri
###################################

###################################
# Görev 1: Veriyi Hazırlama
###################################
# Adım 1: Online Retail II veri setinden 2010-2011 sheet’ini okutunuz.

# pip install openpyxl
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

df_ = pd.read_excel("Github_Recommendation_Systems/Bonus/dataset/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.head()
df.info()

# Adım 2: StockCode’u POST olan gözlem birimlerini drop ediniz. (POST her faturaya eklenen bedel, ürünü ifade etmemektedir.)
df = df[~(df['StockCode'] == 'POST')]

# Adım 3: Boş değer içeren gözlem birimlerini drop ediniz.
df.dropna(inplace=True)

# Adım 4: Invoice içerisinde C bulunan değerleri veri setinden çıkarınız. (C faturanın iptalini ifade etmektedir.)
df = df[~df['Invoice'].str.contains('C', na=False)]

# Adım 5: Price değeri sıfırdan küçük olan gözlem birimlerini filtreleyiniz.
df[df['Price'] < 0]
df[df['Quantity'] < 0]

df = df[df['Quantity'] > 0]
df = df[df['Price'] > 0]

# Adım 6: Price ve Quantity değişkenlerinin aykırı değerlerini inceleyiniz, gerekirse baskılayınız.
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


replace_with_thresholds(df, 'Quantity')
replace_with_thresholds(df, 'Price')

###################################################################
# Görev 2: Alman Müşteriler Üzerinden Birliktelik Kuralları Üretme
###################################################################
# Adım 1: Fatura ürün pivot table’i oluşturacak create_invoice_product_df fonksiyonunu tanımlayınız.
germany_df = df[df['Country'] == 'Germany']

def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


ger_inv_pro_df = create_invoice_product_df(germany_df)

ger_inv_pro_df = create_invoice_product_df(germany_df, id=True)

##################
def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


check_id(germany_df, 84347)

# Adım 2: Kuralları oluşturacak create_rules fonksiyonunu tanımlayınız ve alman müşteriler için kurallarını bulunuz.

def create_rules(dataframe, id=True):
    dataframe = create_invoice_product_df(dataframe, id)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules


rules = create_rules(germany_df)

rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)].sort_values("confidence", ascending=False)
###################################################################
# Görev 3: Sepet İçerisindeki Ürün Id’leri Verilen Kullanıcılara Ürün Önerisinde Bulunma
###################################################################
# Adım 1: check_id fonksiyonunu kullanarak verilen ürünlerin isimlerini bulunuz.

# Kullanıcı 1’in sepetinde bulunan ürünün id'si: 21987
# Kullanıcı 2’in sepetinde bulunan ürünün id'si : 23235
# Kullanıcı 3’in sepetinde bulunan ürünün id'si : 22747

check_id(germany_df, 21987)   # --> PACK OF 6 SKULL PAPER CUPS
check_id(germany_df, 23235)   # --> STORAGE TIN VINTAGE LEAF
check_id(germany_df, 22747)   # --> POPPY'S PLAYHOUSE BATHROOM

# Adım 2: arl_recommender fonksiyonunu kullanarak 3 kullanıcı için ürün önerisinde bulununuz.

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]


arl_recommender(rules, 21987, 1)  # --> 21086
arl_recommender(rules, 23235, 3)  # --> 23243, 23244, 23236
arl_recommender(rules, 22747, 2)  # --> 22745, 22746

# Adım 3: Önerilecek ürünlerin isimlerine bakınız.

check_id(germany_df, 21086)   # --> SET/6 RED SPOTTY PAPER CUPS

check_id(germany_df, 23243)   # --> SET OF TEA COFFEE SUGAR TINS PANTRY
check_id(germany_df, 23244)   # --> ROUND STORAGE TIN VINTAGE LEAF
check_id(germany_df, 23236)   # --> DOILEY STORAGE TIN

check_id(germany_df, 22745)   # --> POPPY'S PLAYHOUSE BATHROOM
check_id(germany_df, 22746)   # --> POPPY'S PLAYHOUSE LIVINGROOM

###################################################################
# Görev 4. Çalışmanın Scriptini Hazırlama
###################################################################

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe


def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


def create_rules(dataframe, id=True, country="Germany"):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df(dataframe, id)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules


def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]



df = df_.copy()
df = retail_data_prep(df)
rules = create_rules(df)

arl_recommender(rules, 21987, 1)  # --> 21086
check_id(germany_df, 21987)   # --> PACK OF 6 SKULL PAPER CUPS
check_id(germany_df, 21086)   # --> SET/6 RED SPOTTY PAPER CUPS

# Yani  PACK OF 6 SKULL PAPER CUPS  ürünü için SET/6 RED SPOTTY PAPER CUPS ürününü önerebiliriz