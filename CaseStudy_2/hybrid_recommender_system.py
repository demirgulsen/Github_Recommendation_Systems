
#############################################
# PROJE: Hybrid Recommender System
#############################################

#########################
# İş Problemi
#########################
# ID'si verilen kullanıcı için item-based ve user-based recomennder yöntemlerini kullanarak tahmin yapınız.
# 5 öneri user-based modelden 5 öneri de item-based modelden ele alınız ve nihai olarak 10 öneriyi 2 modelden yapınız.
git add .gitattributes
git commit -m "CaseStudy_2/datasets/rating.csv"

git add "CaseStudy_2/datasets/rating.csv"
git commit -m "Add large dataset"

##########################
# Veri Seti Hikayesi
##########################
# Veri seti, bir film tavsiye hizmeti olan MovieLens tarafından sağlanmıştır. İçerisinde filmler ile birlikte bu filmlere yapılan
# derecelendirme puanlarını barındırmaktadır. 27.278 filmde 2.000.0263 derecelendirme içermektedir. Bu veri seti ise 17 Ekim 2016
# tarihinde oluşturulmuştur. 138.493 kullanıcı ve 09 Ocak 1995 ile 31 Mart 2015 tarihleri arasında verileri içermektedir. Kullanıcılar
# rastgele seçilmiştir. Seçilen tüm kullanıcıların en az 20 filme oy verdiği bilgisi mevcuttur.

##########################
# Proje Görevleri
##########################

# ****************************************************************************
# User Based Recommendation
# ****************************************************************************

#############################################
# Görev 1: Verinin Hazırlanması
#############################################
# Adım 1: movie, rating veri setlerini okutunuz.
import pandas as pd
pd.pandas.set_option('display.max_columns', None)
pd.pandas.set_option('display.width', 500)

movie = pd.read_csv('Github_Recommendation_Systems/CaseStudy_2/datasets/movie.csv')
rating = pd.read_csv('Github_Recommendation_Systems/CaseStudy_2/datasets/rating.csv')

movie.head()
movie.shape
movie.info()

rating.head()
rating.shape
rating.info()

# Adım 2: rating veri setine Id’lere ait film isimlerini ve türünü movie veri setinden ekleyiniz.
df = movie.merge(rating, how='left', on='movieId')
df.head()

# Adım3: Toplam oy kullanılma sayısı 1000'in altında olan filmlerin isimlerini listede tutunuz ve veri setinden çıkartınız.
comment_count = pd.DataFrame(df['title'].value_counts())

rare_movies = comment_count[comment_count['count'] <= 1000].index
# 1000 ve altı yorum alan filmlerin dışındaki filmleri yani 1000 üzeri yorum alan filmler ile işlem yapmak için bu filmleri seçtik
common_movies = df[~df['title'].isin(rare_movies)]

# Adım 4: index'te userID'lerin sutunlarda film isimlerinin ve değer olarak ratinglerin bulunduğu dataframe için pivot table oluşturunuz.
user_movie_df = common_movies.pivot_table(index=['userId'], columns=['title'], values='rating')

# Adım5: Yapılan tüm işlemleri fonksiyonlaştırınız.
def create_user_movie_df(df):
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df


import pandas as pd
movie = pd.read_csv('Github_Recommendation_Systems/CaseStudy_2/datasets/movie.csv')
rating = pd.read_csv('Github_Recommendation_Systems/CaseStudy_2/datasets/rating.csv')
df = movie.merge(rating, how="left", on="movieId")

user_movie_df = create_user_movie_df(df)

###########################################################################
# Görev 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
###########################################################################
# Adım 1: Rastgele bir kullanıcı id’si seçiniz.
# random_user = int(pd.Series(user_movie_df.index).sample(1).values[0])
random_user = 108170

# Adım 2: Seçilen kullanıcıya ait gözlem birimlerinden oluşan random_user_df adında yeni bir dataframe oluşturunuz.
random_user_df = user_movie_df[user_movie_df.index ==random_user]

# Adım 3: Seçilen kullanıcıların oy kullandığı filmleri movies_watched adında bir listeye atayınız
movies_watched = random_user_df.columns[random_user_df.notna().any()].to_list()

###########################################################################
# Görev 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişilmesi
###########################################################################
# Adım 1: Seçilen kullanıcının izlediği fimlere ait sutunları user_movie_df'ten seçiniz ve movies_watched_df adında yeni bir dataframe oluşturunuz.
movies_watched_df = user_movie_df[movies_watched]

# Adım 2: Her bir kullancının seçili user'in izlediği filmlerin kaçını izlediğini bilgisini taşıyan user_movie_count adında yeni bir dataframe oluşturunuz.

user_movie_count = movies_watched_df.T.notnull().sum()

user_movie_count = user_movie_count.reset_index()

user_movie_count.columns = ["userId", "movie_count"]

# Adım 3: Seçilen kullanıcının oy verdiği filmlerin yüzde 60 ve üstünü izleyenlerin kullanıcı id’lerinden users_same_movies adında bir liste oluşturunuz.
perc = len(movies_watched) * 60 / 100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

###########################################################################
# Görev 4: Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi
###########################################################################
# Adım 1: user_same_movies listesi içerisindeki seçili user ile benzerlik gösteren kullanıcıların id’lerinin bulunacağı şekilde movies_watched_df
# dataframe’ini filtreleyiniz.
final_df = movies_watched_df[movies_watched_df.index.isin(users_same_movies)]

# Adım 2: Kullanıcıların birbirleri ile olan korelasyonlarının bulunacağı yeni bir corr_df dataframe’i oluşturunuz.
# corr_df = final_df.T.corr().unstack().sort_values()

corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()

# corr_df[corr_df['user_id_1'] == random_user]

# Adım 3: Seçili kullanıcı ile yüksek korelasyona sahip (0.65’in üzerinde olan) kullanıcıları filtreleyerek top_users adında yeni bir dataframe oluşturunuz.
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]].reset_index(drop=True)

top_users = top_users.sort_values(by='corr', ascending=False)

top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

# Adım 4: top_users dataframe’ine rating veri seti ile merge ediniz.
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]

###########################################################################
# Görev 5: Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması
###########################################################################
# Adım 1: Her bir kullanıcının corr ve rating değerlerinin çarpımından oluşan weighted_rating adında yeni bir değişken oluşturunuz.
top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

# Adım 2: Film id’si ve her bir filme ait tüm kullanıcıların weighted rating’lerinin ortalama değerini içeren recommendation_df adında yeni bir
# dataframe oluşturunuz.
recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

recommendation_df = recommendation_df.reset_index()

# Adım 3: recommendation_df içerisinde weighted rating'i 3.5'ten büyük olan filmleri seçiniz ve weighted rating’e göre sıralayınız.
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)

# Adım 4: movie veri setinden film isimlerini getiriniz ve tavsiye edilecek ilk 5 filmi seçiniz.

movies_to_be_recommend.merge(movie[["movieId", "title"]])["title"][:5]

# ****************************************************************************
# Item Based Recommendation
# ****************************************************************************

###########################################################################
# Görev 1: Kullanıcının izlediği en son ve en yüksek puan verdiği filme göre item-based öneri yapınız.
###########################################################################
# Adım 1: movie, rating veri setlerini okutunuz.
movie = pd.read_csv('Github_Recommendation_Systems/CaseStudy_2/datasets/movie.csv')
rating = pd.read_csv('Github_Recommendation_Systems/CaseStudy_2/datasets/rating.csv')

# Adım 2: Seçili kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sinin alınız.
user = 108170
movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]

# Adım3: User based recommendation bölümünde oluşturulan user_movie_df dataframe’ini seçilen film id’sine göre filtreleyiniz.

movie_df = user_movie_df[movie[movie["movieId"] == movie_id]["title"].values[0]]

# Adım 4: Filtrelenen dataframe’i kullanarak seçili filmle diğer filmlerin korelasyonunu bulunuz ve sıralayınız.
user_movie_df.corrwith(movie_df).sort_values(ascending=False).head(10)

# Son iki adımı uygulayan fonksiyon
def item_based_recommender(movie_name, user_movie_df):
    movie = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie).sort_values(ascending=False).head(10)


# Adım 5: Seçili film’in kendisi haricinde ilk 5 film’I öneri olarak veriniz.
movies_from_item_based = item_based_recommender(movie[movie["movieId"] == movie_id]["title"].values[0], user_movie_df)
movies_from_item_based[1:6]


