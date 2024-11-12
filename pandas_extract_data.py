import pandas as pd


tmdb_original = pd.read_csv('TMDB_all_movies.csv.', encoding='utf-8')
df = pd.DataFrame(tmdb_original)

# 필요한 열만 추출 -> 12개 coloumn 추출
select_column = df[['id', 'title', 'vote_average','release_date','runtime','original_language','overview','genres','production_companies','production_countries','director','cast','poster_path']]

# 원어가 영어인 것만 추출 -> 추출 후 506351 rows
filtered_en = select_column.loc[select_column['original_language'] == 'en']

# NaN 값 제외 -> 제외 후 130338 row
filtered_nan = filtered_en.dropna()

# 교육용인지 아닌지 판별하기
# 교육용 데이터 세트의 tmdbid와 df의 id가 같다면 true
add_is_educational = pd.read_csv('links.csv', encoding='utf-8')

# 교육용이면서 tmdb에 있는 것 -> 42209 row
# tmdb 데이터 중에서 교육용인 것을 표시하기 위함 -> 130256 row
education_mergered = pd.merge(filtered_nan, add_is_educational, left_on='id', right_on='tmdbId', how='left') 

# moiveid 제외 (imdb는 나중에 level 추가할 때 필요함)
education_mergered.drop(columns=['movieId'], inplace=True)
education_mergered['is_for_educational'] = education_mergered['tmdbId'].notna()  # tmdbId가 있는 경우 True로 설정

# tmdbId 삭제
education_mergered.drop(columns=['tmdbId'], inplace=True)

# column 명 변경
education_mergered.rename(columns={
    'id':'tmdbid',
    'vote_average':'rank',
    'overview':'summary',
    'production_companies': 'production',
    'production_countries': 'country',
    'cast': 'starrinig' 
}, inplace=True)

# prodcution company, county 하나만 남기기
education_mergered['production'] = education_mergered['production'].apply(lambda x: x.split(',')[0] if isinstance(x, str) else x)
education_mergered['country'] = education_mergered['country'].apply(lambda x: x.split(',')[0] if isinstance(x, str) else x)

# 언어 컬럼 지우기
education_mergered.drop(columns='original_language', inplace=True)

#imdbid가 없는 것 지우기 -> 41366 rows
education_mergered = education_mergered.dropna()

# 숫자 값들 int로 변형
education_mergered['runtime'] = education_mergered['runtime'].astype(int)
education_mergered['imdbId'] = education_mergered['imdbId'].astype(int)

print(education_mergered)
education_mergered.to_csv('movie_with_education.csv', index=False)