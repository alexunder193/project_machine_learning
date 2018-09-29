from os import path
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import io
import os

texts={}
df=pd.read_csv("train_set.csv",sep="\t")

categories=['Business', 'Politics', 'Film', 'Football', 'Technology']
my_stop_words=['will','one','two','four','new','now','day','year','month','week','ago','late','little','many','said','last','time','first','second','make','say','saying','may','maybe','long','short','use','says','old','made','today', 'back', 'face','believe','around','become','th','high']
stop_words = STOPWORDS.union(my_stop_words)


if not os.path.exists("Images"):
    os.makedirs("Images")

for i in categories:
	texts[i]=df.ix[df['Category']==i]['Content']
	texts[i]=texts[i].to_string(header=False)
	texts[i]=texts[i].replace('\n', ' ').replace('\r', '')
    
	wordcloud = WordCloud(max_font_size=50, min_font_size=2, max_words=500, stopwords=stop_words, background_color="white", relative_scaling=.4).generate(texts[i])
	image = wordcloud.to_image()
	image.save("Images/"+i+'.png')
	image.show()
  

