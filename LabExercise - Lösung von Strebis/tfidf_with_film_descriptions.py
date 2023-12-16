import pandas as pd

# install package Scikit-learn
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

# intialise data of lists.
data = {'title': ['Wildlike', 'Shaun the Shep Movie', 'Amy', 'A Hard Day'],
        'overview': ["Wildlike is a 2015[1] American feature film written and directed by Frank Hall Green.[2] Filmed in Alaska and starring Ella Purnell, Bruce Greenwood, Brian Geraghty, Nolan Gerard Funk and Ann Dowd, Wildlike has a 92% Fresh rating on Rotten Tomatoes, and has played over 150 film festivals and won over 100 festival awards.[3] The movie was filmed on location in Denali National Park, Juneau, Anchorage, Palmer, Whittier, Matanuska Glacier and on the state ferry boat Kennicott run by the Alaska Marine Highway System. The film was produced by Green, with Julie Christeas, Schuyler Weiss and Joseph Stephans. The Executive Producer was Christine Vachon.[4] The Director of Photography was Hillary Spera,[5] and it was edited by Mako Kamitsuna. The music was composed by Daniel Bensi and Saunder Jurriaans, and the production Designer was Chad Keith.",
        "Shaun the Sheep Movie (also called Shaun the Sheep: The Movie) is a 2015 British stop-motion animated adventure comedy film based on the 2007 television series of the same name, created by Nick Park, in turn a spin off of the Wallace and Gromit film, A Close Shave (1995). The film follows Shaun and his flock into the big city to rescue their farmer, who found himself amnesiac there as a result of their mischief. It was produced by Aardman Animations, and financed by StudioCanal in association with Anton Capital Entertainment,[6][8] with the former also distributing the film in the United Kingdom and several other European countries.[9] Richard Starzak and Mark Burton wrote and directed the film, Ilan Eshkeri composed the music, and Justin Fletcher, John Sparkes, and Omid Djalili provided the voices. The film premiered on 24 January 2015, at the Sundance Film Festival, and was theatrically released in the United Kingdom on 6 February 2015. The film made $106.2 million at the box office and received near-universal acclaim from critics, with many calling it, fun, absurd, and endearingly inventive, and praised the animation. It holds an approval rating of 99% on Rotten Tomatoes and is one of the highest rated animated films on the website.[10][11][12] Shaun the Sheep Movie was nominated at the 88th Academy Awards for the Academy Award for Best Animated Feature. It was nominated in The 73rd Golden Globe Awards, BAFTA Awards and won in the Toronto Film Critics Awards for Best Animated Film. It earned five nominations at the Annie Awards including Best Animated Feature. A sequel entitled A Shaun the Sheep Movie: Farmageddon was released on 18 October 2019.",
        "Amy is a 2015 British documentary film about the life and death of British singer-songwriter Amy Winehouse. The film was directed by Asif Kapadia and produced by James Gay-Rees, George Pank, and Paul Bell and co-produced by Krishwerkz Entertainment, On The Corner Films, Playmaker Films, and Universal Music, in association with Film4. The film covers Winehouse's life and her struggle with substance abuse, both before and after her career blossomed, and which eventually caused her death. In February 2015, a teaser trailer based on the life of Winehouse debuted at the pre-Grammy event in the build-up to the 2015 Grammy Awards. David Joseph, CEO of Universal Music UK, announced that the documentary entitled simply Amy would be released later that year. He further stated: About two years ago we decided to make a movie about her—her career and her life. It's a very complicated and tender movie. It tackles lots of things about family and media, fame, addiction, but most importantly, it captures the very heart of what she was about, which is an amazing person and a true musical genius.[4] The film was shown in the Midnight Screenings section at the 2015 Cannes Film Festival[5] and received its UK premiere at the Edinburgh International Film Festival.[6] The film is distributed by the Altitude Film Distribution and A24, and was released theatrically on 3 July 2015 in the United Kingdom and the United States, and worldwide on 10 July; it received critical acclaim. Amy became the highest-grossing British documentary of all time, taking £3 million at the box office in its first weekend.[7] The film has received 33 nominations and has won a total of 30 film awards, including for Best European Documentary at the 28th European Film Awards,[8] Best Documentary at the 69th British Academy Film Awards, Best Music Film at the 58th Grammy Awards, the Academy Award for Best Documentary Feature at the 88th Academy Awards[9] and for Best Documentary at the 2016 MTV Movie Awards. The success of the film and the music from the soundtrack of the same name also led Winehouse her second posthumous nomination at the 2016 BRIT Awards for British Female Solo Artist [10]",
        "A Hard Day (Korean: 끝까지 간다; RR: Kkeutkkaji Ganda; lit. Take It to the End) is a 2014 South Korean action thriller film written and directed by Kim Seong-hun, and starring Lee Sun-kyun and Cho Jin-woong.[1][2][3][4][5] It was selected to compete in the Directors' Fortnight section of the 2014 Cannes Film Festival.[6][7][8][9]"
        ]}



#data = {'title': ['Movie1', 'Movie2', 'Movie3'],
#        'overview': [" is a 2014 American film",
#                     " is a 2014 British film",
#                     " is a 2015 British film"
#                    ]}
# Create DataFrame
metadata = pd.DataFrame(data)
docs=metadata['overview']
print(docs)


# instantiate CountVectorizer()
# Convert a collection of text documents to a matrix of token counts
cv = CountVectorizer(stop_words='english')
#cv = CountVectorizer()

# this steps generates word counts for the words in your docs
# word_count_vector - Transform documents to document-term matrix
word_count_vector = cv.fit_transform(docs)
print("#####")
print("Word frequency for each document")
print(word_count_vector)

print("#####")
print("Number of documents and number of different word")
print(word_count_vector.shape)

### Compute  now inverse document frequency
tfidf_transformer=TfidfTransformer()
tfidf_transformer.fit(word_count_vector)

# print idf values
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=sorted(cv.vocabulary_.keys()), columns=["idf_weights"])

# sort ascending
print("#####")
print("IDF for a word")
print(df_idf.sort_values(by=['idf_weights']))

# tf-idf scores
tf_idf_vector = tfidf_transformer.transform(word_count_vector)
print("#######")
print("TF-IDF per document")
print(tf_idf_vector)
print("#######")
print("TF-IDF as array")
print(tf_idf_vector.toarray())