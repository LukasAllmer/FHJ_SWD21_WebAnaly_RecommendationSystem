critics={'Lisa Rose': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5,
 'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5, 
 'The Night Listener': 3.0},
'Gene Seymour': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5, 
 'Just My Luck': 1.5, 'Superman Returns': 5.0, 'The Night Listener': 3.0, 
 'You, Me and Dupree': 3.5}, 
'Michael Phillips': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0,
 'Superman Returns': 3.5, 'The Night Listener': 4.0},
'Claudia Puig': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0,
 'The Night Listener': 4.5, 'Superman Returns': 4.0, 
 'You, Me and Dupree': 2.5},
'Mick LaSalle': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, 
 'Just My Luck': 2.0, 'Superman Returns': 3.0, 'The Night Listener': 3.0,
 'You, Me and Dupree': 2.0}, 
'Jack Matthews': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
 'The Night Listener': 3.0, 'Superman Returns': 5.0, 'You, Me and Dupree': 3.5},
'Toby': {'Snakes on a Plane':4.5,'You, Me and Dupree':1.0,'Superman Returns':4.0}}

#first step: test the dictionary           
#from recommendations import critics
#critics["Lisa Rose"]["Lady in the Water"]
#critics["Lisa Rose"]


#####second step
#calculate an euclidean distance between two persons
from math import sqrt

def sim_euclidean_distance(prefs, person1, person2):
    #get the list of shared items
    si={}
    for item in prefs[person1]:
        if item in prefs[person2]:
            si[item]=1
    #if they have no ratings in common, return 0
    if len(si)==0:
        return 0
    #add up the squares of all differences
    sum_of_squares=sum([pow(prefs[person1][item]-prefs[person2][item],2) for item in si])
    return 1/(1+sqrt(sum_of_squares))
    

#import recommendations
#reload(recommendations)    
#recommendations.sim_euclidean_distance(recommendations.critics,"Lisa Rose", "Gene Seymour") 

########third step
#calculate and sort the euclidean distance between one person and all other and sort the result
def sort_by_similarity_euclidean(prefs,person1):
    so={}
    for person2 in prefs:
        so[person2]=sim_euclidean_distance(prefs,person1, person2)

    #sorted output
    from collections import OrderedDict
    print(OrderedDict(sorted(so.items(), key=lambda x: x[1], reverse=True)))
#recommendations.sort_by_similarity_euclidean(recommendations.critics,"Lisa Rose")    


# more Distance measures http://gedas.bizhat.com/dist.htm

##########fourth step
#calculate the pearson score
def sim_pearson_distance(prefs,person1,person2):
    #get then list of mutually rated items
    si={}
    for item in prefs[person1]:
        if item in prefs[person2]:
            si[item]=1
            
    #find the number of elements
    n=len(si)
    #if they have no ratings in common, return 0
    if n==0:
        return 0
    #Add up all preferences
    sum_person1 = sum([prefs[person1][item] for item in si])
    sum_person2 = sum([prefs[person2][item] for item in si])
        
    #sum up the squares
    sum_person1_square = sum([pow(prefs[person1][item],2) for item in si])
    sum_person2_square = sum([pow(prefs[person2][item],2) for item in si])
    
    #sum up the products
    sum_person1_multiply_person2 = sum([prefs[person1][item]*prefs[person2][item] for item in si])
    
    #calculate pearson score
    num=sum_person1_multiply_person2- sum_person1*sum_person2/n
    den = sqrt((sum_person1_square-pow(sum_person1,2)/n)*(sum_person2_square-pow(sum_person2,2)/n))
    if den==0:
        return 0
    r=num/den
    return r
#recommendations.sim_pearson_distance(recommendations.critics,"Lisa Rose", "Jack Matthews") 

########fifth step
#calculate and sort the pearson distance between one person and all other and sort the result
def sort_by_similarity_pearson(prefs,person1):
    so={}
    for item in prefs:
        so[item]=sim_pearson_distance(prefs,person1, item)           
    from collections import OrderedDict
    print(OrderedDict(sorted(so.items(), key=lambda x: x[1], reverse=True)))
#recommendations.sort_by_similarity_pearson(recommendations.critics,"Lisa Rose")    

#########sixth step
#recommend an item
def getRecommendations(prefs, person, similarity=sim_pearson_distance):
    totals={}
    simSums={}
    for other in prefs:
        #don't compre to the person itself
        if other==person: continue
        sim=similarity(prefs, person, other)
        
        #ignore scores of zero or lower
        if sim<=0: continue
        
        for item in prefs[other]:
            #only score movies person have not seen yet
            if item not in prefs[person] or prefs[person][item]==0:
                #similarity * score
                totals.setdefault(item, 0)
                totals[item] +=prefs[other][item]*sim
                #sum of similarities
                simSums.setdefault(item,0)
                simSums[item]+=sim
    #create normalized list
    rankings=[(total/simSums[item],item) for item,total in totals.items()]
    
    #return the sorted list
    rankings.sort()
    rankings.reverse()
    return rankings
#recommendations.getRecommendations(recommendations.critics,"Toby")    
#recommendations.getRecommendations(recommendations.critics,"Toby",similarity=recommendations.sim_euclidean_distance)    

#############seventh step
#collaborative item based filtering
#add the swapped critics
def transformPrefs(prefs):
    result={}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item,{})
            
            #flip item and person
            result[item][person]=prefs[person][item]
    return result
    
def topItems(critics,person,similarity=sim_pearson_distance):
    result = transformPrefs(critics) 
    #print result  
    print(getRecommendations(result,person,similarity))  
        
#recommendations.topItems(recommendations.critics,"Just My Luck")    
#recommendations.topItems(recommendations.critics,"Just My Luck",similarity=recommendations.sim_euclidean_distance)    

######ranking the critics
#returns the best matches for person from the prefs dictionary
#number of results and similarity funtion are optional params.
def topMatches(prefs, person, n=5, similarity=sim_pearson_distance):
    scores = [(similarity(prefs,person,other),other) for other in prefs if other!=person]
    
    #sort the lsit to the highest scores appear at the top
    scores.sort()
    scores.reverse()
    return scores[0:n]
#recommendations.topMatches(recommendations.critics, "Toby", n=3)

################eight step
#content based filtering
def calculateSimilarItems(prefs,n=10):
    #create dict of items showing which other items they are most similar to
    
    result={}
    #invert the pprefs-matrix to be item-centric
    itemPrefs = transformPrefs(prefs) 
    
    c=0
    for item in itemPrefs:
        #status updates for large datasets
        c+=1
        if c%100==0: print("%d / %d" %(c,len(itemPrefs)))
        #find most similar items to this one
        scores = topMatches(itemPrefs, item, n=n, similarity=sim_euclidean_distance)
        result[item] = scores
    return result
#itemsim=recommendations.calculateSimilarItems(recommendations.critics)

def getRecommendedItems(prefs, itemMatch,user):
    userRatings=prefs[user]
    scores={}
    totalSim={}
    
    #loop over items rated by this user
    for (item, rating) in userRatings.items():
        #loop over items similar to this one
        for (similarity,item2) in itemMatch[item]:
             #Ignore if zhis user has already rated this item
             if item2 in userRatings: continue
             
             #weighted sum of rating times similarity
             scores.setdefault(item2,0)
             scores[item2]+=similarity*rating
             
             #sum of all the similarities
             totalSim.setdefault(item2,0)
             totalSim[item2]+=similarity
        #divide each total score by total weighting to get an average
        rankings =[(score/totalSim[item],item) for item,score in scores.items()]
        
        #retunr the rankings from highest to lowest
        rankings.sort()
        rankings.reverse()
        return rankings
        
#itemsim=recommendations.calculateSimilarItems(recommendations.critics) 
#recommendations.getRecommendedItems(recommendations.critics,itemsim,"Toby")          












