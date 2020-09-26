import numpy as np
import codecs
from utils import operations

def load_embeddings(embeddings_path):
    with codecs.open(embeddings_path + '.vocab', 'r', 'utf-8') as f_in:
        index2word = [line.strip() for line in f_in]
    word2index = {w: i for i, w in enumerate(index2word)}
    wv = np.load(embeddings_path + '.npy')

    return wv, index2word, word2index


def flowers_insects_attitude():

    # flowers
    # flowers = ["aster", "clover", "hyacinth", "marigold", "poppy", "azalea", "crocus", "iris", "orchid", "rose", "bluebell", "daffodil", "lilac", "pansy", "tulip", "buttercup", "daisy","lily", "peony", "violet", "carnation", "gladiola", "magnolia", "petunia", "zinnia"]
    # target1 = flowers
    #
    # #insects
    # insects = ["ant", "caterpillar", "flea", "locust", "spider", "bedbug", "centipede", "fly","maggot", "tarantula", "bee", "cockroach", "gnat", "mosquito", "termite", "beetle", "cricket", "hornet", "moth", "wasp", "blackfly", "dragonfly", "horsefly", "roach", "weevil"]
    # target2 = insects
    #
    # # pleasant attributes
    # pleasant1 = ["caress", "freedom", "health", "love", "peace", "cheer", "friend","heaven", "loyal", "pleasure", "diamond", "gentle", "honest", "lucky", "rainbow","diploma", "gift", "honor", "miracle", "sunrise", "family", "happy", "laughter","paradise", "vacation"]
    # attribute1 = pleasant1
    #
    # # unpleasant attributes
    # unpleasant1 = ["abuse" , "crash" , "filth" , "murder" , "sickness" , "accident" , "death" , "grief" , "poison" , "stink" , "assault" , "disaster" , "hatred" , "pollute" , "tragedy" , "divorce" , "jail" , "poverty" , "ugly" , "cancer" , "kill" , "rotten" , "vomit" , "agony" , "prison"]
    # attribute2 = unpleasant1
    flowers = ["woman", "mother"]
    target1 = flowers

    #insects
    insects = ["man", "father"]
    target2 = insects

    # pleasant attributes
    pleasant1 = ["health", "happy"]
    attribute1 = pleasant1

    # unpleasant attributes
    unpleasant1 = ["pollute", "tragedy"]
    attribute2 = unpleasant1

    return target1,target2,attribute1,attribute2

def instruments_weapons_attitude():

    # instruments
    instruments = ["bagpipe","cello", "guitar", "lute", "trombone", "banjo", "clarinet", "harmonica", "mandolin", "trumpet", "bassoon", "drum", "harp", "oboe", "tuba", "bell", "fiddle", "harpsichord", "piano", "viola", "bongo", "flute", "horn", "saxophone", "violin"]
    target1 = instruments

    # weapons
    weapons = ["arrow", "club", "gun", "missile", "spear", "axe", "dagger", "harpoon", "pistol", "sword", "blade", "dynamite", "hatchet", "rifle", "tank", "bomb", "firearm", "knife", "shotgun", "teargas", "cannon", "grenade", "mace", "slingshot", "whip"]
    target2 = weapons

    # pleasant attributes
    pleasant2 = ["caress", "freedom", "health", "love", "peace", "cheer", "friend","heaven", "loyal", "pleasure", "diamond", "gentle", "honest", "lucky", "rainbow","diploma", "gift", "honor", "miracle", "sunrise", "family", "happy", "laughter","paradise", "vacation"]
    attribute1 = pleasant2

    # unpleasant attributes
    unpleasant2 = ["abuse" , "crash" , "filth" , "murder" , "sickness" , "accident" , "death" , "grief" , "poison" , "stink" , "assault" , "disaster" , "hatred" , "pollute" , "tragedy" , "divorce" , "jail" , "poverty" , "ugly" , "cancer" , "kill" , "rotten" , "vomit" , "agony" , "prison"]
    attribute2 = unpleasant2

    return target1,target2,attribute1,attribute2

def racial_attitude():

    # EuropeanAmerican names
    EuropeanAmerican = ["Adam", "Harry", "Josh", "Roger","Alan", "Frank", "Justin", "Ryan", "Andrew", "Jack", "Matthew", "Stephen", "Brad", "Greg", "Paul", "Jonathan", "Peter", "Amanda", "Courtney", "Heather", "Melanie", "Katie", "Betsy", "Kristin", "Nancy", "Stephanie", "Ellen", "Lauren", "Colleen", "Emily", "Megan", "Rachel"]
    target1 = EuropeanAmerican

    # AfricanAmerican names
    AfricanAmerican = ["Alonzo", "Jamel", "Theo", "Alphonse", "Jerome", "Leroy", "Torrance", "Darnell", "Lamar", "Lionel", "Tyree", "Deion", "Lamont", "Malik", "Terrence", "Tyrone", "Lavon", "Marcellus", "Wardell", "Nichelle", "Shereen", "Ebony", "Latisha", "Shaniqua", "Jasmine", "Tanisha", "Tia", "Lakisha", "Latoya", "Yolanda", "Malika", "Yvette"]
    target2 = AfricanAmerican

    # pleasant attributes
    pleasant3 = ["caress", "freedom", "health", "love", "peace", "cheer", "friend","heaven", "loyal", "pleasure", "diamond", "gentle", "honest", "lucky", "rainbow","diploma", "gift", "honor", "miracle", "sunrise", "family", "happy", "laughter","paradise", "vacation"]
    attribute1 = pleasant3

    # unpleasant attributes
    unpleasant3 = ["abuse" , "crash" , "filth" , "murder" , "sickness" , "accident" , "death" , "grief" , "poison" , "stink" , "assault" , "disaster" , "hatred" , "pollute" , "tragedy" , "bomb" , "divorce" , "jail" , "poverty" , "ugly" , "cancer" , "evil" , "kill" , "rotten" , "vomit"]
    attribute2 = unpleasant3

    return target1,target2,attribute1,attribute2

def gender_bias_math_arts():

    # math terms
    math1 = ["math" , "algebra" , "geometry" , "calculus" , "equations" , "computation" , "numbers" , "addition"]
    target1 = math1

    # arts terms
    arts1 = ["poetry" , "art" , "sculpture" , "dance" , "literature" , "novel" , "symphony" , "drama"]
    target2 = arts1

    # male attributes
    male1 = ["brother" , "male" , "man" , "boy" , "son" , "he" , "his" , "him"]
    attribute1 = male1

    # female attributes
    female1 = ["sister" , "female" , "woman" , "girl" , "daughter" , "she" , "hers" , "her"]
    attribute2 = female1
    return target1,target2,attribute1,attribute2

def racial_attitude_market_discrimination():

    # Bertrand, 2003: "Are Emily and Greg More Employable than Lakisha and Jamal? A Field Experiment on Labor Market Discrimination"
    # European American names from the market discrimination study
    # for glove
    EuropeanAmericanNamesMarketDiscrimination = ["Todd", "Neil", "Geoffrey", "Brett", "Brendan", "Greg", "Matthew", "Brad","Allison","Anne","Carrie","Emily","Jill","Laurie","Meredith","Sarah"]
    target1 = EuropeanAmericanNamesMarketDiscrimination


    # African American names from the market discrimination study
    # for glove
    AfricanAmericanNamesMarketDiscrimination = ["Kareem", "Darnell", "Tyrone", "Hakim", "Jamal", "Leroy","Jermaine","Rasheed","Aisha","Ebony","Keisha","Kenya","Lakisha","Latoya","Tamika", "Tanisha"]
    target2 = AfricanAmericanNamesMarketDiscrimination

    #pleasant attributes
    pleasant4 = ["caress", "freedom", "health", "love", "peace", "cheer", "friend","heaven", "loyal", "pleasure", "diamond", "gentle", "honest", "lucky", "rainbow","diploma", "gift", "honor", "miracle", "sunrise", "family", "happy", "laughter","paradise", "vacation"]
    attribute1 = pleasant4

    #unpleasant attributes
    unpleasant4 = ["abuse" , "crash" , "filth" , "murder" , "sickness" , "accident" , "death" , "grief" , "poison" , "stink" , "assault" , "disaster" , "hatred" , "pollute" , "tragedy" , "bomb" , "divorce" , "jail" , "poverty" , "ugly" , "cancer" , "evil" , "kill" , "rotten" , "vomit"]
    attribute2 = unpleasant4

    return target1,target2,attribute1,attribute2

def racial_attitude_market_discrimination_small():

    # Bertrand, 2003: "Are Emily and Greg More Employable than Lakisha and Jamal? A Field Experiment on Labor Market Discrimination"

    # European American names from the market discrimination study
    # for glove
    EuropeanAmericanMarketNamesDiscrimination1 = ["Todd", "Neil", "Geoffrey", "Brett", "Brendan", "Greg", "Matthew", "Brad","Allison","Anne","Carrie","Emily","Jill","Laurie","Meredith","Sarah"]

    target1 = EuropeanAmericanMarketNamesDiscrimination1

    # African American names from the market discrimination study
    # for glove
    AfricanAmericanNamesMarketDiscrimination1 = ["Kareem", "Darnell", "Tyrone", "Hakim", "Jamal", "Leroy","Jermaine","Rasheed","Aisha","Ebony","Keisha","Kenya","Lakisha","Latoya","Tamika", "Tanisha"]
    target2 = AfricanAmericanNamesMarketDiscrimination1

    # pleasant attributes
    pleasant5 = ["joy" , "love" , "peace" , "wonderful" , "pleasure" , "friend" , "laughter" , "happy"]
    attribute1 = pleasant5

    # unpleasant attributes
    unpleasant5 = ["agony" , "terrible" , "horrible" , "nasty" , "evil" , "war" , "awful" , "failure"]
    attribute2 = unpleasant5

    return target1,target2,attribute1,attribute2

def gender_bias_career_family():

    # Nosek, 2002: "Harvesting Implicit Group Attitudes and Beliefs From a Demonstration Web Site"
    # http://projectimplicit.net/nosek/papers/harvesting.GroupDynamics.pdf

    # male names
    maleNames1 = ["John" , "Paul" , "Mike" , "Kevin" , "Steve" , "Greg" , "Jeff" , "Bill"]
    target1 = maleNames1

    # female names
    femaleNames1 = ["Amy" , "Joan" , "Lisa" , "Sarah" , "Diana" , "Kate" , "Ann" , "Donna"]
    target2 = femaleNames1

    # career attributes
    career = ["executive" , "management" , "professional" , "corporation" , "salary" , "office", "business" , "career"]
    attribute1 = career

    # family attributes
    family = ["home" , "parents" , "children" , "family" , "cousins" , "marriage" , "wedding" , "relatives"]
    attribute2 = family

    return target1,target2,attribute1,attribute2

def gender_bias_science_arts():

    # Nosek, 2002: "Math = Male, Me = Female, Therefore Math != Me"
    # http://projectimplicit.net/nosek/papers/nosek.math.JPSP.2002.pdf

    # science terms
    science1 = ["science" , "technology" , "physics"  , "chemistry" , "Einstein","NASA" , "experiment" , "astronomy"]
    target1 = science1

    # arts terms
    arts2 = ["poetry" , "art" , "Shakespeare" , "dance" , "literature" , "novel" , "symphony" , "drama"]
    target2 = arts2

    # male attributes
    male2 = ["brother" , "father" , "uncle" , "grandfather" , "son" , "he" , "his" , "him"]
    attribute1 = male2

    # female attributes
    female2 = ["sister" , "mother" , "aunt" , "grandmother" , "daughter" , "she" , "hers" , "her"]
    attribute2 = female2

    return target1,target2,attribute1,attribute2

def mental_physical_illness_controllability():

    # http://ccf.fiu.edu/research/publications/articles-2010-present/jscp2011305484.pdf
    # Mentally ill people and physically ill people being associated with being controllable or uncontrollable.

    # Terms related to depression
    depressed1 = ["sad" , "hopeless" , "gloomy" , "tearful" , "miserable" , "depressed"]
    target1 = depressed1

    # Terms related to physical illness
    physicallyIll = ["sick" , "illness" , "influenza" , "disease" , "virus" , "cancer"]
    target2 = physicallyIll

    # Attributes about being uncontrollable
    # for glove
    temporary = ["impermanent", "unstable", "variable",  "fleeting", "short-term", "brief", "occasional"]
    # for word2vec
    # temporary = ["impermanent", "unstable", "variable",  "fleeting", "short", "brief", "occasional"]
    attribute1 = temporary


    # Attributes about being controllable
    # actually uses at fault instead of faulty
    permanent = ["stable", "always", "constant", "persistent", "chronic", "prolonged", "forever"]
    attribute2 = permanent

    return target1,target2,attribute1,attribute2

def age_attitude():

    # Nosek, 2002: "Harvesting Implicit Group Attitudes and Beliefs From a Demonstration Web Site"
    # Attitude towards the elderly.
    # Young people's names
    youngNames = ["Tiffany" , "Michelle" , "Cindy" , "Kristy" , "Brad" , "Eric" , "Joey" , "Billy"]
    target1 = youngNames

    # Old people's names
    oldNames = ["Ethel" , "Bernice" , "Gertrude" , "Agnes" , "Cecil" , "Wilbert" , "Mortimer" , "Edgar"]
    target2 = oldNames


    # pleasant attributes
    pleasant6 = ["joy" , "love" , "peace" , "wonderful" , "pleasure" , "friend" , "laughter" , "happy"]
    attribute1 = pleasant6

    # unpleasant terms
    unpleasant6 = ["agony" , "terrible" , "horrible" , "nasty" , "evil" , "war" , "awful" , "failure"]
    attribute2 = unpleasant6

    return target1,target2,attribute1,attribute2


print("Pick a number for bias type to test:" + "\n" +
    "type 1:  biasType = flowers-insects-attitude" + "\n" +
    "type 2:  biasType = instruments-weapons-attitude" + "\n" +
    "type 3:  biasType = racial-attitude" + "\n" +
    "type 4:  biasType = racial-attitude-market-discrimination" + "\n" +
    "type 5:  biasType = racial-attitude-market-discrimination-small" + "\n" +
    "type 6:  biasType = gender-bias-career-family" + "\n" +
    "type 7:  biasType = gender-bias-math-arts" + "\n" +
    "type 8:  biasType = gender-bias-science-arts" + "\n" +
    "type 9:  biasType = mental-physical-illness-stability" + "\n" +
    "type 10:  biasType = age-attitude")

bias = int(input())

while bias not in [1,2,3,4,5,6,7,8,9,10]:
    print("Enter a valid bias")
    bias = int(input())


data = load_embeddings("embedding")
vectors = data[0]
index2word = data[1]
word2index = data[2]

print("Generating results for bias" , bias)

distribution = "normal"
# distribution = "empirical"
if distribution == "empirical":
	iterations = 1000000

elif distribution == "normal":
	iterations = 100000

if(bias == 1):
    targets_attributes = flowers_insects_attitude()
    biasType = 'flowers_insects'
elif(bias == 2):
    targets_attributes = instruments_weapons_attitude()
    biasType = 'instruments_weapons_attitude'
elif(bias == 3):
    targets_attributes = racial_attitude()
    biasType = 'racial_attitude'
elif(bias == 4):
    targets_attributes = racial_attitude_market_discrimination()
    biasType = 'racial_attitude_market_discrimination'
elif(bias == 5):
    targets_attributes = racial_attitude_market_discrimination_small()
    biasType = 'racial_attitude_market_discrimination_small'
elif(bias == 6):
    targets_attributes = gender_bias_career_family()
    biasType = 'gender_bias_career_family'
elif(bias == 7):
    targets_attributes = gender_bias_math_arts()
    biasType = 'gender_bias_math_arts'
elif(bias == 8):
    targets_attributes = gender_bias_science_arts()
    biasType = 'gender_bias_science_arts'
elif(bias == 9):
    targets_attributes = mental_physical_illness_controllability()
    biasType = 'mental_physical_illness_controllability'
elif(bias == 10):
    targets_attributes = age_attitude()
    biasType = 'age_attitude'

concept1 = targets_attributes[0]
concept2 = targets_attributes[1]
stereotype1 = targets_attributes [2]
stereotype2 = targets_attributes [3]
caseSensitive = True
myOperations = operations(concept1,concept2,stereotype1,stereotype2,caseSensitive, iterations, vectors, word2index, distribution)
results = myOperations.getPValueAndEffect()
print(biasType , ": p-value: ", results[0] ,"  ---  effectSize: ", results[1] )
