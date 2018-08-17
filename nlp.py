import re
import nltk
import string
from nltk.stem.porter import PorterStemmer

en_stopwords = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst",
"amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand",
"behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do",
"done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill",
"find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here",
"hereafter", "hereby", "herein", "hereupon", "hers", "herself","him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last",
"latter", "latterly", "least", "less", "ltd", "made","many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither",
"never", "nevertheless", "next", "nine", "nobody", "none", "noone", "nor", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise",
"our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side",
"since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence",
"there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards",
"twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein",
"whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]

punctuations = set(string.punctuation)
porter_stemmer = PorterStemmer()


def read_file(file_name):
    with open(file_name, 'r') as file:
        content = file.read()
    print("[INFO] processing file : " + file_name + " ...")
    content = content.replace('\n','').replace('.', '')
    return content


def process_document(docu, if_stemming):
    # case fold
    docu = docu.lower()
    # remove all punctuation
    rmd_punc = ''.join(s for s in docu if s not in punctuations)
    # remove all digits
    rmd_digit = re.sub(r'\d+', '', rmd_punc)
    # tokenize
    word_tokens = nltk.word_tokenize(rmd_digit)
    # filter stop words
    tokens = [t for t in word_tokens if t not in en_stopwords]

    if if_stemming:
        stemmed_tokens = []
        for t in tokens:
            stemmed_tokens.append(porter_stemmer.stem(t))
        tokens = stemmed_tokens

    return " ".join(tokens)


def word_to_vect(token, w2v_model):
    word_vec = None
    try:
        word_vec = w2v_model.wv[token]
    except KeyError:
        pass

    return word_vec
