import sys
import codecs
import json
import Stemmer
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
import collections
import glob

stemmer = Stemmer.Stemmer('spanish')


def cleanDoc(doc):
    stopset = set(stopwords.words('spanish'))
    stemmer = Stemmer.Stemmer('spanish')
    tokens = WordPunctTokenizer().tokenize(doc)
    clean = [token.lower() for token in tokens if
             token.lower() not in stopset and len(token) > 2]
    final = stemmer.stemWords(clean)
    return final


def load():
    files = glob.glob('corpus/spanishText_*')
    documents = collections.OrderedDict()
    unprocessed_documents = collections.OrderedDict()

    for f in files:
        print('File %s f' % f)
        file = codecs.open('%s' % f, 'r', encoding='ISO-8859-1')
        soup = BeautifulSoup(file.read(), "lxml")

        for a in soup.find_all('doc', title=True):
            content = a.text.replace('\n',' ').replace('ENDOFARTICLE', '').strip().rstrip()
            documents[a['title']] = cleanDoc(content)
            unprocessed_documents[a['title']] = content

    print('Writing unprocessed documents to json\n')
    with open('path_pre_process/unprocessed_docs.json', 'w') as f:
        f.write(json.dumps(unprocessed_documents))

    return documents


def load_unprocessed_documents():
    unprocessed_documents = collections.OrderedDict()
    with open('path_pre_process/unprocessed_docs.json', 'r') as f:
        unprocessed_documents = json.loads(f.read())
    return unprocessed_documents
