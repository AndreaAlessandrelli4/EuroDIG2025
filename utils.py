import os
import torch
from weaviate.gql.get import HybridFusion
import weaviate
from FlagEmbedding import FlagReranker
import unicodedata
import re
from weaviate.classes.query import MetadataQuery
from weaviate.classes.config import Configure, Property, DataType
import weaviate, os
import weaviate.classes as wvc
import pdfplumber



##########################################################################################
##########################          READING                ###############################
##########################################################################################
def extract_text_from_pdf(uploaded_file):
    """Estrae il testo da un PDF usando pdfplumber."""
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_txt(uploaded_file):
    """Legge il contenuto di un file di testo."""
    return uploaded_file.read().decode("utf-8")


def cleaning_page(text):
    #text=re.sub(r'\s+',' ',text)
    text=re.sub(r"""Pagina \d+ di \d+|pagina \d+ di \d+|Pag. \d+ di \d+|
                                Pag. \d+|Pagina \d+|pagina \d+""",'',text)
    return text.strip()


##########################################################################################
##########################          WEAVIATE               ###############################
##########################################################################################
# Set these environment variables
#URL = os.getenv("WEAVIATE_URL")
#APIKEY = os.getenv("WEAVIATE_API_KEY")

# full_docx_NO_summary
# docx_WITH_summary



def connection(URL, APIKEY):
    # Connect to Weaviate Cloud
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=URL,
        auth_credentials=wvc.init.Auth.api_key(APIKEY),
    )
    if client.is_ready():
        print('Connection success')
        return client
    else:
        return print("Connection problems, try again in a few minutes")
##########################################################################################
##########################################################################################
##########################################################################################




##########################################################################################
##########################          CLEANING               ###############################
##########################################################################################
# remove some end and start of page strings
def cleaning(text):
    text=re.sub(r'\s+',' ',text)
    text=re.sub(r'\n+','\n',text)
    return text

def remove_unicode_characters(string):
    normalizzed_string = unicodedata.normalize('NFKD', string)
    try:
        ascii_string = normalizzed_string.encode('ascii', 'ignore').decode('ascii')
        ascii_string = ascii_string.replace("\\\\","-").replace("\\","-").replace("/","-").replace("//","-")
    except:
        ascii_string = string.replace("\\\\","-").replace("\\","-").replace("/","-").replace("//","-")
    return ascii_string


##########################################################################################
##########################          MODELLI                ###############################
##########################################################################################
from sentence_transformers import SentenceTransformer
#model_rag = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device="cuda")
#reranker = FlagReranker('BAAI/bge-reranker-large')#, use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation  

def generate_embeddings(entry, model):
    embedding = model.encode(entry, convert_to_numpy=True, normalize_embeddings=True)
    return embedding


def get_sorted_indices(lst):
    # Step 1: Pair each element with its index
    indexed_list = [(value, index) for index, value in enumerate(lst)]
    # Step 2: Sort the list of tuples based on the first element (the value)
    sorted_indexed_list = sorted(indexed_list, key=lambda x: x[0], reverse=True)
    # Step 3: Extract the indices from the sorted list
    sorted_indices = [index for value, index in sorted_indexed_list]
    return sorted_indices


def reranking(tot_results, chuncks, query, max_doc, reranker):
    list_mat= [[query,t.strip()] for t in chuncks]
    score = reranker.compute_score(list_mat)
    sorted_pos=get_sorted_indices(score)
    temp = []
    sorted_pos_unique = []
    for i in sorted_pos:
        if tot_results[i]['id_originale'] not in temp:
            temp.append(tot_results[i]['id_originale'])
            sorted_pos_unique.append(i)
        if len(sorted_pos_unique)==max_doc:
            break
    reranked_only_summary = [tot_results[i]['partial_text'] for i in sorted_pos_unique]
    #reranked_only_title = [tot_results[i]['identification'] for i in sorted_pos_unique]
    reranked_text = [tot_results[i]['full_text'] for i in sorted_pos_unique]
    reranked_tot = [tot_results[i] for i in sorted_pos_unique]
    
    return reranked_only_summary,  reranked_text, reranked_tot#, reranked_only_title

##########################################################################################
##########################          SEARCH                 ###############################
##########################################################################################
def wea_materials(query, model_emb, client, collection_name, type):
    collection = client.collections.get(collection_name)
    coll = []
    response_tot = []
    text_sent_tot = []
    text_sent=[]
    text_summary = []
    if type == 'key':
        responses = []
        response = collection.query.bm25(
                                            query=query,
                                            query_properties=['partial_text^2', 'full_text'],
                                            limit=50,
                                            return_metadata=MetadataQuery(distance=True, score=True)
                                        )
        for o in response.objects:
            responses.append(o)
        # As we use different kinds of searches and there may be duplication of documents, we ensure that each chunk appears only once in the output.
        for o in responses:
            # unique identification of chunck
            temp = o.properties['id_split']
            if temp not in coll:
                # list to check the unicity of chuncks
                coll.append(temp)
                text_sent.append(o.properties['partial_text'])
                text_sent_tot.append(o.properties['full_text'])
                text_summary.append(o.properties['summary'])
                response_tot.append(o.properties)
    else:
        for a in [0.0, 0.5, 1.0]:
            responses = []
            response = collection.query.hybrid(
                                                query=query,
                                                query_properties=['partial_text^2', 'full_text'],
                                                vector={'partial_text':generate_embeddings(query, model_emb)},
                                                target_vector=['partial_text'],  # Specify the target vectors
                                                alpha=a,
                                                limit=50,
                                                return_metadata=MetadataQuery(distance=True, score=True)
                                            )
            for o in response.objects:
                responses.append(o)
            # As we use different kinds of searches and there may be duplication of documents, we ensure that each chunk appears only once in the output.
            for o in responses:
                # unique identification of chunck
                temp = o.properties['id_split']
                if temp not in coll:
                    # list to check the unicity of chuncks
                    coll.append(temp)
                    text_sent.append(o.properties['partial_text'])
                    text_sent_tot.append(o.properties['full_text'])
                    text_summary.append(o.properties['summary'])
                    response_tot.append(o.properties)
    return response_tot, text_sent, text_sent_tot, text_summary



#This function generates a tuple of sorted lists
#the length of the result list is governed by the variable max_doc (set to 10 by default)
#in each position of the tuple you find:
#  * 1st position: the summary of the legal ruling
#  * 2nd position:the unique identification (title) of the legal ruling
#  * 3rd: position:the orignial text of the legal ruling

def quering_database(query, reranker, model_emb, client, collection_name, type, max_doc):
    # check on max_doc (it is better to choose values between 0 and 100)
    if max_doc <0 or max_doc > 100:
        max_doc = 100
    tot_results, chuncks, textx, summaries = wea_materials(query, model_emb, client, collection_name, type)
    #re_summary, re_title, re_text, re_tot = reranking(tot_results, chuncks, query, max_doc, reranker)
    if len(tot_results)>=1:
        re_summary, re_text, re_tot = reranking(tot_results, chuncks, query, max_doc, reranker)
    else: 
        re_summary, re_text, re_tot = [[],[],[]]
    
    return re_summary, re_text, re_tot#, re_title