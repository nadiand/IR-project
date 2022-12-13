# import relevant libraries
from beir.datasets.data_loader import GenericDataLoader
from beir import util
import pathlib, os
import sys
from pyserini.search.lucene import LuceneSearcher 
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
import random
import math
import gc

from sentence_transformers import losses, SentenceTransformer
from beir import util, LoggingHandler
from beir.retrieval.train import TrainRetriever
import torch

def DCG(query_relevancy_labels, k):
  ### calculate the Discounted Cumulative Gain @k
  result = 0

  for i in range(0, min(k, len(query_relevancy_labels))):
    if math.log2(i+2) != 0:
      result += query_relevancy_labels[i] / math.log2(i+2)

  return result


def NDCG(query_relevancy_labels, qrels, k):
  ### calculate the Normalized Discounted Cumulative Gain @k
  numerator = DCG(query_relevancy_labels, k)
  denominator = DCG(qrels, k)

  if denominator != 0:
    return numerator / denominator
  else:
    return 0

def NDCG_allqueries(query_relevancy_labels, qrels):
  ### calculate the Normalized Discounted Cumulative Gain @k for each ranking and qrels in a dictionary of multiple.
  score = 0

  # for each query
  for key in query_relevancy_labels.keys():

    # sort the dictionary on relevancy
    dictionary = query_relevancy_labels[key]
    items = list(dictionary.items())
    dictionary = dict(sorted(items, reverse=True, key= lambda x: x[1]))

    # get the labels and qrels such that NDCG@k can be calculated 
    dictionary_qrels = qrels[key]
    qrels_labels = list(dictionary_qrels.values())
    qrels_labels.sort(reverse=True)
    ranking = list(dictionary.keys()) 
    labels = list(map(lambda x: dictionary_qrels[x] if x in dictionary_qrels.keys() else 0, ranking))

    # calculate the NDCG@10
    result = NDCG(labels, qrels_labels, 10)
    score += result
  return score/len(query_relevancy_labels)
    
def load_dataset_traintest(dataset):
  ### load the train and test set for a given dataset
  
  url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
  out_dir = os.path.join(pathlib.Path(os.path.dirname(sys.executable)).parent.absolute(), "datasets")
  data_path = util.download_and_unzip(url, out_dir)
  train_corpus, train_queries, train_qrels = GenericDataLoader(data_path).load(split="train")

  url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
  out_dir = os.path.join(pathlib.Path(os.path.dirname(sys.executable)).parent.absolute(), "datasets")
  data_path = util.download_and_unzip(url, out_dir)
  test_corpus, test_queries, test_qrels = GenericDataLoader(data_path).load(split="test")

  return train_corpus, train_queries, train_qrels, test_corpus, test_queries, test_qrels

def load_dataset(dataset):
  ### load just the test set for a given dataset
  
  url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
  out_dir = os.path.join(pathlib.Path(os.path.dirname(sys.executable)).parent.absolute(), "datasets")
  data_path = util.download_and_unzip(url, out_dir)
  test_corpus, test_queries, test_qrels = GenericDataLoader(data_path).load(split="test")

  return test_corpus, test_queries, test_qrels


def finetune(corpus, queries, qrels, pyserini_qrels, nr_queries=1000, num_epochs = 3, evaluation_steps = 1000):
  ####### finetune BERT model on corpus, queries, qrels and pyserini_qrels. Possibly given some number of queries, epochs and evaluation steps #######
  
  ### Start with initial BERT model and set a batch-size on the train retriever
  model_finetuned = SentenceTransformer("msmarco-distilbert-base-v3")
  retriever_finetuned = TrainRetriever(model=model_finetuned, batch_size=32)

  ### Prepare training samples
  queries = dict([(x, queries[x]) for x in list(queries.keys())[:nr_queries]])
  qrels = dict([(x, qrels[x]) for x in list(qrels.keys()) if x in queries])
  pyserini_qrels = [(x, pyserini_qrels[x]) for x in list(pyserini_qrels.keys()) if x in queries]

  # Add negative examples based on pyserini ranking
  for query_key, pyqrel in pyserini_qrels:
    for item in pyqrel.keys():
      if item not in list(qrels[query_key]):
        qrels[query_key][item] = 0

  train_samples = retriever_finetuned.load_train(corpus, queries, qrels)
  train_dataloader = retriever_finetuned.prepare_train(train_samples, shuffle=True)

  ### Training SBERT with MultipleNegativesRanking as the loss function
  train_loss = losses.MultipleNegativesRankingLoss(model=retriever_finetuned.model)

  retriever_finetuned.fit(train_objectives=[(train_dataloader, train_loss)], 
                  epochs=num_epochs,
                  evaluation_steps=evaluation_steps,
                  use_amp=False)
  
  ### Save the model
  save_path = './tmp_model/'
  retriever_finetuned.model.save(save_path, model_name='tmp')

  # return model path
  return save_path

def get_pyeserini(index, queries):
  ### Get the standard ranking with pyserini
  
  searcher = LuceneSearcher.from_prebuilt_index(index)
  pyserini_results_nfcorpus = {}
  for q in queries:
    hits = searcher.search(queries[q], 100) # get the 100 top documents from the covid data for the given query
    query_hits = {}
    for h in hits:
      query_hits[h.docid] = h.score
    pyserini_results_nfcorpus[q] = query_hits # create the correct data type to pass to bert

  return pyserini_results_nfcorpus

def standard_bert(corpus, queries, pyserini):
  ### Rerank with standard BERT
  model = DRES(models.SentenceBERT("msmarco-distilbert-base-v3"), batch_size=128)
  retriever = EvaluateRetrieval(model, score_function="cos_sim")
  reranked_results = retriever.rerank(corpus, queries, pyserini, 99)
  return reranked_results

def finetuned_bert(model, corpus, queries, pyserini):
  ### Rerank with finetuned BERT
  model_finetuned = DRES(models.SentenceBERT(model), batch_size=128)
  retriever_finetuned = EvaluateRetrieval(model_finetuned, score_function="cos_sim")
  reranked_results = retriever_finetuned.rerank(corpus, queries, pyserini, 99)
  return reranked_results

def get_NDCGs(pyserini, standard, finetuned, qrels):
  ### print(qrels)
  relevancy_judgements_pyserini = pyserini
  relevancy_judgements_standard = standard
  relevancy_judgements_finetuned = finetuned

  print("pyserini score: ", NDCG_allqueries(relevancy_judgements_pyserini, qrels))
  print("reranked score: ", NDCG_allqueries(relevancy_judgements_standard, qrels))
  print("reranked / finetuned score: ", NDCG_allqueries(relevancy_judgements_finetuned, qrels))
  print()

def run_tests(finetune_dataset, index, test_datasets_indexes):
  ### run BM25, BERT and BERT-F on the given datasets
  
  # load the train data and finetune our model on it
  train_corpus, train_queries, train_qrels, test_corpus, test_queries, test_qrels = load_dataset_traintest(finetune_dataset)
  pyserini_qrels = get_pyeserini(index, train_queries)
  model = finetune(train_corpus, train_queries, train_qrels, pyserini_qrels, nr_queries=1000, num_epochs = 5, evaluation_steps = 100) #TODO rmeove

  def run_test(model, corpus, queries, index, qrels):
    ### function to run one test
    
    # rerank using pyserini
    pyserini_results_nfcorpus = get_pyeserini(index, queries)
    
    # rerank using BERT
    reranked_results = standard_bert(corpus, queries, pyserini_results_nfcorpus)
    
    # rerank using BERT-F
    reranked_results_finetuned = finetuned_bert(model, corpus, queries, pyserini_results_nfcorpus)
    
    # calculate and display average NDCG@10 for all 3 rankings
    get_NDCGs(pyserini_results_nfcorpus, reranked_results, reranked_results_finetuned, qrels)
    
    # free used memory because Python did not do this automatically :D
    gc.collect()

  # run validation on train-set to measure whether our fine-tuning works as expected
  print(f"Results for {finetune_dataset}-train with Bert finetuned on {finetune_dataset}")
  run_test(model, train_corpus, train_queries,index, train_qrels)

  # run validation on test-set related to our training data
  print(f"Results for {finetune_dataset} with Bert finetuned on {finetune_dataset}")
  run_test(model, test_corpus, test_queries, index, test_qrels)

  # run test with our finetuned model on all other given test sets
  for test_dataset, test_index in test_datasets_indexes:
    try:
      print(f"Results for {test_dataset} with Bert finetuned on {finetune_dataset}")
      test_corpus, test_queries, test_qrels = load_dataset(test_dataset)
      run_test(model, test_corpus, test_queries, test_index, test_qrels)
    except:
      print("failed")
      
# define finetune and test dataset for our experiments
finetune_dataset, index = "nfcorpus", "beir-v1.0.0-nfcorpus-flat"
test_datasets = [
    ("quora", "beir-v1.0.0-quora-flat"),
    ("trec-covid", "beir-v1.0.0-trec-covid-flat"),
    ("arguana", "beir-v1.0.0-arguana-flat"),
    ("scifact", "beir-v1.0.0-scifact-flat"),
    ("fiqa", "beir-v1.0.0-fiqa-flat")
]

# run tests on our given datasets
run_tests(finetune_dataset, index, test_datasets)
