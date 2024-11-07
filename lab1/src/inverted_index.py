import pandas as pd
from collections import defaultdict
import json
import math

def add_skip_pointers(postings):
    skip_distance = int(math.sqrt(len(postings)))
    for i in range(0, len(postings), skip_distance):
        if i + skip_distance < len(postings):
            postings[i] = [postings[i], i + skip_distance]
        else:
            postings[i] = [postings[i], None]
    return postings

def build_inverted_index(csv_file):
    inverted_index = defaultdict(list)
    
    df = pd.read_csv(csv_file)
    for _, row in df.iterrows():
        doc_id = row.iloc[0]
        terms = eval(row.iloc[1])
        for term in terms:
            inverted_index[term].append(doc_id)
    
    # Sort the postings and add skip pointers
    for term in inverted_index:
        inverted_index[term] = sorted(inverted_index[term])
        inverted_index[term] = add_skip_pointers(inverted_index[term])
    
    return inverted_index

def save_inverted_index(inverted_index, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(inverted_index, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    book_input_file_path = '../data/result/book_split_jieba.csv'
    book_output_file_path = '../data/result/book_inverted_index.json'

    movie_input_file_path = '../data/result/movie_split_jieba.csv'
    movie_output_file_path = '../data/result/movie_inverted_index.json'
    
    book_inverted_index = build_inverted_index(book_input_file_path)
    save_inverted_index(book_inverted_index, book_output_file_path)

    movie_inverted_index = build_inverted_index(movie_input_file_path)
    save_inverted_index(movie_inverted_index, movie_output_file_path)
