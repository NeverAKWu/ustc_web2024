import pandas as pd
from collections import defaultdict
import json
import math
import argparse

class IndexBuilder:
    def add_skip_pointers(postings):
        skip_distance = int(math.sqrt(len(postings)))
        if skip_distance > 1:
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
            inverted_index[term] = IndexBuilder.add_skip_pointers(inverted_index[term])
        
        return inverted_index

    def save_inverted_index(inverted_index, output_file):
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(inverted_index, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inverted Index Builder')
    parser.add_argument('--input_file', type=str, help='Path to the words split file')
    parser.add_argument('-b', '--books', action='store_true', help='Build book inverted index')
    parser.add_argument('-m', '--movies', action='store_true', help='Build movie inverted index')
    args = parser.parse_args()

    if args.movies:
        input_file_path = '../data/result/movie_split_jieba.csv'
        output_file_path = '../data/result/movie_inverted_index.json'
    else:
        input_file_path = '../data/result/book_split_jieba.csv'
        output_file_path = '../data/result/book_inverted_index.json'

    book_inverted_index = IndexBuilder.build_inverted_index(input_file_path)
    IndexBuilder.save_inverted_index(book_inverted_index, output_file_path)
