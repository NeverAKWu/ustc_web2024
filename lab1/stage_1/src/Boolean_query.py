import argparse
import json
import math
from index_decompress import IndexDecompressor
import time

class SkipList:
    def __init__(self, elements):
        self.elements = elements

    def add_skip_pointers(postings):
        skip_distance = int(math.sqrt(len(postings)))
        if skip_distance > 1:
            for i in range(0, len(postings), skip_distance):
                if i + skip_distance < len(postings):
                    postings[i] = [postings[i], i + skip_distance]
                else:
                    postings[i] = [postings[i], None]
        return postings
    
    def merge(self, other, op):
        result = []
        i, j = 0, 0
        while i < len(self.elements) and j < len(other.elements):
            if isinstance(self.elements[i], list):
                self_id = self.elements[i][0]
                self_skip = self.elements[i][1]
            else:
                self_id = self.elements[i]
                self_skip = None

            if isinstance(other.elements[j], list):
                other_id = other.elements[j][0]
                other_skip = other.elements[j][1]
            else:
                other_id = other.elements[j]
                other_skip = None

            if op == 'AND':
                if self_id == other_id:
                    result.append(self_id)
                    i += 1
                    j += 1
                elif self_id < other_id:
                    if(self_skip is not None and self.elements[self_skip][0] <= other_id):
                        i = self_skip
                    else:
                        i += 1
                else:
                    if(other_skip is not None and other.elements[other_skip][0] <= self_id):
                        j = other_skip
                    else:
                        j += 1
            elif op == 'OR':
                if self_id == other_id:
                    result.append(self_id)
                    i += 1
                    j += 1
                elif self_id < other_id:
                    result.append(self_id)
                    i += 1
                else:
                    result.append(other_id)
                    j += 1
            elif op == 'AND_NOT':
                if self_id == other_id:
                    i += 1
                    j += 1
                elif self_id < other_id:
                    result.append(self_id)
                    i += 1
                else:
                    j += 1
            
        if op == 'OR':
            while i < len(self.elements):
                if isinstance(self.elements[i], list):
                    result.append(self.elements[i][0])
                else:
                    result.append(self.elements[i])
                i += 1
            while j < len(other.elements):
                if isinstance(other.elements[j], list):
                    result.append(other.elements[j][0])
                else:
                    result.append(other.elements[j])
                j += 1

        return result

class BooleanQuery:
    def __init__(self, index_file, file_type):
        if file_type == 'normal':
            with open(index_file, 'r', encoding='utf-8') as f:
                self.inverted_index = json.load(f)
        else:
            decompressor = IndexDecompressor(index_file)
            if file_type == 'delta':
                self.inverted_index = decompressor.load_from_delta_file()
            else:
                self.inverted_index = decompressor.load_from_binary_file()

    def get_terms(self, query_string):
        terms = []
        term = ''
        for char in query_string:
            if char == '（':
                char = '('
            elif char == '）':
                char = ')'
            if char in ['(', ')']:
                if term:
                    terms.append(term)
                    term = ''
                terms.append(char)
            elif char.isspace():
                if term:
                    terms.append(term)
                    term = ''
            else:
                term += char
        if term:
            terms.append(term)
        
        terms = [term.upper() if term in ['and', 'or', 'and_not'] else term for term in terms]
        return terms
    
    def query(self, query_string):
        terms = self.get_terms(query_string)

        if len(terms) == 1:
            return [doc[0] if isinstance(doc, list) else doc for doc in self.inverted_index.get(terms[0], [])]

        stack = []
        current = None
        for term in terms:
            if term == '(':
                stack.append((None, None))
                current = None
            elif term == ')':
                stack.pop()
                if stack[-1][1] is None:
                    stack.pop()
                if stack:
                    last_current, last_op = stack.pop()
                    if last_op is not None:
                        current = SkipList(last_current.merge(current, last_op))
                        current.elements = SkipList.add_skip_pointers(current.elements)
                stack.append((current, None))
            elif term in ['AND', 'OR', 'AND_NOT']:
                stack[-1] = (stack[-1][0], term)
            else:
                if stack and stack[-1][1] is not None:
                    last_current, last_op = stack.pop()
                    current = SkipList(last_current.merge(SkipList(self.inverted_index.get(term, [])), last_op))
                    current.elements = SkipList.add_skip_pointers(current.elements)
                else:
                    current = SkipList(self.inverted_index.get(term, []))
                stack.append((current, None))
        
        result = stack[-1][0]
        return [doc[0] if isinstance(doc, list) else doc for doc in result.elements]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boolean Query Program")
    parser.add_argument('--index_file', type=str, help='Path to the inverted index file')
    parser.add_argument('-d', '--delta', action='store_true', help='Use delta encoded index file')
    parser.add_argument('-v', '--varbyte', action='store_true', help='Use varbyte encoded index file')
    parser.add_argument('-b', '--books', action='store_true', help='Query books')
    parser.add_argument('-m', '--movies', action='store_true', help='Query movies')
    parser.add_argument('-t', '--time', action='store_true', help='Print time taken to load index and execute query')
    args = parser.parse_args()

    if args.delta:
        if args.movies:
            index_file = '../data/result/movie_delta_encode.json'
        else:
            index_file = '../data/result/book_delta_encode.json'
        file_type = 'delta'
    elif args.varbyte:
        if args.movies:
            index_file = '../data/result/movie_varbyte_encode.idx'
        else:
            index_file = '../data/result/book_varbyte_encode.idx'
        file_type = 'varbyte'
    else:
        if args.movies:
            index_file = '../data/result/movie_inverted_index.json'
        else:
            index_file = '../data/result/book_inverted_index.json'
        file_type = 'normal'

    if args.index_file:
        index_file = args.index_file

    if args.time:
        start_time = time.time()
    bq = BooleanQuery(index_file, file_type)
    if args.time:
        end_time = time.time()
        print("Index loaded in {:.6f} seconds".format(end_time - start_time))

    while True:
        query_string = input("Enter your boolean query (or type 'exit' to quit): ")
        if query_string.lower() == 'exit':
            break
        if args.time:
            start_time = time.time()
        result = bq.query(query_string)
        if args.time:
            end_time = time.time()
            print("Query excuted in {:.6f} seconds".format(end_time - start_time))
        print("Documents matching the query:", result)
