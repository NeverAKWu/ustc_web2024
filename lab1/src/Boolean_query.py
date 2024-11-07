import json
import math

class SkipList:
    def __init__(self, elements):
        self.elements = elements

    def add_skip_pointers(postings):
        skip_distance = int(math.sqrt(len(postings)))
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
    def __init__(self, index_file):
        with open(index_file, 'r') as f:
            self.inverted_index = json.load(f)

    def query(self, query_string):
        terms = query_string.split()
        if len(terms) == 1:
            return [doc[0] if isinstance(doc, list) else doc for doc in self.inverted_index.get(terms[0], [])]
        
        result = None
        op = None
        for term in terms:
            if term in ['AND', 'OR', 'AND_NOT']:
                op = term
            else:
                if result is None:
                    result = SkipList(self.inverted_index.get(term, []))
                else:
                    result = SkipList(result.merge(SkipList(self.inverted_index.get(term, [])), op))
                    result.elements = SkipList.add_skip_pointers(result.elements)
            
        return [doc[0] if isinstance(doc, list) else doc for doc in result.elements]

if __name__ == "__main__":
    index_file = '../data/result/book_inverted_index.json'
    bq = BooleanQuery(index_file)

    while True:
        query_string = input("Enter your boolean query (or type 'exit' to quit): ")
        if query_string.lower() == 'exit':
            break
        result = bq.query(query_string)
        print("Documents matching the query:", result)
