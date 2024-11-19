import struct
import json
import argparse
from inverted_index import IndexBuilder

class IndexDecompressor:
    def __init__(self, index_file):
        self.index_file = index_file

    def delta_decode_postings(self, postings):
        result = []
        result.append(postings[0])
        prev = postings[0][0] if isinstance(postings[0], list) else postings[0]
        for i in range(1, len(postings)):
            if isinstance(postings[i], list):
                result.append([postings[i][0] + prev, postings[i][1]])
                prev = result[-1][0]
            else:
                result.append(postings[i] + prev)
                prev = result[-1]
        return result

    def varbyte_decode(self, bytes_list):
        numbers = []
        n = 0
        for byte in bytes_list:
            if byte < 128:
                n = 128 * n + byte    
            else:
                n = 128 * n + (byte - 128)
                numbers.append(n)
                n = 0
        return numbers

    def load_from_binary_file(self):
        inverted_index = {}
        with open(self.index_file, 'rb') as f:
            while True:
                term_length_bytes = f.read(4)
                if not term_length_bytes:
                    break
                term_length = struct.unpack('I', term_length_bytes)[0]
                term = str(f.read(term_length).decode('utf-8'))
                postings_length_bytes = f.read(4)
                postings_length = struct.unpack('I', postings_length_bytes)[0]
                postings = f.read(postings_length)
                decoded_postings = self.varbyte_decode(postings)
                decoded_postings = IndexBuilder.add_skip_pointers(decoded_postings)
                inverted_index[term] = self.delta_decode_postings(decoded_postings)
        return inverted_index
    
    def load_from_delta_file(self):
        inverted_index = {}
        with open(self.index_file, 'r', encoding='utf-8') as f:
            inverted_index = json.load(f)
        for term in inverted_index:
            inverted_index[term] = self.delta_decode_postings(inverted_index[term])
        return inverted_index

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Index Compressor')
    parser.add_argument('--index_file', type=str, help='Path to the Compressed index file')
    parser.add_argument('-b', '--books', action='store_true', help='Decompress book index')
    parser.add_argument('-m', '--movies', action='store_true', help='Decompress movie index')
    args = parser.parse_args()

    if args.movies:
        index_file_path = '../data/result/movie_varbyte_encode.idx'
        output_file = '../data/result/movie_decompressed.json'
    else:
        index_file_path = '../data/result/book_varbyte_encode.idx'
        output_file = '../data/result/book_decompressed.json'

    decompressor = IndexDecompressor(index_file_path)
    inverted_index = decompressor.load_from_binary_file()

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(inverted_index, f, ensure_ascii=False, indent=4)

    