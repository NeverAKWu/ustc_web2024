import json
import struct
import argparse

class IndexCompressor:
    def __init__(self, index_file):
        with open(index_file, 'r', encoding='utf-8') as f:
            self.inverted_index = json.load(f)

    def delta_compress(self):
        self.delta_result = {}
        for term in self.inverted_index:
            self.delta_result[term] = self.delta_encode_postings(self.inverted_index[term])
        return self.delta_result

    def delta_encode_postings(self, postings):
        result = []
        result.append(postings[0])
        prev = postings[0][0] if isinstance(postings[0], list) else postings[0]
        for i in range(1, len(postings)):
            if isinstance(postings[i], list):
                result.append([postings[i][0] - prev, postings[i][1]])
                prev = postings[i][0]
            else:
                result.append(postings[i] - prev)
                prev = postings[i]
        return result

    def varbyte_encode(self, number):
        bytes_list = []
        while True:
            bytes_list.insert(0, number % 128)
            if number < 128:
                break
            number //= 128
        bytes_list[-1] += 128
        return bytes_list

    def varbyte_encode_postings(self, postings):
        encoded_postings = []
        for i in range(0,len(postings)):
            if isinstance(postings[i], list):
                encoded_postings.extend(self.varbyte_encode(postings[i][0]))
            else:
                encoded_postings.extend(self.varbyte_encode(postings[i]))
        return encoded_postings

    def varbyte_compress(self):
        varbyte_encoded = {}
        for term in self.delta_result:
            varbyte_encoded[term] = self.varbyte_encode_postings(self.delta_result[term])
        return varbyte_encoded
    
    def save_to_binary_file(self, varbyte_encoded, output_file):
        with open(output_file, 'wb') as f:
            for term, postings in varbyte_encoded.items():
                term_bytes = term.encode('utf-8')
                term_length = len(term_bytes)
                postings_length = len(postings)
                
                # Write term length and term bytes
                f.write(struct.pack('I', term_length))
                f.write(term_bytes)
                
                # Write postings length and postings bytes
                f.write(struct.pack('I', postings_length))
                f.write(bytearray(postings))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Index Compressor')
    parser.add_argument('--index_file', type=str, help='Path to the inverted index file')
    parser.add_argument('-b', '--books', action='store_true', help='Compress book index')
    parser.add_argument('-m', '--movies', action='store_true', help='Compress movie index')
    args = parser.parse_args()

    if args.movies:
        index_file = '../data/result/movie_inverted_index.json'
        output_file_delta = '../data/result/movie_delta_encode.json'
        output_file_varbyte = '../data/result/movie_varbyte_encode.idx'
    else:
        index_file = '../data/result/book_inverted_index.json'
        output_file_delta = '../data/result/book_delta_encode.json'
        output_file_varbyte = '../data/result/book_varbyte_encode.idx'

    if args.index_file:
        index_file = args.index_file

    index_compressor = IndexCompressor(index_file)
    delta_compressed_index = index_compressor.delta_compress()
    varbyte_compressed_index = index_compressor.varbyte_compress()
    with open(output_file_delta, 'w', encoding='utf-8') as f:
        json.dump(delta_compressed_index, f, ensure_ascii=False, indent=4)
    index_compressor.save_to_binary_file(varbyte_compressed_index, output_file_varbyte)
    
    

