import json
import os
import thulac
import pickle as pkl
import numpy as np

def convert_json(query_name='oil price', js_path='data/oil/oil_clean.json'):
	fact = []
	query_clean = []
	# oil_text = json.load(open('test.json','r'))
	oil_text_dict = json.load(open(js_path,'r'))
	# print(oil_text_dict)
	stop_word_file = 'data/oil/ENstopwords.txt'
	stop_words = load_stop_words(stop_word_file)
	fact_clean = []
	fact = []
	for arxiv_id, text in oil_text_dict.items():
		# print(arxiv_id)
		# words = thu1.cut(text, text=True)
		words = text.split(' ')
		for i in range(len(words)):
			words[i] = clean_word(words[i])
		cur_clean = []
		for word in words:
			# print(stop_words)
			if (not word in stop_words) and not havenumber(word):
				cur_clean.append(word)

		fact_clean.append(cur_clean)
		query_clean.append(query_name)
		fact.append(text)
		
	return fact, fact_clean, query_clean
	
exclude_list = [',',':',';','\'','.',']','(',')','$','emph','infty','textbf','-','`','/','[']
def clean_word(s):
	for exculde_char in exclude_list:
		s = s.replace(exculde_char,'')
	return s

def load_stop_words(stop_word_file):
	stop_words = []
	for line in open(stop_word_file, 'r'):
		if line.strip()[0:1] != "#":
			for word in line.split():
				stop_words.append(word)
	return stop_words
	
def havenumber(s):
	f = False
	for i in s:
		if (i >= '0') and (i <= '9'):
			f = True
			break
	return f

if not os.path.exists('data/oil/preprocessed_data.pkl'):
	thu1 = thulac.thulac(seg_only=True)
	fact1, fact_clean1, query_clean1 = convert_json('oil price', 'data/oil/oil_clean.json')
	fact2, fact_clean2, query_clean2 = convert_json('gas price', 'data/oil/gas_clean.json')
	fact = fact1 + fact2
	fact_clean = fact_clean1 + fact_clean2
	query_clean = query_clean1 + query_clean2

	data = {
		'fact_original':fact,
		'fact_clean':fact_clean,
		'query_clean': query_clean
	}
	# print(data['fact_clean'])
	with open('data/oil/preprocessed_data.pkl', 'wb') as f:
		pkl.dump(data, f)
else:
	with open('data/oil/preprocessed_data.pkl', 'rb') as f:
		data = pkl.load(f)
		# print(data['fact_clean'])
print('Done preparing data')

if not os.path.exists('data/oil/used_wv.pkl'):
	from gensim.models import KeyedVectors
	wv_from_text = KeyedVectors.load_word2vec_format('data/vectors.txt', binary=False)
	print('Done loading word embedding')

	used_wv = {}
	oov = set()
	exact_oov = 0
	for i in data['fact_clean']:
		for j in i:
			if not j in used_wv:
				if j in wv_from_text.vocab:
					used_wv[j] = wv_from_text.word_vec(j).tolist()
				else:
					oov.add(j)
					ebd = []
					for k in j:
						if k in wv_from_text.vocab:
							ebd.append(wv_from_text.word_vec(k).tolist())
					if len(ebd) > 0:
						used_wv[j] = np.mean(np.array(ebd), axis=0)
					else:
						used_wv[j] = np.random.rand(300) * 2 - 1
						exact_oov += 1
	print(len(used_wv), len(oov), exact_oov)
	with open('data/oil/used_wv.pkl', 'wb') as f:
		pkl.dump(used_wv, f)
else:
	with open('data/oil/used_wv.pkl', 'rb') as f:
		used_wv = pkl.load(f)
		# print(used_wv.keys())
print('Done preparing word vectors')

if not os.path.exists('data/oil/vocab.pkl'):
	MAX_VOCAB_SIZE = 10000
	UNK, PAD = '<UNK>', '<PAD>'

	def build_vocab(text, max_size, min_freq):
		vocab_dic = {}
		for line in text:
			for word in line:
				vocab_dic[word] = vocab_dic.get(word, 0) + 1
		vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[
			:max_size]
		vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
		vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
		return vocab_dic

	word_to_id = build_vocab(data['fact_clean'], max_size=MAX_VOCAB_SIZE, min_freq=1)
	pkl.dump(word_to_id, open('data/oil/vocab.pkl', 'wb'))

	embeddings = np.random.rand(len(word_to_id), 300)
	with open('data/oil/used_wv.pkl', 'rb') as f:
		wv = pkl.load(f)
	for i in wv:
		if i in word_to_id:
			idx = word_to_id[i]
			embeddings[idx] = wv[i]
	np.savez_compressed('data/oil/embeddings.npz', embeddings=embeddings)
	print('Done generating vocab and embeddings')




	