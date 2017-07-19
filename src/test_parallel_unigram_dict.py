import os
import sys

from probe2vec.unigram_dictionary import UnigramDictionary        
        
    
# load the parallel derived UnigramDict
p = UnigramDictionary()
p_load = os.path.expanduser(sys.argv[1])
p.load(p_load)

q = UnigramDictionary()
q_load = os.path.expanduser(sys.argv[2])
q.load(q_load)

p_tokens = [t for t in p.get_token_list()]
q_tokens = [t for t in q.get_token_list()]

for t in p_tokens:
    p_tfreq = p.get_frequency(p.get_id(t))
    q_tfreq = q.get_frequency(q.get_id(t))
    
    if p_tfreq != q_tfreq:
        print("mismatch for token ", t, " : p is ", p_tfreq, " while q is ", q_tfreq)
        
    