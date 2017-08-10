'''
The TokenMap provides a mapping from tokens (plain words) to integer ids,
as well as a map from ids back to tokens.
'''

import gzip
from collections import OrderedDict


SILENT = 0
WARN = 1
ERROR = 2
UNK = 0
basedict={'A':'T', 'T':'A', 'C':'G', 'G':'C', 'N':'N', 'K':'K', 'U':'U'}  
# ^^^ 'U', 'K' present so that 'UNK' maps to 'KUN'

def get_rc(s):
    '''
    Return the reverse complement of DNA-token s
    '''
    return ''.join([basedict[b] for b in s[::-1]])

def ensure_str(s):
    '''
    Ensures that the string is encoded as a unicode str, not bytes
    '''
    try:
        return s.decode('utf8')
    except AttributeError:
        return s


class TokenMap(object):

    def __init__(self, on_unk=WARN, tokens=None):
        '''
        Create a new TokenMap.  Most common usage is to call this without
        any arguments.

        on_unk: Controls the behavior when asked to provide the token_id
            for a token not found in the map.  Default is WARN, which 
            means returning 0 (which is id reserved for unknown tokens) 
            and then printing a warning to stout.  Choose from SILENT, 
            WARN, or ERROR.

        tokens: List of strings corresponding to a map that should be 
            used.  The index of a token in the list is used as its ID.
            Not normally used, because TokenMap provides functions to 
            build the map easily from a corpus.  The first element in the 
            list should be 'UNK', becuase id 0 is reserved for unknown 
            tokens.  Not doing so is an error.
        '''

        # Validate on_unk
        if on_unk not in (SILENT, WARN, ERROR):
            raise ValueError(
                'on_unk must be one of token_map.SILENT, token_map.WARN, '
                'or token_map.ERROR.'
            )
        self.on_unk = on_unk

        # Initialize the token mapping
        if tokens is None:
            self.map = {'UNK':UNK}  # keys are tokens, values are ids
            self.tokens = ['UNK']   # entries are tokens, indices are ids

        # If an initial lexicon was provided, build the map from it
        else:
            if tokens[0] != 'UNK':
                raise ValueError(
                    'tokens[0] must be "UNK" because ID 0 is reserved for '
                    'unknown tokens.'
                )

            self.tokens = [ensure_str(t) for t in tokens]
            self.map = dict((t, idx) for idx, t in enumerate(self.tokens))


    def compact(self):
        '''
        Recreate the tokens list and mapping such that `None`s are 
        removed (which are holes left by calls to `remove()`.
        '''
        self.tokens = [t for t in self.tokens if t is not None]
        self.map = dict((t, idx) for idx, t in enumerate(self.tokens))


    def remove(self, token):
        token = ensure_str(token)
        idx = self.get_id(token)
        if idx == UNK:
            raise ValueError(
                'Cannot remove token %s because it does not ' 
                'exist or is reserved.' % str(token)
            )
        self.tokens[idx] = None


    def add(self, token):
        token = ensure_str(token)
        try:
            return self.map[token]
        except KeyError:
            next_id = len(self.tokens)
            self.map[token] = next_id
            self.tokens.append(token)
            return next_id


    def get_vocab_size(self):
        return len(self.tokens)


    def update(self, token_iterable):
        return [self.add(token) for token in token_iterable]


    def get_id(self, token):
        token = ensure_str(token)
        try:
            return self.map[token]

        # If that token isn't in the vocabulary, what should we do?
        # This depends on the setting for on_unk.  We can return the UNK
        # token silently, return the UNK token with a warning, or 
        # raise an error.
        except KeyError:
            if self.on_unk == SILENT:
                return self.map['UNK']  # i.e. return 0
            elif self.on_unk == WARN:
                print('Warning, unrecognized token: %s' % token)
                return self.map['UNK']  # i.e. return 0
            elif self.on_unk == ERROR:
                raise
            else:
                raise ValueError(
                    'Unrecognized value for on_unk in TokenMap.'
                )


    def get_ids(self, token_iterable):
        return [self.get_id(token) for token in token_iterable]


    def get_token(self, idx):
        return self.tokens[idx]

    def get_tokens(self, idx_iterable):
        return [self.tokens[idx] for idx in idx_iterable]


    def __len__(self):
        return len(self.tokens)


    def save(self, filename):
        if filename.endswith('.gz'):
            f = gzip.open(filename, 'w')
        else:
            f = open(filename, 'w')

        for idx, token in enumerate(self.tokens):
            try:
                f.write(token + '\n')
            except TypeError:
                token = token + '\n'
                f.write(token.encode())


    def load(self, filename):
        self.map = {}
        self.tokens = []

        if filename.endswith('.gz'):
            f = gzip.open(filename)
        else:
            f = open(filename)

        for idx, line in enumerate(f):
            try:
                token = line.strip().decode('utf-8')
            except AttributeError:
                line = line.decode('utf-8')
                token = line.strip()
            self.map[token] = idx
            self.tokens.append(token)


class SeqTokenMap(TokenMap):
    
    def __init__(self, on_unk=WARN, tokens=None):
        '''
        Create a new TokenMap that which contains short DNA sequences.
        This limits the vocabulary, and means we assign the same ID to a 
        token and its reverse complement.  
        
        on_unk: Controls the behavior when asked to provide the token_id
            for a token not found in the map.  Default is WARN, which 
            means returning 0 (which is id reserved for unknown tokens) 
            and then printing a warning to stout.  Choose from SILENT, 
            WARN, or ERROR.
    
        tokens: List of strings corresponding to a map that should be 
            used.  The index of a token in the list is used as its ID.
            Not normally used, because TokenMap provides functions to 
            build the map easily from a corpus.  The first element in the 
            list should be 'UNK', becuase id 0 is reserved for unknown 
            tokens.  Not doing so is an error.
        '''
    
        # Validate on_unk
        if on_unk not in (SILENT, WARN, ERROR):
            raise ValueError(
                'on_unk must be one of token_map.SILENT, token_map.WARN, '
                'or token_map.ERROR.'
            )
        self.on_unk = on_unk
    
        # Initialize the token mapping
        if tokens is None:
            self.token_map = OrderedDict()
            self.token_map['UNK'] = UNK   # keys are tokens, values are ids
            self.tokens = ['UNK']  # what tokens and their RCs have we seen
            self.maxid = UNK    # the present highest valued ID
    
        # If an initial lexicon was provided, build the map from it
        else:
            if tokens[0] != 'UNK':
                raise ValueError(
                    'tokens[0] must be "UNK" because ID 0 is reserved for '
                    'unknown tokens.'
                )
            
            self.tokens = [ensure_str(t) for t in tokens]
            self.token_map = OrderedDict()
            idx = 0
            for t in self.tokens:
                rc = get_rc(t)
                if rc in self.token_map:
                    self.token_map[t] = self.token_map[rc]
                else:
                    self.token_map[t] = idx
                    idx = idx + 1
            self.maxid = idx
                
                
    def compact(self):
        '''
        Recreate the tokens list and mapping such that `None`s are 
        removed (which are holes left by calls to `remove()`.
        '''
        self.tokens = [t for t in self.token_map if self.token_map[t] is not None]
        self.token_map = OrderedDict()
        idx = 0
        for t in self.tokens:
            rc = get_rc(t)
            if rc in self.token_map:
                self.token_map[t] = self.token_map[rc]
            else:
                self.token_map[t] = idx
                idx = idx + 1        
        self.maxid = idx

    def remove(self, token):
        '''
        Remove this token from the map.
        '''
        token = ensure_str(token)
        idx = self.get_id(token)
        if idx == UNK:
            raise ValueError(
                'Cannot remove token %s because it does not ' 
                'exist or is reserved.' % str(token)
            )
        self.token_map[token] = None
        # edge case: idx(token) is maxid && token's rc not in the map
        if get_rc(token) not in self.token_map and idx == self.maxid:
            self.maxid = self.maxid - 1


    def add(self, token):
        '''
        Add the token to the map
        '''
        token = ensure_str(token)
        try:
            return self.token_map[token]
        except KeyError:
            rc = get_rc(token)
            token_id = self.token_map[rc] if rc in self.token_map else self.maxid + 1
            self.token_map[token] = token_id
            self.maxid = token_id if token_id > self.maxid else self.maxid
            return token_id


    def get_vocab_size(self):
        return len(self.token_map)


    def update(self, token_iterable):
        return [self.add(token) for token in token_iterable]


    def get_id(self, token):
        token = ensure_str(token)
        try:
            return self.token_map[token]

        # If that token isn't in the vocabulary, what should we do?
        # This depends on the setting for on_unk.  We can return the UNK
        # token silently, return the UNK token with a warning, or 
        # raise an error.
        except KeyError:
            if self.on_unk == SILENT:
                return self.token_map['UNK']  # i.e. return 0
            elif self.on_unk == WARN:
                print('Warning, unrecognized token: %s' % token)
                return self.token_map['UNK']  # i.e. return 0
            elif self.on_unk == ERROR:
                raise
            else:
                raise ValueError(
                    'Unrecognized value for on_unk in TokenMap.'
                )


    def get_ids(self, token_iterable):
        return [self.get_id(token) for token in token_iterable]


    def get_token(self, idx, rc=True):
        ''' Each token shares an idx with its RC, return both by default '''
        if rc:
            token = self.tokens[idx]
            return token, get_rc(token)            
        else:
            return self.tokens[idx]

    def get_tokens(self, idx_iterable, rc=True):
        return [self.get_token(idx, rc) for idx in idx_iterable]


    def __len__(self):
        return len(self.token_map)


    def save(self, filename):
        if filename.endswith('.gz'):
            f = gzip.open(filename, 'w')
        else:
            f = open(filename, 'w')

        for token, idx in self.token_map.items():
            try:
                f.write(token + " " + str(idx) + '\n')
            except TypeError:
                token = token + " " + str(idx) + '\n'
                f.write(token.encode())


    def load(self, filename):
        self.token_map = OrderedDict()
        self.tokens = []

        if filename.endswith('.gz'):
            f = gzip.open(filename)
        else:
            f = open(filename)
            
        idx = 0
        for line in f:
            try:
                line = line.strip().decode('utf-8')
                token, idx = line.split()
            except AttributeError:
                line = line.decode('utf-8')
                token, idx = line.strip().split()
            self.token_map[token] = idx
            self.tokens.append(token)        