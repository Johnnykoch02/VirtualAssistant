from enum import Enum


class langIncrementer:
    class ptrs(Enum):
        VERB = 0
        NOUN = 1
        CUNJ = 2
    def __init__(self):
        self.ptr = self.ptrs.VERB
    def curr(self):
        return self.ptr
    def inc(self):
        if self.ptr == self.ptrs.VERB:
            self.ptr = self.ptrs.NOUN
        elif self.ptr == self.ptrs.NOUN:
            self.ptr = self.ptrs.CUNJ
        else:
            self.ptr = self.ptrs.VERB





class CommandController(object):
    def __init__(self):
        
        self.commands = {
            'spotify': [
                'play', 'shuffle', 'search','playlist','volume'
            ],
            'led': [
                'mode', 'party', 
            ]
        }

    def process(self):

        # Import required libraries
        import nltk
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        from nltk import pos_tag, word_tokenize, RegexpParser
        
        # Example text
        sample_text = "Open up chrome and searh for Howels and Hood Breakfast"
        
        # Find all parts of speech in above sentence
        tagged = pos_tag(word_tokenize(sample_text))
        commands = []
        langInc = langIncrementer()
        currCmd = []
        currData = ''
        tempData = ''
        for token in tagged:
            if 'VB' in token[1]:
                if langInc.curr() == langInc.ptrs.VERB:
                    '''HIT'''
                    langInc.inc()
                    currCmd.append(token[0])
                
            elif 'NN' in token[1]:
                if langInc.curr() == langInc.ptrs.NOUN:
                    currData+= token[0]
                    
            elif 'CC' in token[1]:
                if langInc.curr() == langInc.ptrs.VERB:
                    currData+=token[0]
                    langInc.ptr= langInc.ptrs.NOUN
                else:
                    currCmd.append(currData)
                    commands.append(currCmd.copy())
                    currData = ''
                    currCmd = []
                    langInc.ptr= langInc.ptrs.VERB
        if len(currData) > 0 or len(currCmd) > 0:
            currCmd.append(currData)
            commands.append(currCmd)

        print(commands)
        #Extract all parts of speech from any text
        
        
CommandController().process()