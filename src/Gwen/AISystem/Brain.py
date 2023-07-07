# https://huggingface.co/



class Brain:
    ''' 
        This class is designed to do advanced operations such as automated data collection, intelectual thought, and research.
    '''
    def __init__(self):
        pass
    
    def CollectData(self, topic, links=[]):
        responses = []
        
        file_obj = open('summarization_{}.txt'.format(topic.replace(" ", '_')),'w')

        print('Collecting Summaries...')
        for link in links:
            #Collect Response from Server
            print('*'*5+'Analyzing'+'*'*5)
            response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"{link}\nfrom this link, list all relevant data and information about {topic} and include specific information.",
            temperature=1,
            max_tokens=1800,
            top_p=1,
            frequency_penalty=0.12,
            presence_penalty=0.11
            )
            # Keep track of the response for later,
            responses.append(response['choices'][0]['text'])
            file_obj.write(responses[-1]+ '\n'+ '-'*10 + '\n')

            # Don't Overload the server with requests
            t.sleep(20)

        file_obj.close()
        print('Done Collecting summarizations...')
        
        
    def WriteEssay(self, prompt, research_topic, links=[]):
        # Example Links to Use
        responses = self.CollectData(research_topic, links)
        
        responses_string = ''
        for i in responses:
            responses_string += i
        responses_string = ''.join(responses_string.replace('\n','').replace('  ', '').split(' ')[::2900])


        file_obj = open('writing_prompt_{}.txt'.format(research_topic.replace(" ", '_').replace('-', '')),'w')
        print('Writing Prompt...')

        response = openai.Completion.create(
                model="text-davinci-003",
                prompt=f"{responses_string}\nWith that stated, write a thesis about {prompt}.",
                temperature=0.9,
                max_tokens=4000,
                top_p=1,
                frequency_penalty=0.06,
                presence_penalty=0.11
                )  

        thesis = response['choices'][0]['text']+'\n'

        file_obj.write(thesis)
        responses_string+=''.join(thesis.replace('\n','').replace('  ', '').split(' ')[::2900])
        t.sleep(10)
        response = openai.Completion.create(
                model="text-davinci-003",
                prompt=f'{responses_string}\nGiven this thesis, write in detail about {prompt}, and provide specific examples and data.',
                temperature=0.9,
                max_tokens=3800,
                top_p=1,
                frequency_penalty=0.06,
                presence_penalty=0
                )  

        file_obj.write(response['choices'][0]['text']+'\n')

        file_obj.close()
