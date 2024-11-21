from LLM2 import llm_models,llm_config
# ----------------------------------------------------------------------------------------------------------------------
class Agent_Wavefront(object):
    def __init__(self,folder_in,folder_out):
        self.folder_in = folder_in
        self.folder_out = folder_out

        self.LLM = llm_models.get_model(llm_config.get_config_openAI().filename_config_chat_model, model_type='QA')
        self.history = []
        self.example_cube = ''
        self.example_plane = ''
        self.example_pyramid = ''
        self.example_tetrahedron = ''

        self.example_mat_cube = ''
        self.example_mat_plane = ''
        self.example_mat_pyramid = ''
        self.example_mat_tetrahedron = ''

        self.load_FSLs()

        return
# ----------------------------------------------------------------------------------------------------------------------
    def load_FSLs(self):
        self.example_cube = open(self.folder_in + 'cube.obj', 'r').read()
        self.example_mat_cube = open(self.folder_in + 'cube.mtl', 'r').read()

        self.example_plane = open(self.folder_in + 'plane.obj', 'r').read()
        self.example_mat_plane = open(self.folder_in + 'plane.mtl', 'r').read()

        self.example_pyramid = open(self.folder_in + 'pyramid.obj', 'r').read()
        self.example_mat_pyramid = open(self.folder_in + 'pyramid.mtl', 'r').read()

        self.example_tetrahedron = open(self.folder_in + 'tetrahedron.obj', 'r').read()
        self.example_mat_tetrahedron = open(self.folder_in + 'tetrahedron.mtl', 'r').read()

        return
# ----------------------------------------------------------------------------------------------------------------------

    def create_prompt(self, query):

        hst = '\n'.join(self.history[:5])
        prompt = (f'You are assistant to create OBJ Wavefront files.'
                  f'This is the history of previous questions and answers: {hst}.\n\n'

                  f'This is example of properly prepared OBJ file for Cube:\n\n{self.example_cube}\n\n'
                  f'This is material of Cube:\n\n{self.example_mat_cube}\n\n'

                  f'This is example of Plane:\n\n{self.example_plane}\n\n'
                  f'This is material of Plane:\n\n{self.example_mat_plane}\n\n'

                  f'This is example of Pyramid:\n\n{self.example_pyramid}\n\n'
                  f'This is material of Pyramid:\n\n{self.example_mat_pyramid}\n\n'
                  
                  f'This is example of Tetrahedron:\n\n{self.example_tetrahedron}\n\n'
                  f'This is material of Tetrahedron:\n\n{self.example_mat_tetrahedron}\n\n'

                  f'Tag your reply with START and END signatures.\n'
                  
                  f'Ensure content between START and END signatures is the OBJ file '
                  f'so that each vertex, normal, face etc is split by new line.\n'
                  
                  f'Do not construct materials as they are already defined in separate files.\n'
                  
                  f'Remember that altitude (elevation) is encoded using Y axis '
                  f'so for example point elevated on 10 above the ground is (0,10,0).\n'
                  
                  f'Axis X comes for left-right, and axis Z comes for back forward.\n'
                  
                  f'Ensure each object should index vertexes of the faces starting proper offset.\n'
                  
                  f'For example, cube has 8 vertexes, so faces of second cube (leading f signature in OBJ file) '
                  f'should have offset 9 to index vertexes.\n'
                  
                  f'Another example: if there are 2 cubes of 8 vertex each, '
                  f'then indexing the faces of the next object should start with offset 2*8=16.\n\n'
                  
                  f'Now process user query below: {query}')

        return prompt

    # ----------------------------------------------------------------------------------------------------------------------
    def run_query(self, query):
        prompt = self.create_prompt(query)
        result = self.LLM.invoke(prompt)
        if not isinstance(result, str):
            result = result.content

        self.history.append('Question: ' + query + '\n' + '. Result:' + result + '\n')

        with open(self.folder_out + 'prompt.txt', 'w') as f:
            f.write(prompt)

        with open(self.folder_out + 'X.obj', 'w') as f:
            #extract content between START and END signatures
            start = result.find('START')
            end = result.find('END')
            result = result[start + 5:end].strip()
            f.write(result)

        return result
# ----------------------------------------------------------------------------------------------------------------------
