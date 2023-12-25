from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
import speech_recognition as sr
import pyttsx3

engine=pyttsx3.init()
r=sr.Recognizer()
with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        print("Ask your query")

        audio=r.listen(source)
        Query = r.recognize_google(audio)
        print("You said: ",Query)



def getLLMResponse(Query):
    llm = CTransformers(model="C:/Users/Kalpe/Downloads/mistral-7b-instruct-v0.1.Q2_K.gguf",
                        model_type='llama',
                        config={'max_new_tokens': 256,
                                'temperature': 0.01})

    template = """
    {query}
    """

    prompt = PromptTemplate(
        input_variables=["query"],
        template=template,
    )

    response = llm(prompt.format(query=Query))
    return response

def main():
    print("LLM Chatbot 🤖")
    
    response = getLLMResponse(Query)
    engine.say(response)
    engine.runAndWait()
    
    print(response)

if __name__ == "_main_":
    main()