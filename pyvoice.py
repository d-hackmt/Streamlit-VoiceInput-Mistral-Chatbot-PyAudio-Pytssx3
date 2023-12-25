from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
import speech_recognition as sr
import pyttsx3
import streamlit as st


st.set_page_config(page_title="LLM Chatbot",
                   page_icon='🤖',
                   layout='centered',
                   initial_sidebar_state='collapsed')
st.header("LLM Chatbot 🤖")

query = st.text_area('Enter your query', height=100)
record_button = st.button("Record")
submit_button = st.button("Submit")

def getLLMResponse(query):
    llm = CTransformers(model="C:/Users/jaikr\Desktop/VS_Code_Projects/Zephyr-Lang-Chainlit/mistral-7b-openorca.Q2_K.gguf",
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

    response = llm(prompt.format(query=query))
    return response


def main():
    text_query = ""
    if record_button:
        engine = pyttsx3.init()
        r = sr.Recognizer()
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            print("Ask your query")

            audio = r.listen(source)

            try:
                text_query = r.recognize_google(audio)
                print("You said: ", text_query)
                st.text_area('You said:', value=text_query, height=20)
            except sr.UnknownValueError:
                st.write("Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                st.write(f"Could not request results from Google Speech Recognition service; {e}")
    if submit_button:
        response = getLLMResponse(text_query)
        st.write("LLM Response:")
        st.write(response)
        engine.say(response)
        engine.runAndWait()
        print(response)

if __name__ == "__main__":
    main()