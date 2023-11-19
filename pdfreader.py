import streamlit as st
import openai
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import re

openai.api_key = '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'



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 500,
        chunk_overlap = 100,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def generate_keywords(text):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=text,
            max_tokens=50  # Adjust max_tokens as needed for your desired length
        )
        return response.choices[0].text.strip()
    except Exception as e:
        st.error(f"Error occurred: {e}")
        return None




def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = OpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorstore.as_retriever(),
        memory = memory
    )
    return conversation_chain




def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history'][1:]

    #summary display so that it doesn't disappear when a new prompt is entered
    with st.sidebar:
        for i in range(2):
            st.text("")

        st.session_state.summary_messages = response['chat_history'][1:2]
        for message in st.session_state.summary_message:
            st.markdown(
                f'<div style="background-color: #cce0ff; border-radius: 5px; padding: 10px; margin-bottom: 10px; color: #333;"><strong>------------------ SUMMARY ------------------</strong> {message}</div>',
                unsafe_allow_html=True
            )

        for i in range(2):
            st.text("")

        st.session_state.question_bullets = response['chat_history'][3:4]
        for question in st.session_state.question_bullets:
            question = str(question).replace('content=\'', '').strip()
            st.markdown(
                f'<div style="background-color: #cce0ff; border-radius: 5px; padding: 10px; margin-bottom: 10px; color: #333;"><strong>--------------- QUESTIONS ---------------</strong> {question}</div>',
                unsafe_allow_html=True
            )

    # chat display
    for i, message in enumerate(st.session_state.chat_history[3:]):
        cleaned_message = str(message)[:-1].replace('content=\'', '').strip()
        # cleaned_message = str(message).strip().replace('content=\'', '')[:-1]
        if i % 2 == 0:
            st.markdown(
                f'<div style="background-color: #e6f2ff; border-radius: 5px; padding: 10px; margin-bottom: 10px; color: #333;">{cleaned_message}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div style="background-color: #cce0ff; border-radius: 5px; padding: 10px; margin-bottom: 10px; color: #333;"><strong>{cleaned_message}</strong> </div>',
                unsafe_allow_html=True
            )



def generate_summary(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    summary_messages = []
    for i, message in enumerate(st.session_state.chat_history):
        cleaned_message = str(message)[:-1].replace('content=\'', '').strip()
        if i % 2 != 0:
            summary_messages.append(cleaned_message)
    return summary_messages



def get_document_summary():
    if st.session_state.conversation:
        user_question = "Provide a summary of the uploaded file. If the file is a textbook or a book, disregard the table of contents and generate a brief overview of the primary contents of the book or chapters in the book."
        return generate_summary(user_question)
    else:
        st.warning("Please upload a document first.")



def get_questions(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    question_output = []
    for i, question in enumerate(st.session_state.chat_history):
        cleaned_message = str(question)[:-1].replace('content=\'', '').strip()
        if i % 2 != 0:
            if cleaned_message.endswith('?'):
                cleaned_message = cleaned_message.replace('\n', ' ')
                question_output.append(cleaned_message)
            else:
                cleaned_message = re.sub(r'[^a-zA-Z0-9\s]', '', cleaned_message)  # Remove special characters
                cleaned_message = re.sub(r'\s+', ' ', cleaned_message).strip()  # Remove extra spaces
                cleaned_message = cleaned_message.capitalize()  # Capitalize the sentence
                cleaned_message = cleaned_message.replace('\n', ' ')
                question_output.append(f"{cleaned_message}?")
    return question_output



def get_document_questions():
    if st.session_state.conversation:
        user_question = "Generate a list of detailed questions that quiz the user about the uploaded document. WRITE THE QUESTIONS IN QUESTION FORMAT (e.g What does this topic mean?) for the user to practice about the uploaded document. DO NOT INCLUDE THE CHARACTERS \n WITHIN THE QUESTIONS!"
        return get_questions(user_question)
    else:
        st.warning("Please upload a document first.")





def main():
    load_dotenv()
    st.set_page_config(
        page_title = "NoteFinder",
        page_icon = ":notebook:"
    )

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    col1, col2 = st.columns([1,2])
    with col1:
        st.title("NoteFinder")
    with col2:
        st.image("/Users/sidsomani/Desktop/streamlit_projects/multiFileReader/logovs2.png", width = 80)

    #just to add some space for less jam packed home page
    for i in range(4):
        st.text("")

    with st.sidebar:
        st.image("/Users/sidsomani/Desktop/streamlit_projects/multiFileReader/logovs2.png", width = 200)

        for i in range(3):
            st.text("")

        pdf_docs = st.file_uploader("Upload your files", accept_multiple_files=True)

        for i in range(2):
                    st.text("")

        if pdf_docs and st.button("Process"):
            st.text("")
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                # st.write(raw_text)

                # get text chunks
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)
                keywords = []
                for chunk in text_chunks:
                    keywords.append(generate_keywords(chunk))
                    # if keywords:
                    #     st.write("Chunk keywords: "+keywords)

                # create vector store with embedded chunks
                vectorstore = get_vectorstore(text_chunks)
                # st.write(vectorstore)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

                for i in range(3):
                    st.text("")

                # get summary of document
                summary = get_document_summary()
                if summary:
                    st.session_state.summary_message = summary
                    for message in st.session_state.summary_message:
                        st.markdown(
                            f'<div style="background-color: #cce0ff; border-radius: 5px; padding: 10px; margin-bottom: 10px; color: #333;"><strong>------------------ SUMMARY ------------------</strong> {message}</div>',
                            unsafe_allow_html=True
                        )

                for i in range(2):
                    st.text("")

                questions = get_document_questions()
                if questions:
                    st.session_state.question_bullet = questions
                    for question in st.session_state.question_bullet[1:]:
                        st.markdown(
                            f'<div style="background-color: #cce0ff; border-radius: 5px; padding: 10px; margin-bottom: 10px; color: #333;"><strong>--------------- QUESTIONS ---------------</strong> {question}</div>',
                            unsafe_allow_html=True
                        )


    user_question = st.text_input("Ask a question about your documents")
    if user_question:
        handle_userinput(user_question)



if __name__ == '__main__':
    main()


