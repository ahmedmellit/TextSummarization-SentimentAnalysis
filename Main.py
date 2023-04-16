import streamlit as st 
from textblob import TextBlob
import pandas as pd
from gensim.summarization import summarize
import transformers import pipeline
sentiment_pipeline = pipeline("sentiment-analysis")


def main():
  """NLP Web App"""

  st.title("NLP Text Summarizer and Sentiment Analyzer")

  activities = ["Text Summarization","Sentiment Analysis"]
  choice = st.sidebar.selectbox("Select a Service",activities)

  if choice == 'Text Summarization':
    st.subheader("Summarize Document")
    raw_text = st.text_area("Enter Text Here")
    summarizer_type = st.selectbox("Summarizer Type",["Gensim","BART"])
    if st.button("Summarize"):
      if summarizer_type == "Gensim":
        summary_result = summarize(raw_text)
      st.write(summary_result)

  if choice == 'Sentiment Analysis':
    st.header('Sentiment Analysis')
    with st.expander('Analyze Text'):
        text = st.text_input('Text here: ')
        if text:
	  st.write('Sentiment: ', sentiment_pipeline(text)['label'])
          st.write('Score: ', sentiment_pipeline(text)['score'])

    with st.expander('Analyze CSV'):
        upl = st.file_uploader('Upload file')

        def score(x):
            sent = TextBlob(x)
            return sent.sentiment.polarity

        def analyze(x):
            if x >= 0.5:
                return 'Positive'
            elif x <= -0.5:
                return 'Negative'
            else:
                return 'Neutral'

        if upl:
            df = pd.read_excel(upl)
            del df['Liked']
            df['score'] = df['Review'].apply(score)
            df['analysis'] = df['score'].apply(analyze)
            st.write(df.head(10))

            @st.cache
            def convert_df(df):
                return df.to_csv().encode('utf-8')
            csv = convert_df(df)

            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='sentiment.csv',
                mime='text/csv',
            )


if __name__ == '__main__':
	main()
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
