import streamlit as st
import re 
import time
import base64
import warnings
warnings.simplefilter('ignore')
from io import BytesIO

import nltk
# nltk.download('punkt')

# Streamlit app code
st.set_page_config(
    page_title='AI Summarizer',
    page_icon='🌼',
    layout='wide',
    initial_sidebar_state='expanded',
  
)

hide_streamlit_style = """
<style>

    /* Hide the page expander */
    div[data-testid='stSidebarNav'] ul {max-height:none}

    thead tr th:first-child {display:none}
    tbody th {display:none}
    
    button[title="View fullscreen"]{
    visibility: hidden;}

    div.block-container{padding-top:2rem;}

    div[class^='css-1544g2n'] { padding-top: 0rem; }
    [data-testid=column]:nth-of-type(1) [data-testid=stVerticalBlock]{
        gap: 0.605rem;
    }
    .stDeployButton, footer, #stDecoration {
        visibility: hidden;
    }

</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.sidebar.title("Select an option")

# from PIL import Image
# image = Image.open('logo.jpeg')
# st.sidebar.image(image, width=None , use_column_width=True, clamp=True, channels="RGB", output_format="auto") #caption= 'Creating a Sangria Experience'


# Session State also supports the attribute based syntax
if 'summary' not in st.session_state:
    st.session_state.summary = None

# Ensure NLTK resources used by rake_nltk are available (stopwords + punkt tokenizers)
for _res in [
    ("corpora/stopwords", "stopwords"),
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab/english", "punkt_tab"),
]:
    resource_path, download_name = _res
    try:
        nltk.data.find(resource_path)
    except LookupError:
        try:
            # Attempt to download the resource; some names (like punkt_tab) may map to 'punkt'
            nltk.download(download_name)
        except Exception:
            # If automatic download fails (no network or blocked), continue — the app will show an error later
            pass

@st.cache_resource(show_spinner=False)
def extract_doc_text(pdf_path):
    document_text = ""
    # Import Tika here to avoid failing app startup if tika/Java are missing
    try:
        from tika import parser
    except Exception:
        st.error("Apache Tika not available. Install 'tika' and ensure Java is on PATH to parse PDFs.")
        return ""

    parsed = parser.from_file(pdf_path)
    document_text = parsed.get("content", "")
    document_text = re.sub('([ \t]+)|([\n]+)', lambda m: ' ' if m.group(1) else '\n', document_text)
    return document_text

# @st.cache_resource(show_spinner=False)
def prep_b4_save(text):
    text = re.sub('Gods', 'God\'s', text)
    text = re.sub('yours', 'your\'s', text)
    text = re.sub('dont', 'don\'t', text)
    text = re.sub('doesnt', 'doesn\'t', text)
    text = re.sub('isnt', 'isn\'t', text)
    text = re.sub('havent', 'haven\'t', text)
    text = re.sub('hasnt', 'hasn\'t', text)
    text = re.sub('wouldnt', 'wouldn\'t', text)
    text = re.sub('theyre', 'they\'re', text)
    text = re.sub('youve', 'you\'ve', text)
    text = re.sub('arent', 'aren\'t', text)
    text = re.sub('youre', 'you\'re', text)
    text = re.sub('cant', 'can\'t', text)
    text = re.sub('whore', 'who\'re', text)
    text = re.sub('whos', 'who\'s', text)
    text = re.sub('whatre', 'what\'re', text)
    text = re.sub('whats', 'what\'s', text)
    text = re.sub('hadnt', 'hadn\'t', text)
    text = re.sub('didnt', 'didn\'t', text)
    text = re.sub('couldnt', 'couldn\'t', text)
    text = re.sub('theyll', 'they\'ll', text)
    text = re.sub('youd', 'you\'d', text)
    return text
# @st.cache_resource(show_spinner=False)
def text_chunking(new_text, size_of_chunk): #, size_of_chunk
    max_chunk = size_of_chunk#250
    # Import tokenizer when needed to avoid import-time errors
    try:
        from transformers import BertTokenizer
    except Exception:
        st.error("transformers not installed; text chunking requires 'transformers' package.")
        return []
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    words = new_text.split(' ')
    current_chunk = 0
    chunks = []
    for word in words:
        if len(chunks) == current_chunk + 1:
            # Check the number of tokens instead of the number of words
            if len(tokenizer.encode(' '.join(chunks[current_chunk]), truncation=False)) + len(tokenizer.encode(word, truncation=False)) <= max_chunk:
                chunks[current_chunk].append(word)
            else:
                current_chunk += 1
                chunks.append([word])
        else:
            chunks.append([word])

    for chunk_id in range(len(chunks)):
        chunks[chunk_id] = ' '.join(chunks[chunk_id])
    return chunks
@st.cache_resource(show_spinner=False)
def transformers_summary(chunks, max_length, min_length, ): #ngram, temp, topk, beams
    # global summary_length
    global summarizer
    # Defer heavy imports to avoid startup crashes when packages are missing or incompatible
    try:
        from transformers import pipeline
    except Exception:
        st.error("transformers not installed. Install 'transformers' and compatible 'torch' to enable summarization.")
        return ""
    try:
        from stqdm import stqdm
    except Exception:
        # fallback: simple iterator
        def stqdm(x, **kw):
            return x

    summarizer = pipeline("", model="philschmid/bart-large-cnn-samsum")

    bulletedSummaryString = "Here are the key insights:\n"

    st.toast("🚀 **Summarizing the text. Please wait...**")
    with st.spinner("Fetching key insights.."):
        # Display the summary
        for chunk in stqdm((chunks), desc= "Progress"):
            summary_placeholder = st.empty()
            try:
                chunk_summary = summarizer(chunk,  min_length = min_length, do_sample=False, no_repeat_ngram_size=4, encoder_no_repeat_ngram_size=3, num_beams=8, repetition_penalty=3.5)
                chunk_sum = chunk_summary[0]['summary_text']

                bulletedSummaryString += '\n⭕ ' + chunk_sum
                chunk_sum = "• " + chunk_sum
                for i in range(len(chunk_sum)+1):
                    summary_placeholder.markdown(chunk_sum[:i])
                    time.sleep(0.003)

            except Exception as e:
                print("Skipped chunk. Error:", e)
        st.success("**Done Summarizing.**")
        return bulletedSummaryString
@st.cache_resource(show_spinner=False)
def find_summary_transformers(pdf_path, size_of_chunk, max_length, min_length): #, size_of_chunk, ngram, temp, topk, beams
    # Extract text using Tika
    with st.spinner('Parsing the document..'):
        document_text_part1 = extract_doc_text(pdf_path)
    
    subtab_tab_1, subtab_tab_2, subtab_tab_3 = st.tabs(['Summary','keywords','Document'])

    with subtab_tab_2:
        st.subheader('Top 10 Keywords')
        from rake_nltk import Rake
        r = Rake()
        r.extract_keywords_from_text(document_text_part1)
        keyword_list= r.get_ranked_phrases()[:10]
        for keys in keyword_list:
            st.write("⭕", keys)

    with subtab_tab_3:
        st.subheader('Document')
        with st.expander('**View parsed data**', expanded=True):
            st.text(document_text_part1)
    
    with subtab_tab_1:
        st.subheader('Summary')
        global chunks
        with st.spinner("Creating chunks of data.."):
            chunks = text_chunking(document_text_part1, size_of_chunk) #, size_of_chunk
        if len(chunks) != 1:
            if len(chunks) <= 1000:
                all_transformers_summaries = transformers_summary(chunks, max_length, min_length) #, ngram, temp, topk, beams
                st.session_state.summary = all_transformers_summaries # added new logic
                summary_by_transformers = prep_b4_save(all_transformers_summaries)
                return summary_by_transformers
            else:
                st.write("Please upload a pdf with less than 500 pages!" )
        else:
            st.write("Not able to parse. Try another document!")
    


@st.cache_resource(show_spinner=False)
def get_pdf(input_string):
    try:
        if not input_string:
            return {"Exception is here: ": 'No summary text to convert to PDF (input is empty or None).'}
        
        # Create in-memory buffer
        pdf_buffer = BytesIO()

        # Import reportlab here to avoid import-time errors when optional package is missing
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet
        except Exception:
            return {"Exception is here: ": "reportlab not installed; cannot create PDF. Install 'reportlab' to enable PDF export."}

        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)

        lines = input_string.split('\n')
        elements = []
        styles = getSampleStyleSheet()
        styles['BodyText'].fontSize = 10
        for line in lines:
            stripped_line = line.strip()
            if stripped_line:
                elements.append(Paragraph(stripped_line, styles['BodyText']))
                elements.append(Spacer(1, 10))

        def header_footer(canvas, doc):
            canvas.saveState()
            canvas.setFont('Times-Roman', 8)
            canvas.drawString(52,750,"AI Summarizer")
            canvas.drawString(453,50,"AI Summarizer")
            canvas.restoreState()

        doc.build(elements, onFirstPage=header_footer, onLaterPages=header_footer)

        # Get the PDF data from the buffer and return it
        pdf_bytes = pdf_buffer.getvalue()
        return pdf_bytes 
    except Exception as e:
        return {"Exception is here: ": f'{e}'}


def make_pdf_download_link(pdf_bytes, download_name):
    """Return an HTML download link for pdf_bytes or show an error if pdf_bytes is a dict."""
    if isinstance(pdf_bytes, dict):
        # Show the error message collected from get_pdf
        try:
            # Format the dict into a readable message
            err_msg = '; '.join([f"{k}: {v}" for k, v in pdf_bytes.items()])
        except Exception:
            err_msg = str(pdf_bytes)
        st.error(f"Failed to create PDF: {err_msg}")
        return None
    try:
        b64 = base64.b64encode(pdf_bytes).decode()  # some strings
        href = f'<a href="data:file/pdf;base64,{b64}" download="{download_name}">Download PDF File</a>'
        return href
    except Exception as e:
        st.error(f"Error encoding PDF for download: {e}")
        return None


title = "AI Powered Python :grey[Summarization] Tool 📖"

# Create a title placeholder
title_placeholder =st.empty()
#**Select an Option**
choice = st.sidebar.radio(
    "🔻",
    ('File upload 📁', 'Website 🌐', 'Text Box 🖋️', 'Instructions ⚗️','Features 🦄', 'About 👩‍🦰')
)

# Typewriting  
for i in range(len(title)+1):
    title_placeholder.title(title[:i])
    time.sleep(0.003)

if choice == 'File upload 📁':
        uploaded_file = st.file_uploader(" ", type=["pdf", "docx", "txt"],  help="PDF, TXT and DOC file supported")
        st.write(" ")

        cola, colb = st.columns(2)
        with cola:
            min_length, max_length = st.slider( 'Select a range of Summarization', 15, 50, (33, 39))
        with colb:
            size_of_chunk = st.slider( 'Adjust Chunk Size', 250, 1000, value=250, help = 'Keep default value for best results, e.g. -250')
        if st.session_state.summary is not None:
            with st.expander("Your past summary:"):
                st.text(st.session_state.summary)
        
        if st.button('Start Summarization'):
            if uploaded_file is not None:
                # file_details = {"Filename":uploaded_file.name, "FileSize":uploaded_file.size}
                if uploaded_file.name.endswith(('.pdf', '.docx', '.txt')):
                    # st.write(file_details)
                
                    # Display the summary
                    # st.subheader('Summary')
                    summary = find_summary_transformers(uploaded_file, size_of_chunk, max_length, min_length ) #, size_of_chunk, ngram, temp, topk, beams
                    st.session_state.summary = summary
                    st.subheader('Result')
                    if not summary:
                        st.error('Summary generation returned empty result; cannot create PDF.')
                    else:
                        pdf_bytes = get_pdf(summary)
                        href = make_pdf_download_link(pdf_bytes, f"{uploaded_file.name}_summary.pdf")
                        if href:
                            st.markdown(href, unsafe_allow_html=True)
                else:
                    st.write("**Supported documents are PDF, Docx and txt!**")

elif choice == 'About 👩‍🦰':
    st.info("AI Summarizer — a tool to quickly extract key insights from documents and web pages.")

elif choice == 'Text Box 🖋️':
    description = st.text_area('**Blog/ Article to summarize (min 150 words)**', placeholder= 'Once upon a time, in a small town nestled between rolling hills and lush greenery...')
    st.write("Adjust the parameters")
    
    cola, colb = st.columns(2)
    with cola:
        min_length, max_length = st.slider( 'Select a range of Summarization', 15, 50, (33, 39))
    with colb:
        size_of_chunk = st.slider( 'Adjust Chunk Size', 250, 1000, value=250,  help = 'Keep default value for best results, e.g. -250')

    if st.session_state.summary is not None:
            with st.expander("Your past summary:"):
                st.text(st.session_state.summary)      
    
    if st.button('Start Summarization') and description and min_length and max_length: # and ngram:
        if len(description.split()) >= 150:
            if len(description.split()) < 50000:
                with st.spinner('Generating your summary..'):
                    

                    summary = transformers_summary(text_chunking(description, size_of_chunk), max_length, min_length) #, size_of_chunk , ngram, temp, topk, beams
                    st.session_state.summary = summary # added
                    st.subheader('Result')
                    pdf_bytes = get_pdf(summary)
                    href = make_pdf_download_link(pdf_bytes, "summary.pdf")
                    with st.expander("Download the pdf summary"):
                        if href:
                            st.markdown(href, unsafe_allow_html=True)
            else:
                st.write("Your text exceeds the 50,000 words limit. Please shorten your text.")
        else:
            st.write("**Too short to Summarize!**")

elif choice == 'Website 🌐':
    from scraper.getSerpResults import scrape_content
    url = st.text_input("**Paste a URL**", placeholder=" Paste a website's url..")
    st.write("Adjust the parameters")

    cola, colb = st.columns(2)
    with cola:
        min_length, max_length = st.slider( 'Select a range of Summarization', 15, 50, (33, 39))
    with colb:
        size_of_chunk = st.slider( 'Adjust Chunk Size', 250, 1000, value=250, help = 'Keep default value for best results, e.g. -250')

    if st.session_state.summary is not None:
            with st.expander("Your past summary:"):
                st.text(st.session_state.summary)

    title, text, summary, keywords = scrape_content(url)
    if st.button('Start Summarization') and text:
        if len(text.split()) >= 50:
            if len(text.split()) < 50000:
                
                st.divider()
                # Display the title
                st.markdown(f"## {title}")
                
                # Display the summary
                st.subheader("Summary")
                with st.spinner():
                    summary = transformers_summary(text_chunking(text, size_of_chunk), max_length, min_length) #, size_of_chunk, ngram, temp, topk, beams
                st.session_state.summary = summary
                st.subheader('Result')
                with st.expander("Download the pdf summary"):
                    if not summary:
                        st.error('Summary generation returned empty result; cannot create PDF.')
                    else:
                        pdf_bytes = get_pdf(summary)
                        href = make_pdf_download_link(pdf_bytes, "Summary.pdf")
                        if href:
                            st.markdown(href, unsafe_allow_html=True)
                st.divider() 

                with st.expander('**View page keywords**'):
                    # Display the keywords
                    st.markdown(f"**Keywords:**\n{', '.join(keywords)}")
                
                with st.expander('**View page text**'):
                    # Display the text
                    st.markdown("**Text:**")
                    st.text(text)
            else:
                st.write("Your text exceeds the 50,000 words limit. Please shorten your text.")
        else:
            st.write("**Too short to Summarize!**")

# st.divider()   
elif choice == 'Instructions ⚗️':
    st.info("To enhance the performance of a summarization model, you can fine-tune several parameters:")

    st.markdown("""
    - **Min Length**: Defines the minimum length of the generated text. It ensures that the output is not too short and contains enough information.
    - **Max Length**: Sets an upper limit to the length of the generated text. It prevents the output from being excessively long.
    - **Chunk Size**: The portion of text in words to summarize at once. It affects the depth of summary getting generated.
    """)

elif choice == 'Features 🦄':
        st.markdown('''### Here's what makes my tool stand out:\n
        - 🚀 **Fast**: Get your summaries in seconds.\n
        - 🧠 **Smart**: Uses AI to understand context and extract key points.\n
        - 📚 **Versatile**: Perfect for academic papers, reports, books, and more.\n
        - 🔒 **Secure**: Your documents are safe with us. We respect your privacy.''')
