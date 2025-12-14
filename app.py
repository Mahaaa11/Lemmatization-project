import streamlit as st
import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import os

st.set_page_config(page_title="Arabic Lemmatization System", page_icon="🔤", layout="wide")

st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stTextArea textarea { font-size: 16px; font-family: 'Traditional Arabic', Arial, sans-serif; }
    .result-box { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; margin: 10px 0; }
    h1 { color: #1f2937; font-weight: 600; }
    h2, h3 { color: #374151; }
    </style>
""", unsafe_allow_html=True)

st.title("Arabic Lemmatization System")
st.markdown("---")

with st.expander("About This System", expanded=False):
    st.write("""
    This system performs Arabic lemmatization using a fine-tuned transformer model. 
    Lemmatization reduces words to their root forms, which is essential for Arabic NLP tasks.
    """)

with st.sidebar:
    st.header("Model Configuration")
    current_dir = os.getcwd()
    files_in_dir = os.listdir(current_dir)
    
    config_exists = any('config' in f.lower() and f.endswith('.json') for f in files_in_dir)
    tokenizer_exists = any('tokenizer' in f.lower() and f.endswith('.json') for f in files_in_dir)
    
    if config_exists and tokenizer_exists:
        st.success("Model files detected!")
    else:
        st.warning("Model files not detected")
    
    model_path = st.text_input("Model Directory", value=current_dir)
    st.markdown("---")
    st.header("Processing Options")
    show_tokens = st.checkbox("Show Tokenization", value=True)

@st.cache_resource
def load_model_and_tokenizer(model_path):
    try:
        files = os.listdir(model_path)
        renamed_files = {}
        for old_name in files:
            if '(2)' in old_name or ' (2)' in old_name:
                new_name = old_name.replace(' (2)', '').replace('(2)', '')
                old_path = os.path.join(model_path, old_name)
                new_path = os.path.join(model_path, new_name)
                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    renamed_files[new_path] = old_path
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        return model, tokenizer, renamed_files
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, {}

tab1, tab2, tab3 = st.tabs(["Single Text", "Batch Processing", "Model Info"])

with tab1:
    st.header("Single Text Lemmatization")
    input_text = st.text_area("Enter Arabic Text", height=150, placeholder="أدخل النص العربي هنا...")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        process_btn = st.button("Process Text", type="primary", use_container_width=True)
    
    if process_btn and input_text:
        with st.spinner("Loading model and processing..."):
            try:
                model, tokenizer, renamed = load_model_and_tokenizer(model_path)
                
                if model is None or tokenizer is None:
                    st.error("Failed to load model. Please check the model path.")
                else:
                    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
                    with torch.no_grad():
                        outputs = model(**inputs)
                    
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
                    
                    st.success("Processing Complete")
                    st.markdown("### Results")
                    
                    if show_tokens:
                        st.markdown("#### Tokenization")
                        st.markdown(f'<div class="result-box">{" | ".join(tokens)}</div>', unsafe_allow_html=True)
                    
                    st.markdown("#### Predicted Lemmas")
                    predicted_labels = [model.config.id2label[pred.item()] for pred in predictions[0]]
                    
                    result_data = []
                    for token, lemma in zip(tokens, predicted_labels):
                        if token not in ['[CLS]', '[SEP]', '[PAD]']:
                            result_data.append({"Token": token, "Lemma": lemma})
                    
                    if result_data:
                        st.table(result_data)
                    
                    st.markdown("#### Statistics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Tokens", len([t for t in tokens if t not in ['[CLS]', '[SEP]', '[PAD]']]))
                    with col2:
                        st.metric("Input Length", len(input_text.split()))
                    with col3:
                        st.metric("Model Vocab Size", len(tokenizer))
            except Exception as e:
                st.error(f"Error: {str(e)}")

with tab2:
    st.header("Batch Processing")
    st.write("Upload a text file with one sentence per line.")
    uploaded_file = st.file_uploader("Choose a text file", type=['txt'])
    
    if uploaded_file is not None:
        content = uploaded_file.read().decode('utf-8')
        lines = content.strip().split('\n')
        st.info(f"Loaded {len(lines)} lines")
        
        if st.button("Process Batch", type="primary"):
            with st.spinner("Processing..."):
                model, tokenizer, renamed = load_model_and_tokenizer(model_path)
                if model and tokenizer:
                    results = []
                    for line in lines:
                        if line.strip():
                            inputs = tokenizer(line, return_tensors="pt", truncation=True)
                            with torch.no_grad():
                                outputs = model(**inputs)
                            predictions = torch.argmax(outputs.logits, dim=-1)
                            predicted_labels = [model.config.id2label[pred.item()] for pred in predictions[0]]
                            results.append({"input": line, "lemmas": " ".join(predicted_labels[1:-1])})
                    
                    st.success("Complete!")
                    for idx, result in enumerate(results, 1):
                        with st.expander(f"Result {idx}"):
                            st.write("**Input:**", result['input'])
                            st.write("**Lemmas:**", result['lemmas'])

with tab3:
    st.header("Model Information")
    st.subheader("Files in Directory")
    files = os.listdir(model_path)
    model_files = [f for f in files if f.endswith(('.json', '.bin', '.safetensors', '.txt'))]
    for f in sorted(model_files):
        st.text(f"📄 {f}")

st.markdown("---")
st.caption("Arabic Lemmatization System | Powered by Transformers")
