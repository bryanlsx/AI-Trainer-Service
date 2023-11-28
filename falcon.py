import streamlit as st
from pdfminer.high_level import extract_text
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftConfig, PeftModel, prepare_model_for_kbit_training

# Model loading function
def load_models():
    # Configuration for your Falcon model
    trained_model_dir = './trained_falcon_5kdialog'
    config = PeftConfig.from_pretrained(trained_model_dir, load_in_8bit_fp32_cpu_offload=True)

    # BitsAndBytesConfig setup
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        "ybelkada/falcon-7b-sharded-bf16", 
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map='auto'
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # Load the custom PeftModel
    trained_model = PeftModel.from_pretrained(model, trained_model_dir)

    # Load the tokenizer
    trained_model_tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)
    trained_model_tokenizer.pad_token = trained_model_tokenizer.eos_token

    return trained_model, trained_model_tokenizer

# Load models
trained_model, trained_model_tokenizer = load_models()

def get_response_from_model(model, tokenizer, query, context, device='cuda'):
    # Combine the context and the query to form the full input text
    input_text = f"{context}<Human>: {query}\n<Assistant>:"

    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

    # Generate a response from the model
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_length=len(input_ids[0]) + 250,  # Adjust based on your needs
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    # Decode the generated tokens to a string
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the response part (if necessary, depending on your formatting)
    response_parts = response.split("<Assistant>:")
    if len(response_parts) > 1:
        return response_parts[1].split("<Human>:")[0].strip()
    else:
        return response_parts[0].strip()


# Load models (ensure this is done in a way that it's executed only once)
models = load_models()

# Initialize session state for feedback and PDF context
if 'feedback_dict' not in st.session_state:
    st.session_state.feedback_dict = {}
if 'pdf_context' not in st.session_state:
    st.session_state.pdf_context = ''

st.title("PDF Query Chatbot")

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file is not None:
    # Process the PDF file
    st.session_state.pdf_context = extract_text(uploaded_file)

# User query input
user_query = st.text_input("Ask a question about the PDF")

# Display model responses
if user_query:
    st.subheader("Responses from Models")
    for model_name, model in models.items():
        # Check for feedback first
        if user_query in st.session_state.feedback_dict:
            response = st.session_state.feedback_dict[user_query]
        else:
            response = get_response_from_model(model, user_query, st.session_state.pdf_context)
        st.write(f"{model_name} response:", response)

# Feedback input
user_feedback = st.text_input("Provide your own answer if not satisfied (optional)")
if user_feedback:
    st.session_state.feedback_dict[user_query] = user_feedback
    st.success("Feedback saved for this session.")
