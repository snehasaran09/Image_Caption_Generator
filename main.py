import streamlit as st
from PIL import Image
from tqdm import tqdm
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
import urllib.request
from itertools import cycle
import os

model = VisionEncoderDecoderModel.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


max_length = 25
num_beams = 5
num_return_sequences = 1
temperature = 1.0
gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "num_return_sequences": num_return_sequences, "temperature": temperature, "do_sample": True}

def show_sample_images():
    filteredImages = {'First': 'sample_images/Image1.png',
                      'Second': 'sample_images/Image2.png', 'Third': 'sample_images/Image3.png'}

    cols = cycle(st.columns(3))
    for filteredImage in filteredImages.values():
        next(cols).image(filteredImage, width=200,)
    for i, filteredImage in enumerate(filteredImages.values()):
        if next(cols).button("Predict Caption", key=i):
            predicted_captions = predict_step([filteredImage], False)
            st.write(str(i + 1) + '. ' + predicted_captions[0])


def image_uploader():
    with st.form("uploader"):
        images = st.file_uploader("Upload Images", accept_multiple_files=True, type=[
                                  "png", "jpg", "jpeg"])

        submitted = st.form_submit_button("Submit")
        if submitted:
            if not images:
                return None
            predicted_captions = predict_step(images, False)
            cols = cycle(st.columns(3))
            for i, caption in enumerate(predicted_captions):
                caption = str(i+1)+'. '+caption
                next(cols).image(images[i], width=200, caption=caption)


def images_url():
    with st.form("url"):
        urls = st.text_input(
            'Enter URL of Images followed by comma for multiple URLs')
        images = urls.split(',')
        submitted = st.form_submit_button("Submit")
        if submitted:
            predicted_captions = predict_step(images, True)
            for i, caption in enumerate(predicted_captions):
                st.write(str(i+1)+'. '+caption)


def main():
    st.set_page_config(page_title="Image Caption", page_icon="🖼️")
    st.title("Image Caption Generator")
    st.header('Welcome to Image Caption Generator!')
    st.code('This is a simple Image Caption Generator built using HuggingFace Transformers and Streamlit built by Sneha Saravanan')
    st.write(
        'Visit the [Github]() repo for detailed exaplaination and to get started right away')
    tab1, tab2, tab3 = st.tabs(
        ["Sample Images", "Image from computer", "Image from URL"])
    with tab1:
        show_sample_images()
    with tab2:
        image_uploader()
    with tab3:
        images_url()


def predict_step(images_list, is_url):
    images = []
    for image in tqdm(images_list):
        if is_url:
            urllib.request.urlretrieve(image, "file.jpg")
            i_image = Image.open("file.jpg")

        else:
            i_image = Image.open(image)

        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    pixel_values = feature_extractor(
        images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    if is_url:
        os.remove('file.jpg')
    return preds


if __name__ == '__main__':
    main()
