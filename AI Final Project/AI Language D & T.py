import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

def language_detection():
    data_identify = pd.read_csv("D:\\Semesters Data\\Semster 5\\Lab\\AI LABS\\AI Project\\dataset.csv")

    x_identify = np.array(data_identify["Text"])
    y_identify = np.array(data_identify["language"])

    cv_identify = CountVectorizer()
    X_identify = cv_identify.fit_transform(x_identify)
    X_identify_train, X_identify_test, y_identify_train, y_identify_test = train_test_split(X_identify, y_identify, test_size=0.33, random_state=42)

    language_model = MultinomialNB()
    language_model.fit(X_identify_train, y_identify_train)

    # Evaluate the language identification model
    accuracy_identify = language_model.score(X_identify_test, y_identify_test)
    print(f'Language Identification Model Accuracy: {accuracy_identify}')

    # User input for language identification
    user_input_identify = input("Enter a Text for language identification: ")
    data_identify_input = cv_identify.transform([user_input_identify]).toarray()
    output_identify = language_model.predict(data_identify_input)
    print(f'Identified Language: {output_identify[0]}')

def language_translation():
    translation_data = pd.read_csv("D:\\Semesters Data\\Semster 5\\Lab\\AI LABS\AI Project\\Project\\data1.csv")  # Corrected typo in file extension

    X_translate = np.array(translation_data['English'])
    y_translate = np.array(translation_data['French'])

    label_encoder = LabelEncoder()
    y_translate_encoded = label_encoder.fit_transform(y_translate)

    # Use a subset for training due to the large dataset
    X_translate_train, _, y_translate_train, _ = train_test_split(X_translate, y_translate_encoded, test_size=0.99, random_state=42)

    cv_translate = CountVectorizer()
    X_translate_train_vectorized = cv_translate.fit_transform(X_translate_train)

    translation_model = MultinomialNB()
    translation_model.fit(X_translate_train_vectorized, y_translate_train)

    print("Training completed.")

    # Evaluate the translation model on the entire dataset
    accuracy_translate = translation_model.score(X_translate_train_vectorized, y_translate_train)
    print(f'Translation Model Accuracy: {accuracy_translate}')

    user_input_translate = input("Enter an English sentence for translation: ")
    user_input_translate_vectorized = cv_translate.transform([user_input_translate])
    output_translate_label = translation_model.predict(user_input_translate_vectorized)[0]

    predicted_french_sentence = label_encoder.inverse_transform([output_translate_label])[0]
    print(f'Predicted French Translation: {predicted_french_sentence}')
    
def language_translation2():
    translation_data = pd.read_excel("D:\\Semesters Data\\Semster 5\\Lab\\AI LABS\\AI Project\\Project\\EU.xlsx") # Corrected typo in file extension

    X_translate = np.array(translation_data['English'])
    y_translate = np.array(translation_data['Urdu'])

    label_encoder = LabelEncoder()
    y_translate_encoded = label_encoder.fit_transform(y_translate)

    # Use a subset for training due to the large dataset
    X_translate_train, _, y_translate_train, _ = train_test_split(X_translate, y_translate_encoded, test_size=0.42, random_state=42)

    cv_translate = CountVectorizer()
    X_translate_train_vectorized = cv_translate.fit_transform(X_translate_train)

    translation_model = MultinomialNB()
    translation_model.fit(X_translate_train_vectorized, y_translate_train)

    print("Training completed.")

    # Evaluate the translation model on the entire dataset
    accuracy_translate = translation_model.score(X_translate_train_vectorized, y_translate_train)
    print(f'Translation Model Accuracy: {accuracy_translate}')

    user_input_translate = input("Enter an English sentence for translation: ")
    user_input_translate_vectorized = cv_translate.transform([user_input_translate])
    output_translate_label = translation_model.predict(user_input_translate_vectorized)[0]

    predicted_french_sentence = label_encoder.inverse_transform([output_translate_label])[0]
    print(f'Predicted Urdu Translation: {predicted_french_sentence}')
# Prompt user for choice

while True:
    # Prompt user for choice
    user_choice = input("Enter \n 1- for language detection, \n 2- for English to French translation \n 3- for English to Urdu translation\n 4- to exit: ").lower()

    if user_choice == '4':
        print("Exiting the program. Goodbye!")
        break
    elif user_choice in ['1', '2', '3']:
        if user_choice == '1':
            language_detection()
        elif user_choice == '2':
            language_translation()
        elif user_choice == '3':
            language_translation2()
    else:
        print("Invalid choice. Please enter '1', '2', '3', or 'no'.")
